"""LiFT-U: distributional lifting for unconditional volume synthesis (Section 4.1).

    c_d = M_phi(xi, gamma(d))              DepthMapper         (Eq. 2)
    v_d = G_2D(c_d; theta)                 UNetGenerator       (Eq. 1)
    V   = Stack_d v_d                      LiFTU               (Eq. 3)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _num_groups(num_channels, target=32):
    for g in range(min(target, num_channels), 0, -1):
        if num_channels % g == 0:
            return g
    return 1


# ── 2D slice generator G_2D ───────────────────────────────────────────────────

class AdaGN(nn.Module):
    """GroupNorm whose scale and shift are predicted from the conditioning vector."""

    def __init__(self, num_channels, ctx_dim):
        super().__init__()
        self.gn = nn.GroupNorm(_num_groups(num_channels), num_channels, affine=False)
        self.proj = nn.Linear(ctx_dim, num_channels * 2)
        self.proj.weight.data.zero_()
        self.proj.bias.data[:num_channels] = 1.0
        self.proj.bias.data[num_channels:] = 0.0

    def forward(self, x, ctx):
        scale, shift = self.proj(ctx)[:, :, None, None].chunk(2, dim=1)
        return self.gn(x) * scale + shift


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ctx_dim):
        super().__init__()
        self.adagn1 = AdaGN(in_ch, ctx_dim)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.adagn2 = AdaGN(out_ch, ctx_dim)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, ctx):
        h = self.conv1(F.silu(self.adagn1(x, ctx)))
        h = self.conv2(F.silu(self.adagn2(h, ctx)))
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(_num_groups(channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x)).reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = torch.einsum('bhdn,bhdm->bhnm', q, k) * (C // self.num_heads) ** -0.5
        out = torch.einsum('bhnm,bhdm->bhdn', attn.softmax(dim=-1), v)
        return x + self.proj(out.reshape(B, C, H, W))


class UNetGenerator(nn.Module):
    """G_2D: maps a conditioning vector c (B, ctx_dim) and spatial noise x (B, 1, H, W)
    to one axial slice in [-1, 1]. Conditioning enters every block through AdaGN."""

    def __init__(self, ctx_dim=512, base_ch=32, img_size=128):
        super().__init__()
        self.img_size = img_size
        ch = [base_ch * m for m in (1, 2, 4, 8, 16)]
        self.in_conv = nn.Conv2d(1, ch[0], 3, 1, 1)

        self.enc_blocks = nn.ModuleList()
        self.enc_downs = nn.ModuleList()
        for i in range(4):
            self.enc_blocks.append(ResBlock(ch[i], ch[i + 1], ctx_dim))
            self.enc_downs.append(nn.Conv2d(ch[i + 1], ch[i + 1], 3, 2, 1))

        self.mid_res1 = ResBlock(ch[4], ch[4], ctx_dim)
        self.mid_attn = SelfAttention2d(ch[4])
        self.mid_res2 = ResBlock(ch[4], ch[4], ctx_dim)

        self.dec_blocks = nn.ModuleList()
        self.dec_ups = nn.ModuleList()
        for i in range(4):
            self.dec_ups.append(nn.Upsample(scale_factor=2, mode='nearest'))
            self.dec_blocks.append(ResBlock(ch[4 - i] * 2, ch[3 - i], ctx_dim))

        self.out_conv = nn.Sequential(
            nn.GroupNorm(_num_groups(ch[0], 8), ch[0]),
            nn.SiLU(),
            nn.Conv2d(ch[0], 1, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, ctx, x=None):
        if x is None:
            x = torch.randn(ctx.shape[0], 1, self.img_size, self.img_size,
                            device=ctx.device, dtype=ctx.dtype)
        h = self.in_conv(x)
        skips = []
        for enc, down in zip(self.enc_blocks, self.enc_downs):
            h = enc(h, ctx)
            skips.append(h)
            h = down(h)
        h = self.mid_res2(self.mid_attn(self.mid_res1(h, ctx)), ctx)
        for up, dec, skip in zip(self.dec_ups, self.dec_blocks, reversed(skips)):
            h = dec(torch.cat([up(h), skip], dim=1), ctx)
        return self.out_conv(h)


# ── Depth mapper M_phi ────────────────────────────────────────────────────────

class FourierDepthPE(nn.Module):
    """gamma(d): random Fourier features of the depth coordinate d in [-1, 1]."""

    def __init__(self, n_freqs=16, scale=3.0):
        super().__init__()
        self.register_buffer('freqs', torch.randn(n_freqs) * scale)

    def forward(self, depth):
        x = depth * self.freqs * math.pi
        return torch.cat([x.sin(), x.cos()], dim=-1)


class DepthMapper(nn.Module):
    """M_phi: global latent xi (B, z_dim) -> per-slice conditioning vectors (B, D, ctx_dim).

    An MLP maps [xi, gamma(d)] to a per-slice code; a broadcast anchor (the code at d = 0)
    is added; a residual 1D convolution along depth couples neighbouring slices; a final
    LayerNorm keeps the codes on the distribution the frozen G_2D was trained on."""

    def __init__(self, z_dim=256, ctx_dim=512, hidden_dim=512, n_layers=4, n_freqs=16,
                 depth_enc_dim=32, n_slices=78, anchor_weight=0.1,
                 conv_layers=4, conv_kernel=7, conv_hidden=64):
        super().__init__()
        self.n_slices = n_slices
        self.anchor_weight = anchor_weight

        self.pe = FourierDepthPE(n_freqs)
        self.depth_proj = nn.Linear(2 * n_freqs, depth_enc_dim)

        layers = [nn.Linear(z_dim + depth_enc_dim, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        last = nn.Linear(hidden_dim, ctx_dim)
        nn.init.normal_(last.weight, std=0.02)
        nn.init.zeros_(last.bias)
        self.slice_mapper = nn.Sequential(*layers, last)

        convs, c_in = [], ctx_dim
        for i in range(conv_layers):
            c_out = conv_hidden if i < conv_layers - 1 else ctx_dim
            convs.append(nn.Conv1d(c_in, c_out, conv_kernel, padding=conv_kernel // 2))
            if i < conv_layers - 1:
                convs.append(nn.SiLU())
            c_in = c_out
        nn.init.zeros_(convs[-1].weight)
        nn.init.zeros_(convs[-1].bias)
        self.conv_refine = nn.Sequential(*convs)

        self.out_norm = nn.LayerNorm(ctx_dim)

    def forward(self, xi):
        B, D = xi.shape[0], self.n_slices
        depth = torch.linspace(-1.0, 1.0, D, device=xi.device).unsqueeze(-1)       # (D, 1)
        d_enc = self.depth_proj(self.pe(depth))                                     # (D, depth_enc_dim)
        base = self.slice_mapper(torch.cat([xi.unsqueeze(1).expand(B, D, -1),
                                            d_enc.unsqueeze(0).expand(B, -1, -1)], dim=-1))
        d0 = self.depth_proj(self.pe(torch.zeros(1, 1, device=xi.device)))
        anchor = self.slice_mapper(torch.cat([xi, d0.expand(B, -1)], dim=-1))
        h = (base + self.anchor_weight * anchor.unsqueeze(1)).transpose(1, 2)       # (B, ctx, D)
        h = (h + self.conv_refine(h)).transpose(1, 2)                                # (B, D, ctx)
        return self.out_norm(h)


# ── LiFT-U ────────────────────────────────────────────────────────────────────

class LiFTU(nn.Module):
    """V = Stack_d G_2D(M_phi(xi, gamma(d)); theta). G_2D is frozen; only M_phi is trained."""

    def __init__(self):
        super().__init__()
        self.generator = UNetGenerator()
        self.depth_mapper = DepthMapper()
        self.generator.requires_grad_(False)

    def forward(self, xi, noise=None):
        """xi: (B, z_dim); noise: per-slice spatial noise (B, D, 1, H, W), sampled from N(0, I)
        when not given. Returns the volume (B, 1, H, W, D)."""
        ctx = self.depth_mapper(xi)                                                  # (B, D, ctx)
        B, D, _ = ctx.shape
        x = None if noise is None else noise.flatten(0, 1)
        slices = self.generator(ctx.flatten(0, 1), x)                                # (B*D, 1, H, W)
        return slices.reshape(B, D, *slices.shape[1:]).permute(0, 2, 3, 4, 1)
