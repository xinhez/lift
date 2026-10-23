"""LiFT-C: supervised lifting for paired translation (Section 4.2).

    h_d             = E_theta(x_d)                            (Eq. 7)
    (c_1, ..., c_D) = M_phiC(h_1:D)                           (Eq. 8)
    h~_d            = h_d + R_phi(c_d),  y_d = D_theta(h~_d)    (Eq. 9)

LiFTC           missing-MR synthesis (Section 5.2): 2D U-Net with a spatial BiGRU context module.
LiFTCResidual   MR-to-CT synthesis (Section 5.3): a frozen 2D translator predicts a baseline CT,
                and a context branch predicts an additive residual correction.

Volumes are (B, C, H, W, D) tensors; D is the through-plane axis.
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


def _fold(vol):
    """(B, C, H, W, D) -> (B*D, C, H, W)."""
    B, C, H, W, D = vol.shape
    return vol.permute(0, 4, 1, 2, 3).reshape(B * D, C, H, W)


def _unfold(slices, B):
    """(B*D, C, H, W) -> (B, C, H, W, D)."""
    return slices.reshape(B, -1, *slices.shape[1:]).permute(0, 2, 3, 4, 1)


# ── 2D encoder-decoder G_theta = D_theta ∘ E_theta ────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.norm1 = nn.GroupNorm(_num_groups(in_ch), in_ch)
        self.norm2 = nn.GroupNorm(_num_groups(out_ch), out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
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


class UNet2D(nn.Module):
    """2D encoder-decoder. `encode` returns the spatial bottleneck h_d and the skip features;
    `decode` maps the bottleneck back to a slice. When a context map c is given, `decode`
    first forms h~ = h + R(cat[h, c]) with the zero-initialised 1x1 convolution R.
    `ctx_proj` is part of the trained parameters and is not used by the spatial context path."""

    def __init__(self, in_ch=3, out_ch=1, base_ch=64, n_levels=4, ctx_dim=512, tanh_out=True):
        super().__init__()
        ch = [min(base_ch * 2 ** i, 512) for i in range(n_levels + 1)]
        self.ch = ch

        self.in_conv = nn.Conv2d(in_ch, ch[0], 3, 1, 1)
        self.enc_blocks = nn.ModuleList(ResBlock(ch[i], ch[i + 1]) for i in range(n_levels))
        self.enc_downs = nn.ModuleList(nn.Conv2d(ch[i + 1], ch[i + 1], 3, 2, 1) for i in range(n_levels))

        bn = ch[n_levels]
        self.mid_res1 = ResBlock(bn, bn)
        self.mid_attn = SelfAttention2d(bn)
        self.mid_res2 = ResBlock(bn, bn)

        if ctx_dim is not None:
            self.ctx_proj = nn.Sequential(nn.Linear(ctx_dim, bn), nn.SiLU())
            self.bottleneck_merge = nn.Conv2d(bn * 2, bn, 1)
            nn.init.zeros_(self.bottleneck_merge.weight)
            nn.init.zeros_(self.bottleneck_merge.bias)

        self.dec_ups = nn.ModuleList(nn.Upsample(scale_factor=2, mode='nearest') for _ in range(n_levels))
        self.dec_blocks = nn.ModuleList(ResBlock(ch[n_levels - i] * 2, ch[n_levels - i - 1])
                                        for i in range(n_levels))

        head = [nn.GroupNorm(_num_groups(ch[0], 8), ch[0]), nn.SiLU(), nn.Conv2d(ch[0], out_ch, 3, 1, 1)]
        self.out_conv = nn.Sequential(*head, nn.Tanh()) if tanh_out else nn.Sequential(*head)

    def encode(self, x):
        h = self.in_conv(x)
        skips = []
        for enc, down in zip(self.enc_blocks, self.enc_downs):
            h = enc(h)
            skips.append(h)
            h = down(h)
        return self.mid_res2(self.mid_attn(self.mid_res1(h))), skips

    def decode(self, h, skips, ctx=None):
        if ctx is not None:
            h = h + self.bottleneck_merge(torch.cat([h, ctx], dim=1))
        for up, dec, skip in zip(self.dec_ups, self.dec_blocks, reversed(skips)):
            h = dec(torch.cat([up(h), skip], dim=1))
        return self.out_conv(h)

    def forward(self, x, ctx=None):
        h, skips = self.encode(x)
        return self.decode(h, skips, ctx)


# ── Missing-MR synthesis ──────────────────────────────────────────────────────

class SinusoidalPosEnc(nn.Module):
    def __init__(self, dim, max_len=256):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.shape[1]]


class BiGRU(nn.Module):
    """Bidirectional GRU over a depth sequence (N, D, feat_dim) -> (N, D, ctx_dim),
    with sinusoidal absolute-position encodings added to its input."""

    def __init__(self, feat_dim=512, ctx_dim=512, hidden_dim=512, n_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(feat_dim, hidden_dim)
        self.pos_enc = SinusoidalPosEnc(hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim // 2, num_layers=n_layers,
                          batch_first=True, bidirectional=True)
        self.out_proj = nn.Linear(hidden_dim, ctx_dim)

    def forward(self, seq):
        h, _ = self.gru(self.pos_enc(self.input_proj(seq)))
        return self.out_proj(h)


class SpatialBiGRUMixer(nn.Module):
    """M_phiC: one BiGRU runs along depth independently at every bottleneck location.
    (B, D, C, h, w) -> (B, D, C, h, w)."""

    def __init__(self, feat_dim=512, ctx_dim=512, hidden_dim=512, n_layers=2):
        super().__init__()
        self.mix = BiGRU(feat_dim, ctx_dim, hidden_dim, n_layers)
        self.proj = nn.Sequential(nn.Conv2d(ctx_dim, ctx_dim, 1), nn.SiLU())

    def forward(self, f):
        B, D, C, h, w = f.shape
        seq = f.permute(0, 3, 4, 1, 2).reshape(B * h * w, D, C)
        ctx = self.mix(seq).reshape(B, h, w, D, -1).permute(0, 3, 4, 1, 2)
        return self.proj(ctx.reshape(B * D, -1, h, w)).reshape(B, D, -1, h, w)


class LiFTC(nn.Module):
    """Missing-MR synthesis: three source contrasts (B, 3, H, W, D) -> target (B, 1, H, W, D)."""

    def __init__(self, in_ch=3, out_ch=1, base_ch=64, n_levels=4, hidden_dim=512):
        super().__init__()
        self.unet = UNet2D(in_ch, out_ch, base_ch, n_levels)
        bn = self.unet.ch[n_levels]
        self.zmixer = SpatialBiGRUMixer(bn, bn, hidden_dim)

    def forward(self, source):
        B = source.shape[0]
        h, skips = self.unet.encode(_fold(source))                                    # (B*D, C, h, w)
        ctx = self.zmixer(h.reshape(B, -1, *h.shape[1:]))                             # (B, D, C, h, w)
        return _unfold(self.unet.decode(h, skips, ctx.flatten(0, 1)), B)


# ── MR-to-CT synthesis ────────────────────────────────────────────────────────

class _RandomFourierEmb(nn.Module):
    def __init__(self, n_freqs=16, scale=3.0):
        super().__init__()
        self.register_buffer('freqs', torch.randn(n_freqs) * scale)

    def forward(self, pos):
        x = pos.unsqueeze(1) * self.freqs.unsqueeze(0)
        return torch.cat([x.sin(), x.cos()], dim=1)


class _MREncoder(nn.Module):
    def __init__(self, feat_dim=224, base=32):
        super().__init__()
        ch = [base * m for m in (1, 2, 4, 8)]

        def blk(ci, co):
            return nn.Sequential(nn.Conv2d(ci, co, 3, 1, 1), nn.GroupNorm(_num_groups(co), co), nn.SiLU(),
                                 nn.Conv2d(co, co, 3, 1, 1), nn.GroupNorm(_num_groups(co), co), nn.SiLU())

        self.b1, self.p1 = blk(1, ch[0]), nn.MaxPool2d(2)
        self.b2, self.p2 = blk(ch[0], ch[1]), nn.MaxPool2d(2)
        self.b3, self.p3 = blk(ch[1], ch[2]), nn.MaxPool2d(2)
        self.b4 = blk(ch[2], ch[3])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(nn.Flatten(), nn.Linear(ch[3], feat_dim), nn.LayerNorm(feat_dim))

    def forward(self, x):
        x = self.p3(self.b3(self.p2(self.b2(self.p1(self.b1(x))))))
        return self.mlp(self.gap(self.b4(x)))


class _CtxNet(nn.Module):
    """Per-slice conditioning code from the MR slice and its normalised position."""

    def __init__(self, ctx=256, base=32, n_freqs=16):
        super().__init__()
        self.mr_encoder = _MREncoder(feat_dim=ctx - 2 * n_freqs, base=base)
        self.pos_emb = _RandomFourierEmb(n_freqs)
        self.fuse = nn.Sequential(nn.Linear(ctx, ctx), nn.LayerNorm(ctx))

    def forward(self, mr, pos):
        return self.fuse(torch.cat([self.mr_encoder(mr), self.pos_emb(pos)], dim=1))


class _CBAM2D(nn.Module):
    def __init__(self, c, r=16, k=7):
        super().__init__()
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                nn.Linear(c, max(1, c // r)), nn.ReLU(inplace=True),
                                nn.Linear(max(1, c // r), c), nn.Sigmoid())
        self.sa = nn.Sequential(nn.Conv2d(2, 1, k, padding=k // 2, bias=False), nn.Sigmoid())

    def forward(self, x):
        x = x * self.ca(x)[:, :, None, None]
        return x * self.sa(torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], 1))


class _AdaGN(nn.Module):
    def __init__(self, c, ctx=256):
        super().__init__()
        self.gn = nn.GroupNorm(_num_groups(c), c, affine=False)
        self.proj = nn.Linear(ctx, c * 2)
        self.proj.weight.data.zero_()
        self.proj.bias.data[:c] = 1.0
        self.proj.bias.data[c:] = 0.0

    def forward(self, x, ctx):
        scale, shift = self.proj(ctx)[:, :, None, None].chunk(2, dim=1)
        return self.gn(x) * scale + shift


class _Block(nn.Module):
    def __init__(self, ci, co, ctx=256):
        super().__init__()
        self.conv1, self.norm1 = nn.Conv2d(ci, co, 3, 1, 1), _AdaGN(co, ctx)
        self.conv2, self.norm2 = nn.Conv2d(co, co, 3, 1, 1), _AdaGN(co, ctx)

    def forward(self, x, ctx):
        return F.silu(self.norm2(self.conv2(F.silu(self.norm1(self.conv1(x), ctx))), ctx))


class _TranslatorUNet(nn.Module):
    def __init__(self, base=32, ctx=256, scale=1.75):
        super().__init__()
        ch = [base * m for m in (1, 2, 4, 8, 16)]
        self.pool = nn.MaxPool2d(2)
        self.scale = scale
        self.enc1, self.enc2 = _Block(1, ch[0], ctx), _Block(ch[0], ch[1], ctx)
        self.enc3, self.enc4 = _Block(ch[1], ch[2], ctx), _Block(ch[2], ch[3], ctx)
        self.bottleneck = _Block(ch[3], ch[4], ctx)
        self.cbam1, self.cbam2 = _CBAM2D(ch[0]), _CBAM2D(ch[1])
        self.cbam3, self.cbam4 = _CBAM2D(ch[2]), _CBAM2D(ch[3])

        def up(ic, oc):
            return nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                 nn.Conv2d(ic, oc, 3, 1, 1))

        self.up4, self.dec4 = up(ch[4], ch[3]), _Block(ch[3] * 2, ch[3], ctx)
        self.up3, self.dec3 = up(ch[3], ch[2]), _Block(ch[2] * 2, ch[2], ctx)
        self.up2, self.dec2 = up(ch[2], ch[1]), _Block(ch[1] * 2, ch[1], ctx)
        self.up1, self.dec1 = up(ch[1], ch[0]), _Block(ch[0] * 2, ch[0], ctx)
        self.out_conv = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x, ctx):
        e1 = self.enc1(x, ctx)
        e2 = self.enc2(self.pool(e1), ctx)
        e3 = self.enc3(self.pool(e2), ctx)
        e4 = self.enc4(self.pool(e3), ctx)
        d = self.bottleneck(self.pool(e4), ctx)
        d = self.dec4(torch.cat([self.up4(d), self.cbam4(e4)], 1), ctx)
        d = self.dec3(torch.cat([self.up3(d), self.cbam3(e3)], 1), ctx)
        d = self.dec2(torch.cat([self.up2(d), self.cbam2(e2)], 1), ctx)
        d = self.dec1(torch.cat([self.up1(d), self.cbam1(e1)], 1), ctx)
        return torch.tanh(self.out_conv(d)) * self.scale


class SliceTranslator(nn.Module):
    """Axial 2D MR-to-CT translator. The U-Net input is the high-pass MR slice (MR minus its
    Gaussian blur); a code from the MR slice and its normalised position conditions every block."""

    def __init__(self, base=32, ctx=256, n_freqs=16, scale=1.75, blur_sigma=6.0, blur_k=25):
        super().__init__()
        self.ctx_net = _CtxNet(ctx, base, n_freqs)
        self.unet = _TranslatorUNet(base, ctx, scale)
        g = torch.exp(-(torch.arange(blur_k).float() - blur_k // 2) ** 2 / (2 * blur_sigma ** 2))
        g = g / g.sum()
        self.register_buffer('blur', (g[None] * g[:, None]).view(1, 1, blur_k, blur_k), persistent=False)

    def forward(self, mr, pos):
        """mr: (N, 1, H, W) slices; pos: (N,) slice positions in [0, 1]."""
        high_pass = mr - F.conv2d(mr, self.blur, padding=self.blur.shape[-1] // 2)
        return self.unet(high_pass, self.ctx_net(mr, pos))


def fourier_depth(D, n_freqs, device):
    """gamma(d) for d = 0, ..., D-1 normalised to [0, 1], with octave frequencies 2^k * pi."""
    pos = torch.arange(D, device=device).float() / max(D - 1, 1)
    ang = pos[:, None] * (2.0 ** torch.arange(n_freqs, device=device).float() * math.pi)[None]
    return torch.cat([ang.sin(), ang.cos()], 1)                                       # (D, 2F)


class DepthBiGRU(nn.Module):
    """M_phiC: a single-layer BiGRU along depth at every bottleneck location, with gamma(d)
    concatenated to its input at each depth. (B, D, C, h, w) -> (B, D, C, h, w)."""

    def __init__(self, c=256, hidden=128, n_freqs=8):
        super().__init__()
        self.n_freqs = n_freqs
        self.norm = nn.GroupNorm(_num_groups(c, 8), c)
        self.gru = nn.GRU(c + 2 * n_freqs, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * hidden, c)
        self.gate = nn.Parameter(torch.tensor(1.0))

    def forward(self, f):
        B, D, C, h, w = f.shape
        x = self.norm(f.flatten(0, 1)).reshape(B, D, C, h, w)
        g = fourier_depth(D, self.n_freqs, f.device).to(f.dtype)
        x = torch.cat([x, g[None, :, :, None, None].expand(B, D, -1, h, w)], 2)
        out, _ = self.gru(x.permute(0, 3, 4, 1, 2).reshape(B * h * w, D, -1))
        ctx = self.head(out).reshape(B, h, w, D, C).permute(0, 3, 4, 1, 2)
        return self.gate * ctx


class ContextRefiner(nn.Module):
    """Residual correction from [MR, baseline CT] volumes (B, 2, H, W, D) -> (B, 1, H, W, D):
    a 2D U-Net whose bottleneck receives depth context through a zero-initialised 1x1 merge,
    scaled by a learnable gate."""

    def __init__(self, in_ch=2, base_ch=32, n_levels=3, hidden=128, n_freqs=8):
        super().__init__()
        self.unet = UNet2D(in_ch, 1, base_ch, n_levels, ctx_dim=None, tanh_out=False)
        nn.init.zeros_(self.unet.out_conv[2].weight)
        nn.init.zeros_(self.unet.out_conv[2].bias)
        bn = self.unet.ch[n_levels]
        self.mixer = DepthBiGRU(bn, hidden, n_freqs)
        self.merge = nn.Conv2d(2 * bn, bn, 1)
        nn.init.zeros_(self.merge.weight)
        nn.init.zeros_(self.merge.bias)
        self.gate = nn.Parameter(torch.tensor(1.0))

    def forward(self, desc):
        B = desc.shape[0]
        h, skips = self.unet.encode(_fold(desc))                                      # (B*D, C, h, w)
        ctx = self.mixer(h.reshape(B, -1, *h.shape[1:])).flatten(0, 1)
        h = h + self.merge(torch.cat([h, ctx], 1))
        return _unfold(self.gate * self.unet.decode(h, skips), B)


class LiFTCResidual(nn.Module):
    """MR-to-CT synthesis: MR (B, 1, H, W, D) -> CT (B, 1, H, W, D) in [-1, 1],
    CT = clamp(baseline + delta) with the 2D translator frozen."""

    def __init__(self):
        super().__init__()
        self.translator = SliceTranslator()
        self.refiner = ContextRefiner()
        self.translator.requires_grad_(False)

    def forward(self, mr):
        B, D = mr.shape[0], mr.shape[-1]
        pos = (torch.arange(D, device=mr.device).float() / max(D - 1, 1)).repeat(B)
        baseline = _unfold(self.translator(_fold(mr), pos).clamp(-1, 1), B)
        delta = self.refiner(torch.cat([mr, baseline], 1))
        return (baseline + delta).clamp(-1, 1)
