# LiFT: Lifted Inter-Slice Feature Trajectories

Model definitions for *Efficient Factorized Volume Synthesis with Lifted Inter-Slice Feature Trajectories* (NeurIPS 2026). Requires PyTorch.

Volumes are `(B, C, H, W, D)` tensors, with `D` the through-plane axis.

## LiFT-U (`lift/lift_u.py`, Section 4.1)

| Class | Paper | Role |
|---|---|---|
| `UNetGenerator` | G_2D, Eq. 1 | 2D slice generator conditioned through adaptive GroupNorm; frozen |
| `DepthMapper` | M_φ, Eq. 2 | global latent ξ + Fourier depth encoding γ(d) → per-slice codes c_d |
| `LiFTU` | Eq. 3 | `LiFTU()(xi)` → `(B, 1, H, W, D)` |

## LiFT-C (`lift/lift_c.py`, Section 4.2)

| Class | Paper | Role |
|---|---|---|
| `UNet2D` | E_θ, D_θ, Eq. 7, 9 | 2D encoder–decoder exposing the spatial bottleneck |
| `SpatialBiGRUMixer` | M_φC, Eq. 8 | per-location bidirectional GRU along depth |
| `LiFTC` | Section 5.2 | missing-MR synthesis: `(B, 3, H, W, D)` → `(B, 1, H, W, D)` |
| `SliceTranslator`, `ContextRefiner`, `DepthBiGRU` | Section 5.3 | frozen 2D MR-to-CT translator, and the context branch predicting its residual |
| `LiFTCResidual` | Section 5.3 | MR-to-CT synthesis: `(B, 1, H, W, D)` → `(B, 1, H, W, D)` |

```python
import torch
from lift import LiFTU, LiFTC

volume = LiFTU()(torch.randn(1, 256))                 # (1, 1, 128, 128, 78)
target = LiFTC()(torch.randn(1, 3, 240, 240, 155))    # (1, 1, 240, 240, 155)
```
