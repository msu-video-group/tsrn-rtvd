"""
Shared building blocks for tsrnn_e30, tsrnn_bt1, tsrnn_w2.

Conventions (per spec):
  • Activation: LeakyReLU(0.1, inplace=True) everywhere
  • No BN / IN / LN
  • All Conv2d with bias=True
  • P = 4 pose vector [dx, dy, dtheta, dzoom];
    only (dx, dy) used for spatial shift — full P fed to MLPs
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────
# Activation shorthand
# ─────────────────────────────────────────────────────────────────


def lrelu() -> nn.LeakyReLU:
    return nn.LeakyReLU(0.1, inplace=True)


# ─────────────────────────────────────────────────────────────────
# RB – Residual Block
#   Conv3x3 → LReLU → Conv3x3 → residual add
# ─────────────────────────────────────────────────────────────────


class RB(nn.Module):
    def __init__(self, C: int):
        super().__init__()
        self.conv1 = nn.Conv2d(C, C, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(C, C, 3, 1, 1, bias=True)
        self.act = lrelu()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.act(self.conv1(x)))


def make_rb_stack(C: int, n: int) -> nn.Sequential:
    return nn.Sequential(*[RB(C) for _ in range(n)])


# ─────────────────────────────────────────────────────────────────
# roll_and_zero – integer roll shift, zero-fill wrapped borders
#   dx > 0 → shift content RIGHT   (roll dim=3 by +sx)
#   dy > 0 → shift content DOWN    (roll dim=2 by +sy)
# ─────────────────────────────────────────────────────────────────

_MAX_ROLL = 32  # cap to avoid wrapping entire feature map


def roll_and_zero(
    feat: torch.Tensor,  # (B, C, H, W)
    sx_vec: np.ndarray,  # (B,) float integer shifts (cols)
    sy_vec: np.ndarray,  # (B,) float integer shifts (rows)
) -> torch.Tensor:
    B, C, H, W = feat.shape
    out = feat.clone()
    for b in range(B):
        sx = int(np.clip(round(float(sx_vec[b])), -_MAX_ROLL, _MAX_ROLL))
        sy = int(np.clip(round(float(sy_vec[b])), -_MAX_ROLL, _MAX_ROLL))
        if sx == 0 and sy == 0:
            continue
        r = torch.roll(feat[b], shifts=(sy, sx), dims=(1, 2))
        if sx > 0:
            r[:, :, :sx] = 0
        elif sx < 0:
            r[:, :, sx:] = 0
        if sy > 0:
            r[:, :sy, :] = 0
        elif sy < 0:
            r[:, sy:, :] = 0
        out[b] = r
    return out


# ─────────────────────────────────────────────────────────────────
# ShiftWarp
#   1. Integer roll shift at feature scale
#   2. Subpixel bilinear correction ∈ [-0.5, 0.5] from pose MLP
#
# scale: spatial downscale factor relative to full resolution
#        (4 for H/4, 8 for H/8)
# P:     pose vector dimension (default 4: dx, dy, dθ, dzoom)
# ─────────────────────────────────────────────────────────────────


class ShiftWarp(nn.Module):
    def __init__(self, scale: int, P: int = 4):
        super().__init__()
        self.scale = scale
        # SubPixMLP_s: P→16→2, zero-init → starts as pure integer shift
        self.subpix = nn.Sequential(
            nn.Linear(P, 16, bias=True),
            lrelu(),
            nn.Linear(16, 2, bias=True),
        )
        nn.init.zeros_(self.subpix[-1].weight)
        nn.init.zeros_(self.subpix[-1].bias)

    def forward(self, feat: torch.Tensor, dp: torch.Tensor) -> torch.Tensor:
        """
        feat: (B, C, H, W)   feature map at 1/scale resolution
        dp:   (B, P)         pose vector; dp[:,0]=dx, dp[:,1]=dy (full-res px)
        """
        # 1. Compute integer shifts at feature scale
        dp_np = dp.detach().cpu().numpy()  # (B, P)
        sx_vec = dp_np[:, 0] / self.scale  # (B,) col shift
        sy_vec = dp_np[:, 1] / self.scale  # (B,) row shift

        shifted = roll_and_zero(feat, sx_vec, sy_vec)

        # 2. Subpixel correction: 0.5 * tanh(MLP(dp)) ∈ [-0.5, 0.5] feature-pixels
        delta = 0.5 * torch.tanh(self.subpix(dp))  # (B, 2): (δx, δy)

        # 3. Bilinear translation warp
        return _bilinear_translate(shifted, delta)


def _bilinear_translate(feat: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """
    feat:  (B, C, H, W)
    delta: (B, 2)  (δx, δy) in feature-space pixels
           δx > 0 → shift content right
           δy > 0 → shift content down
    Output[r][c] = Input[r - δy][c - δx]  (zero-padded at borders)
    """
    B, C, H, W = feat.shape
    # Normalise pixel offset to grid_sample [-1,1] scale
    dx_n = (delta[:, 0] * 2.0 / max(W, 1)).view(B, 1, 1, 1)
    dy_n = (delta[:, 1] * 2.0 / max(H, 1)).view(B, 1, 1, 1)

    ly = torch.linspace(-1.0, 1.0, H, device=feat.device, dtype=feat.dtype)
    lx = torch.linspace(-1.0, 1.0, W, device=feat.device, dtype=feat.dtype)
    gy, gx = torch.meshgrid(ly, lx, indexing="ij")  # (H, W)

    # To shift content right by δx: sample from x - δx → subtract from grid
    gx_s = gx.unsqueeze(0).expand(B, -1, -1) - dx_n.expand(B, H, W, 1)[..., 0]
    gy_s = gy.unsqueeze(0).expand(B, -1, -1) - dy_n.expand(B, H, W, 1)[..., 0]
    grid = torch.stack([gx_s, gy_s], dim=-1)  # (B, H, W, 2)

    return F.grid_sample(
        feat, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )


# ─────────────────────────────────────────────────────────────────
# PoseEmbed
#   MLP: (B, P) → (B, out_ch)   [caller broadcasts spatially]
# ─────────────────────────────────────────────────────────────────


class PoseEmbed(nn.Module):
    def __init__(self, P: int, out_ch: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(P, out_ch // 2, bias=True),
            lrelu(),
            nn.Linear(out_ch // 2, out_ch, bias=True),
        )

    def forward(self, dp: torch.Tensor) -> torch.Tensor:
        """dp: (B, P) → (B, out_ch)."""
        return self.mlp(dp)

    def as_map(
        self,
        dp: torch.Tensor,  # (B, P)
        feat: torch.Tensor,  # used only for shape (B, _, H, W)
    ) -> torch.Tensor:  # (B, out_ch, H, W)
        B, C, H, W = feat.shape
        e = self.forward(dp)  # (B, out_ch)
        return e.view(B, -1, 1, 1).expand(B, e.shape[1], H, W)


# ─────────────────────────────────────────────────────────────────
# UpBlock(Cin, Cskip, Cout, nRB)
#   1. bilinear ×2
#   2. Conv3x3(Cin → Cout) + LReLU
#   3. cat with skip                 → (Cout + Cskip)
#   4. Conv3x3(Cout+Cskip → Cout) + LReLU
#   5. nRB × RB(Cout)
# ─────────────────────────────────────────────────────────────────


class UpBlock(nn.Module):
    def __init__(self, Cin: int, Cskip: int, Cout: int, nRB: int):
        super().__init__()
        self.up_conv = nn.Conv2d(Cin, Cout, 3, 1, 1, bias=True)
        self.fuse = nn.Conv2d(Cout + Cskip, Cout, 3, 1, 1, bias=True)
        self.act = lrelu()
        self.blocks = make_rb_stack(Cout, nRB)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.act(self.up_conv(x))
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.fuse(x))
        return self.blocks(x)


# ─────────────────────────────────────────────────────────────────
# SharedEncoder
#   Input: (B, 3, H, W)
#   Outputs:
#     X2: (B, 32, H/2, W/2)   stem feature (skip for decoder)
#     F4: (B, 64, H/4, W/4)
#     F8: (B, 96, H/8, W/8)
# ─────────────────────────────────────────────────────────────────


class SharedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=True),
            lrelu(),
        )
        self.level4 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=True),
            lrelu(),
            *[RB(64) for _ in range(4)],
        )
        self.level8 = nn.Sequential(
            nn.Conv2d(64, 96, 3, stride=2, padding=1, bias=True),
            lrelu(),
            *[RB(96) for _ in range(6)],
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        X2 = self.stem(x)  # (B, 32, H/2, W/2)
        F4 = self.level4(X2)  # (B, 64, H/4, W/4)
        F8 = self.level8(F4)  # (B, 96, H/8, W/8)
        return X2, F4, F8


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────


def pad_dp(dp: torch.Tensor, P: int = 4) -> torch.Tensor:
    """Pad 2-D dataset displacement to P dims with zeros."""
    if dp.shape[-1] >= P:
        return dp[..., :P]
    return F.pad(dp, (0, P - dp.shape[-1]), value=0.0)


def softmax_weights(logits: torch.Tensor) -> list[torch.Tensor]:
    """logits: (B, N, H, W) → N tensors of (B, 1, H, W) that sum to 1."""
    w = F.softmax(logits, dim=1)
    return list(w.unbind(dim=1))  # each (B, H, W)


def weighted_sum(weights: list, feats: list) -> torch.Tensor:
    """weights: list of (B, H, W); feats: list of (B, C, H, W)."""
    out = weights[0].unsqueeze(1) * feats[0]
    for w, f in zip(weights[1:], feats[1:]):
        out = out + w.unsqueeze(1) * f
    return out
