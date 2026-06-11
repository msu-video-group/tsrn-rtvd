from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class FixedDerivativePreprocess(nn.Module):
    """Non-learned preprocessing: RGB -> gray, Sobel magnitude, Laplacian.

    Input and output are five-dimensional sequence tensors:
      x_rgb: [B, K, 3, H, W], float in [0, 1]
      out:   [B, K, 3, H, W]
    """

    def __init__(
        self,
        mean: Sequence[float] | torch.Tensor | None = None,
        std: Sequence[float] | torch.Tensor | None = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        rgb_to_luma = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32).view(
            1, 3, 1, 1
        )
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        lap = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("rgb_to_luma", rgb_to_luma, persistent=False)
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)
        self.register_buffer("lap_kernel", lap, persistent=False)
        mean_t = (
            torch.zeros(3, dtype=torch.float32)
            if mean is None
            else torch.as_tensor(mean, dtype=torch.float32)
        )
        std_t = (
            torch.ones(3, dtype=torch.float32)
            if std is None
            else torch.as_tensor(std, dtype=torch.float32)
        )
        self.register_buffer("mean", mean_t.view(1, 1, 3, 1, 1), persistent=True)
        self.register_buffer(
            "std", std_t.view(1, 1, 3, 1, 1).clamp_min(eps), persistent=True
        )
        self.eps = eps

    def set_stats(
        self, mean: Sequence[float] | torch.Tensor, std: Sequence[float] | torch.Tensor
    ) -> None:
        mean_t = torch.as_tensor(
            mean, dtype=self.mean.dtype, device=self.mean.device
        ).view(1, 1, 3, 1, 1)
        std_t = torch.as_tensor(std, dtype=self.std.dtype, device=self.std.device).view(
            1, 1, 3, 1, 1
        )
        self.mean.copy_(mean_t)
        self.std.copy_(std_t.clamp_min(self.eps))

    @torch.no_grad()
    def forward(self, x_rgb: torch.Tensor) -> torch.Tensor:
        if x_rgb.ndim != 5:
            raise ValueError(f"Expected [B,K,3,H,W], got {tuple(x_rgb.shape)}")
        b, k, c, h, w = x_rgb.shape
        if c != 3:
            raise ValueError(f"Expected RGB channel count 3, got {c}")
        x = x_rgb.reshape(b * k, c, h, w)
        gray = (x * self.rgb_to_luma.to(dtype=x.dtype)).sum(dim=1, keepdim=True)
        gx = F.conv2d(gray, self.sobel_x.to(dtype=x.dtype), padding=1)
        gy = F.conv2d(gray, self.sobel_y.to(dtype=x.dtype), padding=1)
        sobel_mag = torch.sqrt(gx.square() + gy.square() + self.eps)
        lap = F.conv2d(gray, self.lap_kernel.to(dtype=x.dtype), padding=1)
        out = torch.cat([gray, sobel_mag, lap], dim=1).reshape(b, k, 3, h, w)
        return (out - self.mean.to(dtype=out.dtype)) / self.std.to(dtype=out.dtype)
