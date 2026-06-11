from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class LossOutput:
    total: torch.Tensor
    regression: torch.Tensor
    uncertainty: torch.Tensor
    antisymmetry: torch.Tensor
    mae_norm: torch.Tensor
    mae_raw: torch.Tensor


class DeltaStandardizer(torch.nn.Module):
    def __init__(self, mean=None, std=None, eps: float = 1e-8) -> None:
        super().__init__()
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
        self.register_buffer("mean", mean_t.view(1, 1, 3), persistent=True)
        self.register_buffer("std", std_t.view(1, 1, 3).clamp_min(eps), persistent=True)
        self.eps = eps

    def set_stats(self, mean, std) -> None:
        self.mean.copy_(
            torch.as_tensor(mean, dtype=self.mean.dtype, device=self.mean.device).view(
                1, 1, 3
            )
        )
        self.std.copy_(
            torch.as_tensor(std, dtype=self.std.dtype, device=self.std.device)
            .view(1, 1, 3)
            .clamp_min(self.eps)
        )

    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.mean.to(dtype=y.dtype)) / self.std.to(dtype=y.dtype)

    def denormalize(self, y_norm: torch.Tensor) -> torch.Tensor:
        return y_norm * self.std.to(dtype=y_norm.dtype) + self.mean.to(
            dtype=y_norm.dtype
        )


def tinydelta_loss(
    pred_norm: torch.Tensor,
    log_var: torch.Tensor,
    target_raw: torch.Tensor,
    standardizer: DeltaStandardizer,
    use_uncertainty: bool = True,
    smooth_l1_beta: float = 1.0,
    uncertainty_reg: float = 0.05,
    log_var_min: float = -6.0,
    log_var_max: float = 3.0,
    antisymmetry_weight: float = 0.02,
) -> LossOutput:
    target_norm = standardizer.normalize(target_raw)
    err_axis = F.smooth_l1_loss(
        pred_norm, target_norm, beta=smooth_l1_beta, reduction="none"
    )
    regression = err_axis.mean()

    if use_uncertainty:
        s = torch.clamp(log_var, min=log_var_min, max=log_var_max)
        err_pair = err_axis.mean(dim=-1, keepdim=True)
        uncertainty = (torch.exp(-s) * err_pair + uncertainty_reg * s).mean()
        total = uncertainty
    else:
        uncertainty = pred_norm.new_zeros(())
        total = regression

    pred_raw = standardizer.denormalize(pred_norm)
    if pred_raw.shape[1] == 2:
        antisym = F.smooth_l1_loss(
            pred_raw[:, 0, :] + pred_raw[:, 1, :], torch.zeros_like(pred_raw[:, 0, :])
        )
        total = total + antisymmetry_weight * antisym
    else:
        antisym = pred_raw.new_zeros(())

    mae_norm = (pred_norm - target_norm).abs().mean()
    mae_raw = (pred_raw - target_raw).abs().mean()
    return LossOutput(
        total=total,
        regression=regression,
        uncertainty=uncertainty,
        antisymmetry=antisym,
        mae_norm=mae_norm,
        mae_raw=mae_raw,
    )
