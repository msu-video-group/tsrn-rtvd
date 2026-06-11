"""
Losses used in recent deblurring literature.

Default blend: L1 + Charbonnier (as used in e.g. MIMO-UNet, MPRNet, etc.)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CharbonnierLoss(nn.Module):
    """
    Charbonnier / pseudo-Huber loss:  sqrt( (x - y)^2 + eps^2 )

    Smoother than L1 near zero → better gradient signal in flat regions.
    """

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


class DeblurLoss(nn.Module):
    """
    Weighted sum of L1 and Charbonnier losses.

    Args:
        l1_w    : weight for the L1 term
        charb_w : weight for the Charbonnier term
        eps     : Charbonnier epsilon
    """

    def __init__(self, l1_w: float = 0.5, charb_w: float = 0.5, eps: float = 1e-3):
        super().__init__()
        self.l1_w = l1_w
        self.charb_w = charb_w
        self.l1 = nn.L1Loss()
        self.charb = CharbonnierLoss(eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.l1_w * self.l1(pred, target) + self.charb_w * self.charb(
            pred, target
        )


def build_loss(cfg) -> DeblurLoss:
    lc = cfg.loss
    return DeblurLoss(
        l1_w=lc.l1_weight,
        charb_w=lc.charbonnier_weight,
        eps=lc.charbonnier_eps,
    )
