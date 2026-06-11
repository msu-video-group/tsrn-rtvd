"""
Evaluation metrics and timing helpers.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────────
# PSNR
# ─────────────────────────────────────────────────────────────────────────────


def compute_psnr(
    pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0
) -> float:
    """
    Compute PSNR (dB) between two image batches.
    pred, target: (B, C, H, W) float32 in [0, max_val].
    Returns mean PSNR across the batch.

    Defensive: any NaN/Inf in pred (from fp16 overflow at early training)
    is replaced with 0 before computing MSE so one bad frame does not
    poison the running PSNRMeter average.
    """
    with torch.no_grad():
        pred_s = torch.nan_to_num(pred, nan=0.0, posinf=max_val, neginf=0.0)
        mse = torch.mean((pred_s - target) ** 2, dim=[1, 2, 3])  # (B,)
        mse = mse.clamp(min=1e-10)
        # .new_tensor() keeps same device/dtype as mse — never a cross-device division
        psnr = 10.0 * torch.log10(mse.new_tensor(max_val**2) / mse)
        return psnr.mean().item()


class PSNRMeter:
    """Running average PSNR."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._sum = 0.0
        self._count = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        val = compute_psnr(pred.float(), target.float())
        self._sum += val * pred.shape[0]
        self._count += pred.shape[0]

    @property
    def avg(self) -> float:
        return self._sum / max(self._count, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Timing  (standard pattern used in deblurring / low-level vision literature)
#
#   torch.cuda.synchronize()
#   t0 = time.perf_counter()
#   model(input)
#   torch.cuda.synchronize()
#   elapsed_ms = (time.perf_counter() - t0) * 1000
#
# The two synchronize() calls ensure the CPU waits for all queued CUDA kernels
# to finish before/after reading the clock — without them perf_counter() would
# capture only kernel-launch overhead, not actual compute time.
# ─────────────────────────────────────────────────────────────────────────────


def time_inference_ms(model: nn.Module, *args, **kwargs) -> tuple[float, torch.Tensor]:
    """
    Time a single model forward pass (GPU-only) in milliseconds.

    Returns (elapsed_ms, output).

    Usage in eval loop:
        ms, pred = time_inference_ms(model, blur, traj)
    """
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model(*args, **kwargs)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0, out


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self._sum = 0.0
        self._count = 0

    def update(self, val: float, n: int = 1):
        self._sum += val * n
        self._count += n

    @property
    def avg(self) -> float:
        return self._sum / max(self._count, 1)


# ─────────────────────────────────────────────────────────────────────────────
# GMACs computation
# ─────────────────────────────────────────────────────────────────────────────


def compute_gmacs(
    model: nn.Module,
    blur_shape: tuple,  # (B, K*3, H, W)
    traj_shape: tuple,  # (B, (K-1)*3)
    device: torch.device,
) -> float:
    """
    Returns GMACs (Giga Multiply-Accumulate ops) for a single forward pass.
    Requires the `thop` package (pip install thop).
    Falls back to a manual flop count if thop is unavailable.
    """
    try:
        from thop import profile

        dummy_blur = torch.zeros(*blur_shape).to(device)
        dummy_traj = torch.zeros(*traj_shape).to(device)
        macs, _ = profile(model, inputs=(dummy_blur, dummy_traj), verbose=False)
        return macs / 1e9  # convert to GMACs
    except ImportError:
        print(
            "[WARN] `thop` not installed – GMACs not computed. Install with: pip install thop"
        )
        return float("nan")
