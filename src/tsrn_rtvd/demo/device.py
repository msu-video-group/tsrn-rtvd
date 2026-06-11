from __future__ import annotations

from contextlib import nullcontext
from typing import ContextManager

import torch


def resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested, but PyTorch cannot use it.")
        return torch.device("mps")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but PyTorch cannot use it.")
        return torch.device("cuda")

    if requested == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unknown device: {requested}")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def autocast_context(device: torch.device, enabled: bool) -> ContextManager:
    if not enabled or device.type == "cpu":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16)
