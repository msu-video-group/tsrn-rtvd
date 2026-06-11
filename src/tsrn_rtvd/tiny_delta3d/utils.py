from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch_for_speed() -> None:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        # Small depthwise models can be slower with excessive CPU thread oversubscription.
        try:
            torch.set_num_threads(min(torch.get_num_threads(), 4))
        except Exception:
            pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def device_auto() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mkdir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def load_stats(path: str | Path, map_location="cpu") -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Stats file not found: {path}. Run: python -m tinydelta3d.stats --config <config.yaml>"
        )
    return torch.load(path, map_location=map_location)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rem = seconds - 60 * minutes
    return f"{minutes}m{rem:.0f}s"
