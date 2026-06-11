from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import load_config
from .dataset import build_dataset, collate_samples
from .preprocess import FixedDerivativePreprocess
from .utils import (
    configure_torch_for_speed,
    device_auto,
    mkdir,
    move_batch_to_device,
    seed_everything,
)


@torch.no_grad()
def compute_stats(cfg, device: torch.device) -> dict:
    ds = build_dataset(cfg, cfg.data.train_split, train=True)
    loader = DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(cfg.train.num_workers > 0 and cfg.train.persistent_workers),
        prefetch_factor=cfg.train.prefetch_factor
        if cfg.train.num_workers > 0
        else None,
        collate_fn=collate_samples,
    )
    preprocess = FixedDerivativePreprocess().to(device).eval()

    x_sum = torch.zeros(3, dtype=torch.float64, device=device)
    x_sumsq = torch.zeros(3, dtype=torch.float64, device=device)
    x_count = 0
    y_sum = torch.zeros(3, dtype=torch.float64, device=device)
    y_sumsq = torch.zeros(3, dtype=torch.float64, device=device)
    y_count = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        x = preprocess(batch["x_rgb"]).to(torch.float64)  # [B,K,3,H,W]
        y = batch["target"].to(torch.float64)  # [B,K-1,3]

        x_sum += x.sum(dim=(0, 1, 3, 4))
        x_sumsq += (x * x).sum(dim=(0, 1, 3, 4))
        x_count += x.shape[0] * x.shape[1] * x.shape[3] * x.shape[4]

        y_sum += y.sum(dim=(0, 1))
        y_sumsq += (y * y).sum(dim=(0, 1))
        y_count += y.shape[0] * y.shape[1]

    x_mean = x_sum / max(x_count, 1)
    x_var = (x_sumsq / max(x_count, 1) - x_mean.square()).clamp_min(1e-12)
    y_mean = y_sum / max(y_count, 1)
    y_var = (y_sumsq / max(y_count, 1) - y_mean.square()).clamp_min(1e-12)

    return {
        "preprocess_mean": x_mean.float().cpu(),
        "preprocess_std": torch.sqrt(x_var).float().cpu(),
        "delta_mean": y_mean.float().cpu(),
        "delta_std": torch.sqrt(y_var).float().cpu(),
        "num_windows": len(ds),
        "K": cfg.data.K,
        "height": cfg.data.height,
        "width": cfg.data.width,
        "target_frame": cfg.data.target_frame,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute TinyDelta-3D preprocessing and target statistics."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.train.seed)
    configure_torch_for_speed()
    device = device_auto() if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    stats = compute_stats(cfg, device)
    path = Path(cfg.data.stats_path)
    mkdir(path.parent)
    torch.save(stats, path)
    print(f"saved stats: {path}")
    print(f"windows: {stats['num_windows']}")
    print(f"preprocess_mean: {stats['preprocess_mean'].tolist()}")
    print(f"preprocess_std:  {stats['preprocess_std'].tolist()}")
    print(f"delta_mean:      {stats['delta_mean'].tolist()}")
    print(f"delta_std:       {stats['delta_std'].tolist()}")


if __name__ == "__main__":
    main()
