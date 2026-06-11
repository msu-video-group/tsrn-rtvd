from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import load_config
from .dataset import build_dataset, collate_samples
from .losses import DeltaStandardizer, tinydelta_loss
from .model import build_model_from_config, count_parameters, switch_to_deploy
from .preprocess import FixedDerivativePreprocess
from .utils import (
    configure_torch_for_speed,
    device_auto,
    load_stats,
    move_batch_to_device,
    seed_everything,
)


@torch.no_grad()
def evaluate(
    cfg, ckpt_path: str, device: torch.device, split: str, deploy: bool = False
) -> dict:
    stats = load_stats(cfg.data.stats_path)
    preprocess = (
        FixedDerivativePreprocess(stats["preprocess_mean"], stats["preprocess_std"])
        .to(device)
        .eval()
    )
    standardizer = DeltaStandardizer(stats["delta_mean"], stats["delta_std"]).to(device)
    ds = build_dataset(cfg, split, train=False)
    loader = DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.train.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(cfg.train.num_workers > 0 and cfg.train.persistent_workers),
        prefetch_factor=cfg.train.prefetch_factor
        if cfg.train.num_workers > 0
        else None,
        collate_fn=collate_samples,
    )
    model = build_model_from_config(cfg).to(device).eval()
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    if deploy:
        switch_to_deploy(model)
    if cfg.train.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    totals = {
        "loss": 0.0,
        "mae_raw": 0.0,
        "mae_norm": 0.0,
        "axis_abs_sum": torch.zeros(3, device=device),
        "n_pairs": 0,
    }
    t0 = time.time()
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        x = preprocess(batch["x_rgb"])
        target = batch["target"]
        pred_norm, log_var = model(x)
        loss_out = tinydelta_loss(
            pred_norm,
            log_var,
            target,
            standardizer,
            use_uncertainty=cfg.loss.use_uncertainty,
            smooth_l1_beta=cfg.loss.smooth_l1_beta,
            uncertainty_reg=cfg.loss.uncertainty_reg,
            log_var_min=cfg.loss.log_var_min,
            log_var_max=cfg.loss.log_var_max,
            antisymmetry_weight=cfg.loss.antisymmetry_weight,
        )
        pred_raw = standardizer.denormalize(pred_norm)
        n = target.shape[0]
        pairs = target.shape[0] * target.shape[1]
        totals["loss"] += float(loss_out.total.cpu()) * n
        totals["mae_raw"] += float(loss_out.mae_raw.cpu()) * n
        totals["mae_norm"] += float(loss_out.mae_norm.cpu()) * n
        totals["axis_abs_sum"] += (pred_raw - target).abs().sum(dim=(0, 1))
        totals["n_pairs"] += pairs
    elapsed = time.time() - t0
    n_windows = len(ds)
    return {
        "split": split,
        "windows": n_windows,
        "params": count_parameters(model),
        "loss": totals["loss"] / n_windows,
        "mae_raw": totals["mae_raw"] / n_windows,
        "mae_norm": totals["mae_norm"] / n_windows,
        "mae_axis_raw": (totals["axis_abs_sum"] / max(totals["n_pairs"], 1))
        .detach()
        .cpu()
        .tolist(),
        "windows_per_second": n_windows / max(elapsed, 1e-9),
        "seconds": elapsed,
    }


@torch.no_grad()
def benchmark_forward(
    cfg,
    ckpt_path: str,
    device: torch.device,
    iters: int = 200,
    warmup: int = 30,
    deploy: bool = True,
) -> dict:
    stats = load_stats(cfg.data.stats_path)
    preprocess = (
        FixedDerivativePreprocess(stats["preprocess_mean"], stats["preprocess_std"])
        .to(device)
        .eval()
    )
    model = build_model_from_config(cfg).to(device).eval()
    model.load_state_dict(
        torch.load(ckpt_path, map_location=device)["model"], strict=True
    )
    if deploy:
        switch_to_deploy(model)
    if cfg.train.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    x_rgb = torch.rand(1, cfg.data.K, 3, cfg.data.height, cfg.data.width, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    for _ in range(warmup):
        x = preprocess(x_rgb)
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        x = preprocess(x_rgb)
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return {
        "batch_size": 1,
        "iters": iters,
        "ms_per_window": 1000.0 * dt / iters,
        "deploy": deploy,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TinyDelta-3D.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Fuse reparameterizable depthwise branches before eval.",
    )
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.train.seed)
    configure_torch_for_speed()
    device = device_auto() if args.device == "auto" else torch.device(args.device)
    split = args.split or cfg.data.val_split
    metrics = evaluate(cfg, args.ckpt, device, split, deploy=args.deploy)
    print(metrics)
    if args.benchmark:
        print(benchmark_forward(cfg, args.ckpt, device, deploy=args.deploy))


if __name__ == "__main__":
    main()
