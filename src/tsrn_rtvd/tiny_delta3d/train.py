from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .config import load_config, save_config
from .dataset import build_dataset, collate_samples
from .losses import DeltaStandardizer, tinydelta_loss
from .model import build_model_from_config, count_parameters, make_deploy_copy
from .preprocess import FixedDerivativePreprocess
from .utils import (
    configure_torch_for_speed,
    device_auto,
    format_seconds,
    load_stats,
    mkdir,
    move_batch_to_device,
    save_jsonl,
    seed_everything,
)


def make_loader(cfg, split: str, train: bool, device: torch.device) -> DataLoader:
    ds = build_dataset(cfg, split, train=train)
    return DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        shuffle=train,
        drop_last=train,
        num_workers=cfg.train.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(cfg.train.num_workers > 0 and cfg.train.persistent_workers),
        prefetch_factor=cfg.train.prefetch_factor
        if cfg.train.num_workers > 0
        else None,
        collate_fn=collate_samples,
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    cfg,
    epoch: int,
    best_metric: float,
    stats: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
            "config_name": cfg.name,
            "stats": stats,
        },
        path,
    )


def load_checkpoint_if_any(
    path: str | None, model: nn.Module, optimizer=None, scaler=None, map_location="cpu"
) -> int:
    if not path:
        return 0
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt.get("epoch", 0)) + 1


def step_batch(
    batch: dict,
    model: nn.Module,
    preprocess: FixedDerivativePreprocess,
    standardizer: DeltaStandardizer,
    cfg,
    device: torch.device,
    optimizer=None,
    scaler: torch.amp.GradScaler | None = None,
) -> dict:
    is_train = optimizer is not None
    batch = move_batch_to_device(batch, device)
    with torch.no_grad():
        x = preprocess(batch["x_rgb"])
    target = batch["target"]

    autocast_enabled = bool(cfg.train.amp and device.type == "cuda")
    with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
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
    if is_train:
        optimizer.zero_grad(set_to_none=True)
        assert scaler is not None
        scaler.scale(loss_out.total).backward()
        if cfg.train.grad_clip_norm and cfg.train.grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
    return {
        "loss": float(loss_out.total.detach().cpu()),
        "regression": float(loss_out.regression.detach().cpu()),
        "uncertainty": float(loss_out.uncertainty.detach().cpu()),
        "antisymmetry": float(loss_out.antisymmetry.detach().cpu()),
        "mae_norm": float(loss_out.mae_norm.detach().cpu()),
        "mae_raw": float(loss_out.mae_raw.detach().cpu()),
    }


def average_rows(rows: list[dict]) -> dict:
    keys = rows[0].keys()
    return {k: sum(r[k] for r in rows) / len(rows) for k in keys}


@torch.no_grad()
def evaluate(loader, model, preprocess, standardizer, cfg, device) -> dict:
    model.eval()
    rows = []
    for batch in loader:
        rows.append(step_batch(batch, model, preprocess, standardizer, cfg, device))
    return average_rows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TinyDelta-3D-Pooled.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.train.seed)
    configure_torch_for_speed()
    out_dir = mkdir(cfg.train.output_dir)
    save_config(cfg, out_dir / "config.yaml")

    device = device_auto() if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    stats = load_stats(cfg.data.stats_path)
    preprocess = (
        FixedDerivativePreprocess(stats["preprocess_mean"], stats["preprocess_std"])
        .to(device)
        .eval()
    )
    standardizer = DeltaStandardizer(stats["delta_mean"], stats["delta_std"]).to(device)

    train_loader = make_loader(cfg, cfg.data.train_split, train=True, device=device)
    val_loader = make_loader(cfg, cfg.data.val_split, train=False, device=device)

    model = build_model_from_config(cfg).to(device)
    if cfg.train.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    raw_model = model
    run_model = (
        torch.compile(model)
        if cfg.train.compile and hasattr(torch, "compile")
        else model
    )

    optimizer = AdamW(
        raw_model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    scaler = torch.amp.GradScaler(
        device=device.type, enabled=(cfg.train.amp and device.type == "cuda")
    )
    start_epoch = (
        load_checkpoint_if_any(
            args.resume, raw_model, optimizer, scaler, map_location=device
        )
        if args.resume
        else 0
    )

    print(
        f"device={device} amp={cfg.train.amp and device.type == 'cuda'} compile={run_model is not raw_model}"
    )
    print(
        f"train_windows={len(train_loader.dataset)} val_windows={len(val_loader.dataset)}"
    )
    print(f"params_train_graph={count_parameters(raw_model):,}")

    best_metric = float("inf")
    history_path = out_dir / "history.jsonl"
    t0 = time.time()
    for epoch in range(start_epoch, cfg.train.epochs):
        raw_model.train()
        rows = []
        epoch_t0 = time.time()
        for step, batch in enumerate(train_loader):
            row = step_batch(
                batch,
                run_model,
                preprocess,
                standardizer,
                cfg,
                device,
                optimizer=optimizer,
                scaler=scaler,
            )
            rows.append(row)
            if cfg.train.log_every > 0 and (step + 1) % cfg.train.log_every == 0:
                avg = average_rows(rows[-cfg.train.log_every :])
                print(
                    f"epoch={epoch:03d} step={step + 1:05d}/{len(train_loader):05d} "
                    f"loss={avg['loss']:.5f} mae_raw={avg['mae_raw']:.6f}"
                )

        train_metrics = average_rows(rows)
        val_metrics = None
        if (epoch + 1) % cfg.train.val_every == 0:
            val_metrics = evaluate(
                val_loader, run_model, preprocess, standardizer, cfg, device
            )
            best = val_metrics["mae_raw"] < best_metric
            if best:
                best_metric = val_metrics["mae_raw"]
                save_checkpoint(
                    out_dir / "best.pt",
                    raw_model,
                    optimizer,
                    scaler,
                    cfg,
                    epoch,
                    best_metric,
                    stats,
                )
        else:
            best = False

        row = {
            "epoch": epoch,
            "seconds": time.time() - epoch_t0,
            "train": train_metrics,
            "val": val_metrics,
            "best_metric": best_metric,
        }
        save_jsonl(history_path, row)
        print(
            f"epoch={epoch:03d} done in {format_seconds(row['seconds'])} "
            f"train_loss={train_metrics['loss']:.5f} train_mae_raw={train_metrics['mae_raw']:.6f} "
            + (f"val_mae_raw={val_metrics['mae_raw']:.6f} " if val_metrics else "")
            + ("[best]" if best else "")
        )

        if (epoch + 1) % cfg.train.save_every == 0:
            save_checkpoint(
                out_dir / "last.pt",
                raw_model,
                optimizer,
                scaler,
                cfg,
                epoch,
                best_metric,
                stats,
            )

    # Also export a fused/deploy checkpoint for latency tests.
    deploy_model = make_deploy_copy(raw_model).cpu()
    torch.save(
        {"model": deploy_model.state_dict(), "stats": stats, "config_name": cfg.name},
        out_dir / "deploy_fused.pt",
    )
    print(f"finished in {format_seconds(time.time() - t0)}; output_dir={out_dir}")


if __name__ == "__main__":
    main()
