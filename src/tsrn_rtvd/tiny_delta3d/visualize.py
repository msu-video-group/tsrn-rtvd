from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from .config import load_config
from .dataset import build_dataset, collate_samples
from .losses import DeltaStandardizer
from .model import build_model_from_config, switch_to_deploy
from .preprocess import FixedDerivativePreprocess
from .utils import (
    configure_torch_for_speed,
    device_auto,
    load_stats,
    mkdir,
    move_batch_to_device,
)


@torch.no_grad()
def visualize_sample(
    cfg,
    ckpt_path: str,
    out_path: str | Path,
    split: str,
    index: int,
    device: torch.device,
    deploy: bool = False,
) -> None:
    stats = load_stats(cfg.data.stats_path)
    preprocess = (
        FixedDerivativePreprocess(stats["preprocess_mean"], stats["preprocess_std"])
        .to(device)
        .eval()
    )
    standardizer = DeltaStandardizer(stats["delta_mean"], stats["delta_std"]).to(device)
    ds = build_dataset(cfg, split, train=False)
    sample = ds[index]
    batch = collate_samples([sample])
    batch = move_batch_to_device(batch, device)

    model = build_model_from_config(cfg).to(device).eval()
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    if deploy:
        switch_to_deploy(model)

    x = preprocess(batch["x_rgb"])
    pred_norm, log_var = model(x)
    pred = standardizer.denormalize(pred_norm)[0].detach().cpu()
    target = batch["target"][0].detach().cpu()
    frames = batch["x_rgb"][0].detach().cpu()

    K = cfg.data.K
    c = K // 2
    neighbor_indices = [i for i in range(K) if i != c]

    fig = plt.figure(figsize=(14, 8))
    grid = fig.add_gridspec(2, max(K, 3))
    for i in range(K):
        ax = fig.add_subplot(grid[0, i])
        img = frames[i].permute(1, 2, 0).numpy()
        ax.imshow(img)
        title = f"t{i - c:+d}"
        if i == c:
            title += " center"
        ax.set_title(title)
        ax.axis("off")

    ax = fig.add_subplot(grid[1, :2])
    xs = list(range(len(neighbor_indices)))
    for axis, name in enumerate(["tx", "ty", "tz"]):
        ax.plot(xs, target[:, axis], marker="o", label=f"target {name}")
        ax.plot(xs, pred[:, axis], marker="x", linestyle="--", label=f"pred {name}")
    ax.set_xticks(xs, [f"center→{i - c:+d}" for i in neighbor_indices])
    ax.set_title("Center-relative translation")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax2 = fig.add_subplot(grid[1, 2:])
    err = (pred - target).abs()
    im = ax2.imshow(err.numpy(), aspect="auto")
    ax2.set_xticks([0, 1, 2], ["x", "y", "z"])
    ax2.set_yticks(xs, [f"center→{i - c:+d}" for i in neighbor_indices])
    ax2.set_title(f"Absolute error; mean={err.mean().item():.6f}")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"{sample['scene']} / {sample['center_image']} / log_var={log_var.mean().item():.3f}"
    )
    fig.tight_layout()
    out_path = Path(out_path)
    mkdir(out_path.parent)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_history(history_path: str | Path, out_path: str | Path) -> None:
    rows = []
    with open(history_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    epochs = [r["epoch"] for r in rows]
    train_loss = [r["train"]["loss"] for r in rows]
    train_mae = [r["train"]["mae_raw"] for r in rows]
    val_mae = [None if r["val"] is None else r["val"]["mae_raw"] for r in rows]

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.plot(epochs, train_loss, label="train loss")
    ax.plot(epochs, train_mae, label="train raw MAE")
    if any(v is not None for v in val_mae):
        ax.plot(
            [e for e, v in zip(epochs, val_mae) if v is not None],
            [v for v in val_mae if v is not None],
            label="val raw MAE",
        )
    ax.set_xlabel("epoch")
    ax.set_title("TinyDelta-3D training history")
    ax.grid(True, alpha=0.3)
    ax.legend()
    out_path = Path(out_path)
    mkdir(out_path.parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_architecture(cfg, out_path: str | Path) -> None:
    stages = [
        ("Input", f"B×{cfg.data.K}×3×{cfg.data.height}×{cfg.data.width}"),
        ("Fixed preprocess", "gray + Sobel mag + Laplacian"),
        ("Shared encoder", f"rep-DW conv, D={cfg.model.D}"),
        ("Temporal input", "[F, F-center, r]"),
        (
            "2× Temporal DS Conv1d",
            f"width={cfg.model.D}, k={cfg.model.temporal_kernel}",
        ),
        ("Shared pair head", "[center, neighbor, diff, r]"),
        ("Output", f"B×{cfg.data.K - 1}×(3+log_var)"),
    ]
    fig = plt.figure(figsize=(13, 3.2))
    ax = fig.add_subplot(111)
    ax.axis("off")
    y = 0.5
    box_w = 0.12
    gap = 0.025
    for i, (title, subtitle) in enumerate(stages):
        x = 0.02 + i * (box_w + gap)
        rect = plt.Rectangle((x, y - 0.18), box_w, 0.36, fill=False, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(
            x + box_w / 2,
            y + 0.05,
            title,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
        )
        ax.text(x + box_w / 2, y - 0.07, subtitle, ha="center", va="center", fontsize=7)
        if i < len(stages) - 1:
            ax.annotate(
                "",
                xy=(x + box_w + gap * 0.75, y),
                xytext=(x + box_w + gap * 0.2, y),
                arrowprops=dict(arrowstyle="->"),
            )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    out_path = Path(out_path)
    mkdir(out_path.parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize TinyDelta-3D samples, history, or architecture."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--mode", choices=["sample", "history", "architecture"], default="sample"
    )
    parser.add_argument("--history", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--deploy", action="store_true")
    args = parser.parse_args()

    configure_torch_for_speed()
    cfg = load_config(args.config)
    device = device_auto() if args.device == "auto" else torch.device(args.device)
    if args.mode == "architecture":
        plot_architecture(cfg, args.out)
    elif args.mode == "history":
        history_path = args.history or str(Path(cfg.train.output_dir) / "history.jsonl")
        plot_history(history_path, args.out)
    else:
        if args.ckpt is None:
            raise ValueError("--ckpt is required for --mode sample")
        split = args.split or cfg.data.val_split
        visualize_sample(
            cfg, args.ckpt, args.out, split, args.index, device, deploy=args.deploy
        )
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
