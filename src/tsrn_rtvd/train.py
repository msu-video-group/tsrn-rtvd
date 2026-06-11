"""
train_tsrnn_bt1.py – trainer for TSRNN_BT1.

TSRNN_BT1 implements forward(blur_seq, dp_seq) → (B, L, 3, H, W).
Loss is averaged over all time steps.

Validation uses the same 1-frame-latency streaming protocol as test_tsrnn_bt1.py.

Trajectory convention used here:
    - poses[:, :3] are camera centers / global camera positions by default
    - quaternion convention is camera-to-world by default
    - displacement convention is current -> neighbor
    - dp_seq stores (dx, dy) in full-resolution pixels

Usage:
    tsrn-rtvd-train
    tsrn-rtvd-train training.resume=outputs/tsrnn_bt1/last.pth

Recommended trajectory overrides for DA3/MONST3R camera-center paths:
    data.traj_pose_kind=camera_center \
    data.traj_rotation_convention=c2w \
    data.traj_z_ref=10 \
    data.traj_motion_scale=1 \
    data.traj_sign_x=1 \
    data.traj_sign_y=1
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from torch.amp import GradScaler, autocast

    _AMP_DEVICE = "cuda"
except ImportError:
    from torch.cuda.amp import GradScaler, autocast  # type: ignore

    _AMP_DEVICE = None

from tsrn_rtvd.data.dataset import SceneTraj, load_image
from tsrn_rtvd.data.gopro_sequence_dataset import build_sequence_dataset
from tsrn_rtvd.hub import package_config_path
from tsrn_rtvd.losses.losses import build_loss
from tsrn_rtvd.models.tsrnn_bt1 import TSRNN_BT1
from tsrn_rtvd.utils import (
    AverageMeter,
    PSNRMeter,
    load_checkpoint,
    load_config,
    make_output_dir,
    save_checkpoint,
    set_seed,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=package_config_path("tsrnn_bt1.yaml"))
    p.add_argument("overrides", nargs="*")
    return p.parse_args()


def build_model(cfg):
    return TSRNN_BT1()


def _cfg_get(obj: Any, name: str, default: Any) -> Any:
    """Read config attributes safely for SimpleNamespace/OmegaConf-like objects."""
    try:
        return getattr(obj, name)
    except Exception:
        return default


def trajectory_kwargs(cfg) -> dict[str, Any]:
    """Arguments passed to SceneTraj for both validation and dataset construction.

    These defaults match DA3/MONST3R-style camera-center trajectories and the
    user's current->neighbor convention. The dataset builder must also pass the
    same fields into SceneTraj during training.
    """
    d = cfg.data
    return {
        "z_ref": float(_cfg_get(d, "traj_z_ref", 10.0)),
        "motion_scale": float(_cfg_get(d, "traj_motion_scale", 1.0)),
        "sign_x": float(_cfg_get(d, "traj_sign_x", 1.0)),
        "sign_y": float(_cfg_get(d, "traj_sign_y", 1.0)),
        "pose_kind": str(_cfg_get(d, "traj_pose_kind", "camera_center")),
        "rotation_convention": str(_cfg_get(d, "traj_rotation_convention", "c2w")),
    }


def print_trajectory_config(cfg) -> None:
    kw = trajectory_kwargs(cfg)
    print("[INFO] Trajectory projection:")
    print(
        f"       pose_kind={kw['pose_kind']}  rotation_convention={kw['rotation_convention']}"
    )
    print(
        f"       z_ref={kw['z_ref']}  motion_scale={kw['motion_scale']}  "
        f"sign_x={kw['sign_x']}  sign_y={kw['sign_y']}"
    )
    print(
        "       convention=current->neighbor, dp_seq=(dx,dy), "
        "dp_pos=-dp_seq[t+1] during streaming validation"
    )


def print_dp_stats(loader, max_batches: int = 20) -> None:
    """Print dp_seq distribution before training to catch wrong scale/sign early."""
    vals = []
    try:
        for i, batch in enumerate(loader):
            if "dp_seq" not in batch:
                print("[WARN] Cannot print dp stats: batch has no 'dp_seq'.")
                return
            vals.append(batch["dp_seq"].detach().float().cpu().reshape(-1, 2))
            if i + 1 >= max_batches:
                break
    except Exception as exc:
        print(f"[WARN] Could not compute dp stats: {exc}")
        return

    if not vals:
        print("[WARN] Could not compute dp stats: no batches returned.")
        return

    dp = torch.cat(vals, dim=0)
    dp = torch.nan_to_num(dp, nan=0.0, posinf=0.0, neginf=0.0)
    mag = torch.linalg.norm(dp, dim=1)
    qs = torch.tensor([0.50, 0.90, 0.95, 0.99], dtype=torch.float32)

    def qfmt(x: torch.Tensor) -> str:
        return "[" + ", ".join(f"{float(v):.4g}" for v in x) + "]"

    print("[DP] Stats from first", min(max_batches, len(vals)), "training batches")
    print(f"[DP] zero_frac={float((mag == 0).float().mean()):.4f}")
    print(
        f"[DP] abs dx p50/p90/p95/p99={qfmt(torch.quantile(dp[:, 0].abs(), qs))} max={float(dp[:, 0].abs().max()):.4g}"
    )
    print(
        f"[DP] abs dy p50/p90/p95/p99={qfmt(torch.quantile(dp[:, 1].abs(), qs))} max={float(dp[:, 1].abs().max()):.4g}"
    )
    print(
        f"[DP] mag    p50/p90/p95/p99={qfmt(torch.quantile(mag, qs))} max={float(mag.max()):.4g}"
    )


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, cfg, epoch):
    model.train()
    loss_meter = AverageMeter()
    skip_count = 0

    for step, batch in enumerate(loader):
        blur_seq = batch["blur_seq"].to(device, non_blocking=True)
        sharp_seq = batch["sharp_seq"].to(device, non_blocking=True)
        dp_seq = batch["dp_seq"].to(device, non_blocking=True)

        B, L, _, H, W = blur_seq.shape
        optimizer.zero_grad(set_to_none=True)

        with autocast(_AMP_DEVICE or "cuda", enabled=cfg.training.amp):
            preds = model(blur_seq, dp_seq)
            loss = criterion(
                preds.reshape(B * L, 3, H, W),
                sharp_seq.reshape(B * L, 3, H, W),
            )

        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            skip_count += 1
        else:
            scaler.scale(loss).backward()
            if cfg.training.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            loss_meter.update(loss.item(), B)

        if (step + 1) % cfg.experiment.log_interval == 0:
            skip_str = f"  skipped={skip_count}" if skip_count > 0 else ""
            print(
                f"  [epoch {epoch:>3d} | step {step + 1:>5d}/{len(loader)}] "
                f"loss={loss_meter.avg:.5f}{skip_str}"
            )
            skip_count = 0

    return loss_meter.avg


@torch.no_grad()
def validate(model, val_root, val_traj_root, device, cfg):
    model.eval()
    psnr_meter = PSNRMeter()
    traj_kw = trajectory_kwargs(cfg)

    for scene in sorted(d for d in val_root.iterdir() if d.is_dir()):
        frames = sorted((scene / "blur").glob("*.png"))
        sharp_dir = scene / "sharp"
        if not frames:
            continue

        traj = SceneTraj(val_traj_root / scene.name / "trajectories.npz", **traj_kw)
        N = len(frames)

        B0 = load_image(frames[0]).unsqueeze(0).to(device)
        _, _, H, W = B0.shape
        state = model.init_state(1, H, W, device)

        with autocast(_AMP_DEVICE or "cuda", enabled=cfg.training.amp):
            feat0 = tuple(f.detach() for f in model.encoder(B0))

        state["feat_prev"] = feat0
        state["feat_cur"] = feat0
        state["B_cur"] = B0

        for ci in range(N):
            ci_next = min(N - 1, ci + 1)

            B_next = load_image(frames[ci_next]).unsqueeze(0).to(device)
            sharp = load_image(sharp_dir / frames[ci].name).unsqueeze(0).to(device)

            # Current->previous, returned by SceneTraj as (dy, dx).
            dy_dx_neg = traj.get_disp(
                frames[ci].stem,
                frames[max(0, ci - 1)].stem,
            )
            dp_neg = (
                torch.from_numpy(
                    np.nan_to_num(np.array([dy_dx_neg[1], dy_dx_neg[0]], np.float32))
                )
                .unsqueeze(0)
                .to(device)
            )

            # Training convention for the future side is -dp_seq[t+1].
            # dp_seq[t+1] is next->current, so the negative approximates current->next.
            dy_dx_pos = traj.get_disp(
                frames[ci_next].stem,
                frames[ci].stem,
            )
            dp_pos = (
                torch.from_numpy(
                    np.nan_to_num(np.array([dy_dx_pos[1], dy_dx_pos[0]], np.float32))
                    * -1.0
                )
                .unsqueeze(0)
                .to(device)
            )

            with autocast(_AMP_DEVICE or "cuda", enabled=cfg.training.amp):
                pred, state = model.step(B_next, state, dp_neg, dp_pos)
                pred = pred.float().clamp(0, 1)

            psnr_meter.update(pred, sharp.float())

    return psnr_meter.avg


def main():
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Device: {device}")
    print_trajectory_config(cfg)

    set_seed(cfg.experiment.seed)
    out_dir = make_output_dir(cfg)

    train_ds = build_sequence_dataset(cfg, split="train")
    print(f"[INFO] Training clips: {len(train_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=True,
    )

    if bool(_cfg_get(cfg.data, "print_traj_stats", True)):
        print_dp_stats(
            train_loader, max_batches=int(_cfg_get(cfg.data, "traj_stats_batches", 20))
        )

    model = build_model(cfg).to(device)
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Model: TSRNN_BT1  Parameters: {n_p / 1e6:.2f} M")

    oc = cfg.optimizer
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=oc.lr,
        weight_decay=oc.weight_decay,
        betas=tuple(oc.betas),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=cfg.scheduler.T_max,
        eta_min=cfg.scheduler.eta_min,
    )

    criterion = build_loss(cfg).to(device)
    scaler = (
        GradScaler("cuda", enabled=cfg.training.amp)
        if _AMP_DEVICE
        else GradScaler(enabled=cfg.training.amp)
    )

    start_epoch, best_psnr = 0, 0.0
    if cfg.training.resume:
        start_epoch, best_psnr = load_checkpoint(
            cfg.training.resume,
            model,
            opt,
            scheduler,
        )

    val_root = Path(cfg.data.root) / "test"
    val_traj_root = Path(cfg.data.traj_root) / "test"

    for epoch in range(start_epoch + 1, cfg.training.epochs + 1):
        t0 = time.time()

        loss = train_one_epoch(
            model,
            train_loader,
            opt,
            criterion,
            scaler,
            device,
            cfg,
            epoch,
        )

        scheduler.step()

        log = (
            f"[epoch {epoch:>3d}/{cfg.training.epochs}] "
            f"loss={loss:.5f}  lr={opt.param_groups[0]['lr']:.2e}  "
            f"time={time.time() - t0:.0f}s"
        )

        is_best = False
        if epoch % cfg.experiment.val_interval == 0:
            val_psnr = validate(model, val_root, val_traj_root, device, cfg)
            is_best = val_psnr > best_psnr
            if is_best:
                best_psnr = val_psnr

            log += f"  val_psnr={val_psnr:.2f}dB"
            if is_best:
                log += "  ✓ best"

        print(log)

        save_checkpoint(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_psnr": best_psnr,
            },
            str(out_dir),
            filename="last.pth",
            is_best=is_best,
        )

    print(f"\n[DONE] Best val PSNR: {best_psnr:.2f} dB → {out_dir}")


if __name__ == "__main__":
    main()
