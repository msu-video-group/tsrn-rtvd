"""
Streaming evaluation for TSRNN_BT1.

BT1: 1-frame latency. Encode only B_next per step; feat_prev/cur cached in state.

GPU timing wraps the full model.step() call, including any new encodes.
Pipeline timing adds data loading and .to(device).

Usage:
    tsrn-rtvd-eval --checkpoint path/to/best.pth
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import autocast

from tsrn_rtvd.data.dataset import SceneTraj, load_image
from tsrn_rtvd.hub import TSRNN_CONFIG_NAME, package_config_path
from tsrn_rtvd.models.new_blocks import pad_dp
from tsrn_rtvd.models.tsrnn_bt1 import TSRNN_BT1
from tsrn_rtvd.tiny_delta3d.model import (
    build_model_from_config as build_traj_model_from_config,
)
from tsrn_rtvd.tiny_delta3d.model import (
    make_deploy_copy as make_traj_deploy_copy,
)
from tsrn_rtvd.tiny_delta3d.preprocess import FixedDerivativePreprocess
from tsrn_rtvd.utils import AverageMeter, PSNRMeter, load_checkpoint, load_config
from tsrn_rtvd.utils.traj3d_projection import (
    LearnedTrajectory3DAdapter,
    TrajProjectionConfig,
    Translation3DToPixelDP,
    extract_state_dict,
    find_mean_std,
    find_output_mean_std,
    load_stats_file,
)

P = 4


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=package_config_path(TSRNN_CONFIG_NAME))
    p.add_argument("--checkpoint", required=True)

    p.add_argument(
        "--fast-shift",
        choices=["off", "roll_match", "exact"],
        default="exact",
        help="BT1 shift optimization. Default: exact, i.e. one full-displacement grid_sample.",
    )
    p.add_argument(
        "--no-channels-last",
        dest="channels_last",
        action="store_false",
        help="Disable channels_last inference layout. Enabled by default on CUDA.",
    )
    p.set_defaults(channels_last=True)
    p.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="Override cfg.inference.warmup_frames / default 50.",
    )

    p.add_argument(
        "--traj-config",
        default=None,
        help="Config for TinyDelta3D trajectory predictor.",
    )
    p.add_argument(
        "--traj-checkpoint", default=None, help="Pretrained TinyDelta3D checkpoint."
    )
    p.add_argument(
        "--traj-stats", default=None, help="Stats file for FixedDerivativePreprocess."
    )
    p.add_argument(
        "--traj-deploy",
        action="store_true",
        help="Convert RepDWConv3x3 blocks to deploy graph after loading.",
    )
    p.add_argument(
        "--traj-nonstrict",
        action="store_true",
        help="Load trajectory checkpoint with strict=False.",
    )

    p.add_argument(
        "--traj-fx",
        type=float,
        default=1.0,
        help="Camera fx for 3D-to-pixel projection.",
    )
    p.add_argument(
        "--traj-fy",
        type=float,
        default=1.0,
        help="Camera fy for 3D-to-pixel projection.",
    )
    p.add_argument(
        "--traj-z-ref",
        type=float,
        default=1.0,
        help="Reference scene depth for weak-perspective projection.",
    )
    p.add_argument(
        "--traj-sign-x",
        type=float,
        default=1.0,
        help="Sign correction for projected dx.",
    )
    p.add_argument(
        "--traj-sign-y",
        type=float,
        default=1.0,
        help="Sign correction for projected dy.",
    )
    p.add_argument(
        "--traj-output-scale",
        type=float,
        default=1.0,
        help="Extra multiplier applied to TinyDelta3D 3D output before projection.",
    )
    p.add_argument(
        "--traj-output-order",
        choices=["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"],
        default="xyz",
        help="Axis order of TinyDelta3D output channels.",
    )
    p.add_argument("--oracle-z-ref", type=float, default=1.0)
    p.add_argument("--oracle-motion-scale", type=float, default=1.0)
    p.add_argument("--oracle-sign-x", type=float, default=1.0)
    p.add_argument("--oracle-sign-y", type=float, default=1.0)
    p.add_argument(
        "--oracle-pose-kind",
        choices=["camera_center", "colmap_extrinsic"],
        default="camera_center",
    )
    p.add_argument(
        "--oracle-rotation-convention", choices=["c2w", "w2c"], default="c2w"
    )

    p.add_argument("overrides", nargs="*")
    return p.parse_args()


def build_model(cfg):
    return TSRNN_BT1()


def maybe_resize(img, tr):
    if tr is None:
        return img
    import torch.nn.functional as F

    # img is (3,H,W); resize via torch to avoid a hard torchvision runtime dependency.
    x = img.unsqueeze(0)
    try:
        x = F.interpolate(
            x, size=list(tr), mode="bilinear", align_corners=False, antialias=True
        )
    except TypeError:  # older PyTorch without antialias argument
        x = F.interpolate(x, size=list(tr), mode="bilinear", align_corners=False)
    return x.squeeze(0)


def load_gpu(path, tr, device, channels_last: bool = False):
    x = maybe_resize(load_image(path), tr).unsqueeze(0).to(device, non_blocking=True)
    if channels_last and x.dim() == 4:
        x = x.contiguous(memory_format=torch.channels_last)
    return x


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def get_dp4(traj, stem_t, stem_prev, device, sign=1):
    dy_dx = traj.get_disp(stem_t, stem_prev)
    raw = np.nan_to_num(np.array([dy_dx[1], dy_dx[0]], np.float32))
    dp2 = torch.from_numpy(raw * sign).unsqueeze(0).to(device)
    return pad_dp(dp2, P)


def get_oracle_dp2(traj, stem_t, stem_prev, device, sign=1):
    return get_dp4(traj, stem_t, stem_prev, device, sign=sign)[:, :2]


def build_learned_trajectory_adapter(args, device):
    if args.traj_checkpoint is None:
        return None

    if args.traj_config is None:
        raise ValueError("--traj-config is required when --traj-checkpoint is used.")
    if args.traj_stats is None:
        raise ValueError("--traj-stats is required when --traj-checkpoint is used.")

    traj_cfg = load_config(args.traj_config, [])
    traj_model = build_traj_model_from_config(traj_cfg).to(device)

    if getattr(traj_model, "K", 3) != 3:
        raise ValueError(
            f"This adapter expects TinyDelta3D K=3, got K={traj_model.K}. "
            "Expected triplets [prev, cur, next]."
        )

    ckpt = torch.load(str(args.traj_checkpoint), map_location=device)
    sd = extract_state_dict(ckpt)

    incompatible = traj_model.load_state_dict(sd, strict=not args.traj_nonstrict)
    if args.traj_nonstrict:
        print(f"[WARN] Traj missing keys: {incompatible.missing_keys}")
        print(f"[WARN] Traj unexpected keys: {incompatible.unexpected_keys}")

    if args.traj_deploy:
        traj_model = make_traj_deploy_copy(traj_model).to(device)

    stats = load_stats_file(args.traj_stats)
    prep_mean, prep_std = find_mean_std(stats)

    preprocess = FixedDerivativePreprocess().to(device)
    preprocess.set_stats(prep_mean, prep_std)

    out_mean, out_std = find_output_mean_std(stats)

    proj_cfg = TrajProjectionConfig(
        fx=args.traj_fx,
        fy=args.traj_fy,
        z_ref=args.traj_z_ref,
        sign_x=args.traj_sign_x,
        sign_y=args.traj_sign_y,
        output_scale=args.traj_output_scale,
        output_order=args.traj_output_order,
    )

    projector = Translation3DToPixelDP(
        proj_cfg,
        output_mean=out_mean,
        output_std=out_std,
    ).to(device)

    print(f"[INFO] Loaded learned 3D trajectory model: {args.traj_checkpoint}")
    print(f"[INFO] Loaded trajectory preprocess stats: {args.traj_stats}")
    print(
        "[INFO] 3D→2D projection: "
        f"fx={args.traj_fx}, fy={args.traj_fy}, z_ref={args.traj_z_ref}, "
        f"sign_x={args.traj_sign_x}, sign_y={args.traj_sign_y}, "
        f"scale={args.traj_output_scale}, order={args.traj_output_order}"
    )

    if out_mean is not None and out_std is not None:
        print("[INFO] Using output mean/std denormalization for trajectory output.")
    else:
        print("[INFO] No output mean/std found; using raw trajectory model output.")

    return LearnedTrajectory3DAdapter(
        model=traj_model,
        preprocess=preprocess,
        projector=projector,
    )


def warmup(model, device, cfg, traj_adapter=None, n=50, channels_last: bool = False):
    print(f"[INFO] Warm-up ({n} steps) …")
    tr = cfg.data.test_resize
    H, W = (tr[0], tr[1]) if tr else (720, 1280)
    dummy = torch.zeros(1, 3, H, W, device=device)
    if channels_last and dummy.dim() == 4:
        dummy = dummy.contiguous(memory_format=torch.channels_last)
    zP = torch.zeros(1, 2, device=device)
    state = model.init_state(1, H, W, device)

    with torch.no_grad():
        for _ in range(n):
            with autocast(enabled=cfg.inference.amp):
                if traj_adapter is not None:
                    traj_adapter.predict_left_right(dummy, dummy, dummy)

                _, state = model.step(dummy, state, zP, zP)

    sync(device)
    print("[INFO] Warm-up done.\n")


@torch.no_grad()
def eval_bt1(model, device, cfg, args, traj_adapter=None, channels_last: bool = False):
    use_amp = cfg.inference.amp
    tr = cfg.data.test_resize
    psnr_m = PSNRMeter()
    gpu_m = AverageMeter()
    wall_m = AverageMeter()

    for scene in sorted(
        d for d in (Path(cfg.data.root) / "test").iterdir() if d.is_dir()
    ):
        frames = sorted((scene / "blur").glob("*.png"))
        traj = (
            None
            if traj_adapter is not None
            else SceneTraj(
                Path(cfg.data.traj_root) / "test" / scene.name / "trajectories.npz",
                z_ref=args.oracle_z_ref,
                motion_scale=args.oracle_motion_scale,
                sign_x=args.oracle_sign_x,
                sign_y=args.oracle_sign_y,
                pose_kind=args.oracle_pose_kind,
                rotation_convention=args.oracle_rotation_convention,
            )
        )
        N = len(frames)

        B0 = load_gpu(frames[0], tr, device, channels_last=channels_last)
        _, _, H, W = B0.shape
        state = model.init_state(1, H, W, device)

        B_prev_raw = B0
        B_cur_raw = B0

        with torch.no_grad(), autocast(enabled=use_amp):
            feat0 = tuple(f.detach() for f in model.encoder(B0))

        state["feat_prev"] = feat0
        state["feat_cur"] = feat0
        state["B_cur"] = B0

        for ci in range(N):
            ci_next = min(N - 1, ci + 1)

            sync(device)
            t_wall = time.perf_counter()

            B_next = load_gpu(frames[ci_next], tr, device, channels_last=channels_last)
            sharp = load_gpu(
                scene / "sharp" / frames[ci].name,
                tr,
                device,
                channels_last=channels_last,
            )

            sync(device)
            t0 = time.perf_counter()
            with autocast(enabled=use_amp):
                if traj_adapter is not None:
                    dp_neg, dp_pos = traj_adapter.predict_left_right(
                        B_prev_raw,
                        B_cur_raw,
                        B_next,
                    )

                    if ci == 0:
                        dp_neg = torch.zeros_like(dp_neg)
                    if ci_next == ci:
                        dp_pos = torch.zeros_like(dp_pos)
                else:
                    dp_neg = get_oracle_dp2(
                        traj,
                        frames[ci].stem,
                        frames[max(0, ci - 1)].stem,
                        device,
                    )
                    dp_pos = get_oracle_dp2(
                        traj,
                        frames[ci_next].stem,
                        frames[ci].stem,
                        device,
                        sign=-1,
                    )

                t0 = time.perf_counter()

                pred, state = model.step(B_next, state, dp_neg, dp_pos)

            sync(device)
            gpu_m.update((time.perf_counter() - t0) * 1000)

            sync(device)
            wall_m.update((time.perf_counter() - t_wall) * 1000)
            psnr_m.update(pred.float().clamp(0, 1), sharp.float())

            B_prev_raw = B_cur_raw.detach()
            B_cur_raw = B_next.detach()

        print(f"  {scene.name:30s}  {N:4d}fr  psnr={psnr_m.avg:.2f}dB")

    return psnr_m.avg, gpu_m.avg, wall_m.avg


def main():
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = "TSRNN_BT1"

    print(f"[INFO] Device: {device}  Model: {name}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    model = build_model(cfg).to(device)

    if args.fast_shift != "off":
        from tsrn_rtvd.models.shift_ops import apply_speed_patches

        apply_speed_patches(model, mode=args.fast_shift)

    n_p = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Parameters: {n_p / 1e6:.2f} M")
    load_checkpoint(args.checkpoint, model)
    print(f"[INFO] Loaded: {args.checkpoint}")

    use_channels_last = bool(args.channels_last and device.type == "cuda")
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)
        model._channels_last_inference = True
        print("[INFO] channels_last inference: enabled")
    else:
        print("[INFO] channels_last inference: disabled")

    model.eval()
    print(f"[INFO] fast_shift: {args.fast_shift}\n")

    traj_adapter = build_learned_trajectory_adapter(args, device)

    warmup_steps = args.warmup_steps
    if warmup_steps is None:
        warmup_steps = int(getattr(cfg.inference, "warmup_frames", 50))
    warmup(
        model,
        device,
        cfg,
        traj_adapter=traj_adapter,
        n=warmup_steps,
        channels_last=use_channels_last,
    )

    psnr, gpu_ms, wall_ms = eval_bt1(
        model,
        device,
        cfg,
        args,
        traj_adapter=traj_adapter,
        channels_last=use_channels_last,
    )

    sep = "─" * 58
    latency = 1

    print(f"\n{sep}")
    print(f"  {name}  (latency: {latency} frame(s))")
    print(sep)
    print(f"  PSNR             : {psnr:.2f} dB")
    print(f"  GPU-only latency : {gpu_ms:.2f} ms  ({1000 / gpu_ms:.1f} FPS)")
    print(f"  Pipeline latency : {wall_ms:.2f} ms  ({1000 / wall_ms:.1f} FPS)")
    print(sep)

    for label, fps in [("GPU-only", 1000 / gpu_ms), ("Pipeline", 1000 / wall_ms)]:
        print(f"  ≥30 FPS ({label:10s}): {'✓ YES' if fps >= 30 else '✗ NO'}")

    print("\n  Competitors:")
    for cname, ms in [
        ("MMP-RNN", 367),
        ("MemDeblur", 320),
        ("ESTRNN", 336),
        ("RealTime_VDBLR", 137),
        ("UHDVD", 181),
        ("Ours", wall_ms),
    ]:
        mark = " ◄" if cname == "Ours" else ""
        print(f"    {cname:<22s}: {ms:>7.1f} ms  ({1000 / ms:>5.1f} FPS){mark}")

    print()


if __name__ == "__main__":
    main()
