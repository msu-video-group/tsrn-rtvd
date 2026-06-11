"""
Folder inference for the BT1-only RTVD repository.

Processes every image in an input directory as a streaming sequence and writes
one deblurred output per input frame. Trajectories are predicted online with the
TinyDelta3D branch when --traj-checkpoint/--traj-config/--traj-stats are given;
otherwise the script falls back to zero displacements.

Example:
    tsrn-rtvd-infer-folder --checkpoint path/to/best.pth --input-dir /path/to/frames
"""

from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from PIL import Image

try:
    from torch.amp import autocast as _autocast

    def amp_context(device: torch.device, enabled: bool):
        if device.type == "cuda":
            return _autocast(device_type="cuda", enabled=enabled)
        return nullcontext()

except Exception:  # pragma: no cover - compatibility with older PyTorch
    from torch.cuda.amp import autocast as _cuda_autocast  # type: ignore

    def amp_context(device: torch.device, enabled: bool):
        if device.type == "cuda":
            return _cuda_autocast(enabled=enabled)
        return nullcontext()


from tsrn_rtvd.data.dataset import load_image
from tsrn_rtvd.hub import TSRNN_CONFIG_NAME, package_config_path
from tsrn_rtvd.models.tsrnn_bt1 import TSRNN_BT1
from tsrn_rtvd.tiny_delta3d.model import (
    build_model_from_config as build_traj_model_from_config,
)
from tsrn_rtvd.tiny_delta3d.model import (
    make_deploy_copy as make_traj_deploy_copy,
)
from tsrn_rtvd.tiny_delta3d.preprocess import FixedDerivativePreprocess
from tsrn_rtvd.utils import load_checkpoint, load_config
from tsrn_rtvd.utils.traj3d_projection import (
    LearnedTrajectory3DAdapter,
    TrajProjectionConfig,
    Translation3DToPixelDP,
    extract_state_dict,
    find_mean_std,
    find_output_mean_std,
    load_stats_file,
)

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args():
    p = argparse.ArgumentParser(
        description="TSRNN_BT1 streaming inference on an image folder"
    )

    p.add_argument(
        "--config", default=package_config_path(TSRNN_CONFIG_NAME), help="BT1 config"
    )
    p.add_argument("--checkpoint", required=True, help="BT1 checkpoint")
    p.add_argument(
        "--input-dir", required=True, help="Directory with input frames/images"
    )
    p.add_argument(
        "--output-dir", default=None, help="Default: <input-dir>_OUT next to input dir"
    )

    p.add_argument(
        "--fast-shift",
        choices=["off", "roll_match", "exact"],
        default="exact",
        help="BT1 shift mode. Use exact for exact-finetuned checkpoints.",
    )
    p.add_argument("--no-channels-last", dest="channels_last", action="store_false")
    p.add_argument("--channels-last", dest="channels_last", action="store_true")
    p.set_defaults(channels_last=True)

    p.add_argument(
        "--compile",
        action="store_true",
        help="Try torch.compile(model.step). Optional/experimental.",
    )
    p.add_argument(
        "--compile-mode",
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
    )
    p.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="Override cfg.inference.warmup_frames",
    )
    p.add_argument(
        "--save-suffix",
        default="_out",
        help="Output suffix before extension. Default: _out",
    )
    p.add_argument(
        "--keep-name",
        action="store_true",
        help="Save outputs with original filenames, no suffix",
    )

    # Optional TinyDelta3D trajectory predictor. If omitted, zero dp is used.
    p.add_argument(
        "--traj-config",
        default=None,
        help="Config for TinyDelta3D trajectory predictor",
    )
    p.add_argument("--traj-checkpoint", default=None, help="TinyDelta3D checkpoint")
    p.add_argument(
        "--traj-stats", default=None, help="Stats file for FixedDerivativePreprocess"
    )
    p.add_argument(
        "--traj-deploy",
        action="store_true",
        help="Convert RepDWConv3x3 blocks to deploy graph",
    )
    p.add_argument(
        "--traj-nonstrict",
        action="store_true",
        help="Load trajectory checkpoint with strict=False",
    )

    p.add_argument("--traj-fx", type=float, default=1.0)
    p.add_argument("--traj-fy", type=float, default=1.0)
    p.add_argument("--traj-z-ref", type=float, default=1.0)
    p.add_argument("--traj-sign-x", type=float, default=1.0)
    p.add_argument("--traj-sign-y", type=float, default=1.0)
    p.add_argument("--traj-output-scale", type=float, default=1.0)
    p.add_argument(
        "--traj-output-order",
        choices=["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"],
        default="xyz",
    )

    p.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides, e.g. data.test_resize=[720,1280]",
    )
    return p.parse_args()


def cfg_get(cfg, dotted: str, default=None):
    cur = cfg
    for part in dotted.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(part, default)
        else:
            cur = getattr(cur, part, default)
    return cur


def sorted_input_frames(input_dir: Path) -> list[Path]:
    files = [
        p
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    ]
    files.sort(key=lambda p: p.name)
    if not files:
        raise FileNotFoundError(f"No supported image files found in {input_dir}")
    return files


def maybe_resize(img: torch.Tensor, resize) -> torch.Tensor:
    if resize is None:
        return img
    size = tuple(int(x) for x in resize)
    x = img.unsqueeze(0)
    try:
        x = torch.nn.functional.interpolate(
            x, size=size, mode="bilinear", align_corners=False, antialias=True
        )
    except TypeError:
        x = torch.nn.functional.interpolate(
            x, size=size, mode="bilinear", align_corners=False
        )
    return x.squeeze(0)


def load_frame(
    path: Path, resize, device: torch.device, channels_last: bool
) -> torch.Tensor:
    x = (
        maybe_resize(load_image(path), resize)
        .unsqueeze(0)
        .to(device, non_blocking=True)
    )
    if channels_last and x.ndim == 4:
        x = x.contiguous(memory_format=torch.channels_last)
    return x


def save_tensor_image(x: torch.Tensor, path: Path) -> None:
    """Save a [1,3,H,W] or [3,H,W] float tensor in [0,1] without torchvision."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if x.ndim == 4:
        x = x[0]
    x = x.detach().float().clamp(0.0, 1.0).cpu()
    x = (x.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype("uint8")
    Image.fromarray(x).save(path)


def sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def build_learned_trajectory_adapter(args, device: torch.device):
    provided = [args.traj_config, args.traj_checkpoint, args.traj_stats]
    if not any(provided):
        print("[WARN] No trajectory predictor provided; using zero displacements.")
        return None
    if not all(provided):
        raise ValueError(
            "Provide all three trajectory args: --traj-config, --traj-checkpoint, --traj-stats"
        )

    traj_cfg = load_config(args.traj_config, [])
    traj_model = build_traj_model_from_config(traj_cfg).to(device)

    if getattr(traj_model, "K", 3) != 3:
        raise ValueError(
            f"TinyDelta3D adapter expects K=3, got K={traj_model.K}. "
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
        proj_cfg, output_mean=out_mean, output_std=out_std
    ).to(device)

    print(f"[INFO] Loaded trajectory model: {args.traj_checkpoint}")
    print(f"[INFO] Loaded trajectory stats: {args.traj_stats}")
    print(
        "[INFO] Projection: "
        f"fx={args.traj_fx}, fy={args.traj_fy}, z_ref={args.traj_z_ref}, "
        f"sign=({args.traj_sign_x},{args.traj_sign_y}), "
        f"scale={args.traj_output_scale}, order={args.traj_output_order}"
    )
    return LearnedTrajectory3DAdapter(
        model=traj_model, preprocess=preprocess, projector=projector
    )


def build_model(args, cfg, device: torch.device) -> torch.nn.Module:
    model = TSRNN_BT1().to(device)
    if args.fast_shift != "off":
        from tsrn_rtvd.models.shift_ops import apply_speed_patches

        apply_speed_patches(model, mode=args.fast_shift)

    load_checkpoint(args.checkpoint, model)
    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
        model._channels_last_inference = True
        print("[INFO] channels_last inference: enabled")
    else:
        model._channels_last_inference = False
        print("[INFO] channels_last inference: disabled")

    if args.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is unavailable in this PyTorch version")
        model.step = torch.compile(model.step, mode=args.compile_mode, fullgraph=False)  # type: ignore[method-assign]
        print(f"[INFO] torch.compile enabled for model.step, mode={args.compile_mode}")

    model.eval()
    print(f"[INFO] Loaded BT1 checkpoint: {args.checkpoint}")
    print(f"[INFO] fast_shift: {args.fast_shift}")
    print(
        f"[INFO] Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M"
    )
    return model


def warmup(
    model: torch.nn.Module,
    traj_adapter,
    device: torch.device,
    cfg,
    n: int,
    channels_last: bool,
) -> None:
    if n <= 0:
        return
    resize = cfg_get(cfg, "data.test_resize", None)
    H, W = (int(resize[0]), int(resize[1])) if resize else (720, 1280)
    dummy = torch.zeros(1, 3, H, W, device=device)
    if channels_last and device.type == "cuda":
        dummy = dummy.contiguous(memory_format=torch.channels_last)
    z2 = torch.zeros(1, 2, device=device)
    state = model.init_state(1, H, W, device)
    use_amp = bool(cfg_get(cfg, "inference.amp", True)) and device.type == "cuda"

    print(f"[INFO] Warm-up: {n} steps")
    with torch.no_grad():
        for _ in range(n):
            if traj_adapter is not None:
                traj_adapter.predict_left_right(dummy, dummy, dummy)
            with amp_context(device, use_amp):
                _, state = model.step(dummy, state, z2, z2)
    sync_cuda(device)


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    traj_adapter,
    device: torch.device,
    cfg,
    input_dir: Path,
    output_dir: Path,
    channels_last: bool,
    save_suffix: str,
    keep_name: bool,
) -> tuple[float, float]:
    frames = sorted_input_frames(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resize = cfg_get(cfg, "data.test_resize", None)
    use_amp = bool(cfg_get(cfg, "inference.amp", True)) and device.type == "cuda"

    print(f"[INFO] Input frames: {len(frames)} from {input_dir}")
    print(f"[INFO] Output dir: {output_dir}")

    B0 = load_frame(frames[0], resize, device, channels_last)
    _, _, H, W = B0.shape
    state = model.init_state(1, H, W, device)
    with amp_context(device, use_amp):
        feat0 = tuple(f.detach() for f in model.encoder(B0))
    state["feat_prev"] = feat0
    state["feat_cur"] = feat0
    state["B_cur"] = B0

    B_prev_raw = B0
    B_cur_raw = B0
    gpu_ms_total = 0.0
    wall_start = time.perf_counter()

    for i, frame_path in enumerate(frames):
        next_idx = min(len(frames) - 1, i + 1)
        B_next = load_frame(frames[next_idx], resize, device, channels_last)

        if traj_adapter is not None:
            dp_neg, dp_pos = traj_adapter.predict_left_right(
                B_prev_raw, B_cur_raw, B_next
            )
            if i == 0:
                dp_neg = torch.zeros_like(dp_neg)
            if next_idx == i:
                dp_pos = torch.zeros_like(dp_pos)
        else:
            dp_neg = torch.zeros(1, 2, device=device)
            dp_pos = torch.zeros(1, 2, device=device)

        sync_cuda(device)
        t0 = time.perf_counter()
        with amp_context(device, use_amp):
            pred, state = model.step(B_next, state, dp_neg, dp_pos)
        sync_cuda(device)
        gpu_ms_total += (time.perf_counter() - t0) * 1000.0

        if keep_name:
            out_name = frame_path.name
        else:
            out_name = f"{frame_path.stem}{save_suffix}.png"
        save_tensor_image(pred, output_dir / out_name)

        B_prev_raw = B_cur_raw.detach()
        B_cur_raw = B_next.detach()

        if (i + 1) % 25 == 0 or i == len(frames):
            print(f"  {i + 1:5d}/{len(frames)}  saved: {out_name}")

    wall_ms = (time.perf_counter() - wall_start) * 1000.0 / len(frames)
    gpu_ms = gpu_ms_total / len(frames)
    return gpu_ms, wall_ms


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    use_channels_last = bool(args.channels_last and device.type == "cuda")
    args.channels_last = use_channels_last

    model = build_model(args, cfg, device)
    traj_adapter = build_learned_trajectory_adapter(args, device)

    input_dir = Path(args.input_dir).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else input_dir.parent / f"{input_dir.name}_OUT"
    )

    warmup_steps = args.warmup_steps
    if warmup_steps is None:
        warmup_steps = int(cfg_get(cfg, "inference.warmup_frames", 50))
    warmup(model, traj_adapter, device, cfg, warmup_steps, use_channels_last)

    gpu_ms, wall_ms = run_inference(
        model=model,
        traj_adapter=traj_adapter,
        device=device,
        cfg=cfg,
        input_dir=input_dir,
        output_dir=output_dir,
        channels_last=use_channels_last,
        save_suffix=args.save_suffix,
        keep_name=args.keep_name,
    )

    print("\n[INFO] Inference finished")
    print(
        f"[INFO] GPU-only latency : {gpu_ms:.2f} ms/frame  ({1000.0 / gpu_ms:.1f} FPS)"
    )
    print(
        f"[INFO] Pipeline latency : {wall_ms:.2f} ms/frame  ({1000.0 / wall_ms:.1f} FPS)"
    )


if __name__ == "__main__":
    main()
