from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from tsrn_rtvd.hub import DEFAULT_HF_REPO_ID, resolve_artifacts

from .display import DisplayStats, FpsMeter, compose_side_by_side
from .frame_io import resize_for_model
from .pipeline import ModelOptions, TrajectoryOptions, TSRNNWebcamProcessor
from .repo import default_repo_root


def build_parser() -> argparse.ArgumentParser:
    repo_root = default_repo_root()
    parser = argparse.ArgumentParser(description="Live webcam demo for TSRNN_BT1.")

    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--hf-repo-id", default=DEFAULT_HF_REPO_ID)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--device", choices=["auto", "mps", "cuda", "cpu"], default="auto"
    )
    parser.add_argument(
        "--fast-shift", choices=["roll_match", "exact"], default="exact"
    )
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--compile", dest="compile_step", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=3)

    parser.add_argument("--no-trajectory", action="store_true")
    parser.add_argument(
        "--traj-config",
        type=Path,
        default=None,
    )
    parser.add_argument("--traj-checkpoint", type=Path, default=None)
    parser.add_argument("--traj-stats", type=Path, default=None)
    parser.add_argument("--traj-deploy", action="store_true")
    parser.add_argument("--traj-nonstrict", action="store_true")
    parser.add_argument("--traj-full-res", action="store_true")
    parser.add_argument("--traj-fx", type=float, default=1000.0)
    parser.add_argument("--traj-fy", type=float, default=1000.0)
    parser.add_argument("--traj-z-ref", type=float, default=10.0)
    parser.add_argument("--traj-sign-x", type=float, default=1.0)
    parser.add_argument("--traj-sign-y", type=float, default=1.0)
    parser.add_argument(
        "--traj-output-order",
        choices=["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"],
        default="xyz",
    )
    parser.add_argument("--traj-output-scale", type=float, default=1.0)

    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--max-width", type=int, default=640)
    parser.add_argument("--max-height", type=int, default=0)
    parser.add_argument("--window-name", default="TSRNN Webcam Demo")
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--display-scale", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    artifacts = resolve_artifacts(
        args.repo_root,
        hf_repo_id=args.hf_repo_id,
        with_trajectory=not args.no_trajectory,
        local_files_only=args.local_files_only,
    )
    config = args.config or artifacts.config
    checkpoint = args.checkpoint or artifacts.checkpoint
    traj_config = args.traj_config or artifacts.traj_config or Path()
    traj_checkpoint = args.traj_checkpoint or artifacts.traj_checkpoint or Path()
    traj_stats = args.traj_stats or artifacts.traj_stats or Path()

    model_options = ModelOptions(
        repo_root=artifacts.repo_root or args.repo_root,
        config=config,
        checkpoint=checkpoint,
        device=args.device,
        fast_shift=args.fast_shift,
        channels_last=args.channels_last,
        compile_step=args.compile_step,
        amp=args.amp,
        warmup_steps=args.warmup_steps,
    )
    traj_options = TrajectoryOptions(
        enabled=not args.no_trajectory,
        config=traj_config,
        checkpoint=traj_checkpoint,
        stats=traj_stats,
        deploy=args.traj_deploy,
        nonstrict=args.traj_nonstrict,
        fx=args.traj_fx,
        fy=args.traj_fy,
        z_ref=args.traj_z_ref,
        sign_x=args.traj_sign_x,
        sign_y=args.traj_sign_y,
        output_order=args.traj_output_order,
        output_scale=args.traj_output_scale,
        resize_to_config=not args.traj_full_res,
    )

    processor = TSRNNWebcamProcessor(model_options, traj_options)
    print(f"[demo] device: {processor.device}")
    print(f"[demo] checkpoint: {checkpoint}")
    print(f"[demo] trajectory: {'enabled' if traj_options.enabled else 'disabled'}")

    cap = _open_camera(args.camera, args.width, args.height)
    fps_meter = FpsMeter()

    try:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Camera opened, but did not return a frame.")
        frame = _prepare_frame(frame, args)
        processor.reset(frame)

        while True:
            ok, frame = cap.read()
            if not ok:
                print("[demo] camera frame was not received; stopping.")
                break

            frame = _prepare_frame(frame, args)
            loop_started = time.perf_counter()
            result = processor.process_next(frame)
            sample_fps = 1.0 / max(time.perf_counter() - loop_started, 1e-6)
            fps = fps_meter.update(sample_fps)

            h, w = result.source_bgr.shape[:2]
            panel = compose_side_by_side(
                result.source_bgr,
                result.output_bgr,
                DisplayStats(
                    device=str(processor.device),
                    frame_index=result.frame_index,
                    fps=fps,
                    model_ms=result.model_ms,
                    total_ms=result.total_ms,
                    size=f"{w}x{h}",
                ),
            )
            if args.display_scale != 1.0:
                panel = cv2.resize(
                    panel, None, fx=args.display_scale, fy=args.display_scale
                )

            cv2.imshow(args.window_name, panel)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("r"):
                processor.reset(frame)
                fps_meter = FpsMeter()
                print("[demo] stream state was reset.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def _open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {index}. On macOS, check camera permissions for the launching terminal."
        )
    return cap


def _prepare_frame(frame, args):
    if args.mirror:
        frame = cv2.flip(frame, 1)
    return resize_for_model(frame, max_width=args.max_width, max_height=args.max_height)
