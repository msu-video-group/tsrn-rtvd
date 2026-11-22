r"""Estimate GoPro trajectories from the original high-frame-rate PNGs.

Install the dev dependencies with `pip install -e '.[dev]'` and run on a CUDA GPU.
Supply the paired GOPRO_Large images to identify output frames.

    python tools/estimate_gopro_trajectories.py --raw-root /data/GOPRO_Large_all \
        --images-root /data/GOPRO_Large --output-root /data/GOPRO_estimated \
        --split train --scene GOPR0372_07_01

Each scene is written to <output-root>/<split>/<scene>/ as trajectories.npz and trajectories.txt.
"""

import argparse
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

MODEL_ID = "depth-anything/DA3NESTED-GIANT-LARGE"
MODEL_REVISION = "8615eefb62f2db4f8d6ebaa59160086981672829"
COARSE_STRIDE = 8
WINDOW = 64
OVERLAP = 32


def png_frames(folder):
    paths = sorted(folder.glob("*.png"), key=lambda p: int(p.stem))
    if not paths:
        raise FileNotFoundError(f"No PNG frames in {folder}")
    numbers = np.array([int(p.stem) for p in paths])
    if np.any(np.diff(numbers) != 1):
        raise ValueError(f"Missing or duplicate frames in {folder}")
    return paths


def source_indices(scene, raw, paired):
    # In the archived GoPro labels each blur maps to the first raw frame
    # of its exposure group. Some sequences have a nonzero numbering offset.
    stride = int(scene.split("_")[1])
    if stride <= 0 or len(raw) != len(paired) * stride:
        raise ValueError(
            f"{scene}: {len(raw)} raw frames for {len(paired)} groups of {stride}"
        )
    return np.arange(0, len(raw), stride)


def predict(model, paths):
    images = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"Could not read {path}")
        images.append(image)
    with torch.no_grad():
        pred = model.inference(images, use_ray_pose=True)
    w2c = np.broadcast_to(np.eye(4), (len(paths), 4, 4)).copy()
    w2c[:, :3] = pred.extrinsics
    return np.linalg.inv(w2c), pred.intrinsics


def align_chunk(chunk, start, coarse, keyframes):
    end = start + len(chunk)
    anchors = np.flatnonzero((keyframes >= start) & (keyframes < end))
    local = keyframes[anchors] - start
    if len(anchors) < 4:
        endpoints = np.array([start, end - 1])
        closest = np.abs(endpoints[:, None] - keyframes).argmin(axis=1)
        anchors = np.concatenate((anchors, closest))
        local = np.concatenate((local, endpoints - start))

    source = chunk[local, :3, 3]
    target = coarse[anchors, :3, 3]
    x = source - source.mean(axis=0)
    y = target - target.mean(axis=0)
    u, values, vt = np.linalg.svd(y.T @ x / len(x))
    signs = np.array([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])
    scale = (values @ signs) / ((x * x).sum() / len(x) + 1e-12)
    transform = np.eye(4)
    transform[:3, :3] = scale * ((u * signs) @ vt)
    transform[:3, 3] = target.mean(axis=0) - transform[:3, :3] @ source.mean(axis=0)
    return transform @ chunk


def estimate(model, paths):
    keyframes = np.arange(0, len(paths), COARSE_STRIDE)
    coarse, intrinsics = predict(model, [paths[i] for i in keyframes])
    camera = np.median(intrinsics.astype(np.float64), axis=0)
    accum = np.zeros((len(paths), 4, 4))
    weights = np.zeros(len(paths))
    starts = [0] if len(paths) <= WINDOW else range(0, len(paths), WINDOW - OVERLAP)
    for start in starts:
        batch = paths[start : start + WINDOW]
        chunk, _ = predict(model, batch)
        aligned = align_chunk(chunk, start, coarse, keyframes)
        i = np.arange(len(batch))
        edge = np.minimum(i, len(batch) - 1 - i)
        w = 0.5 + 0.5 * np.minimum(edge / (OVERLAP / 2), 1.0)
        region = slice(start, start + len(batch))
        accum[region] += aligned * w[:, None, None]
        weights[region] += w
    c2w = accum / weights[:, None, None]
    u, _, vt = np.linalg.svd(c2w[:, :3, :3])
    c2w[:, :3, :3] = u @ vt
    c2w[:, :3, 3] = gaussian_filter1d(c2w[:, :3, 3], sigma=1.0, axis=0)
    return c2w, camera


def export_scene(directory, raw, paired, indices, c2w, camera):
    selected = c2w[indices]
    quaternions = Rotation.from_matrix(selected[:, :3, :3]).as_quat()
    poses = np.column_stack((selected[:, :3, 3], quaternions))
    source = [raw[i] for i in indices]
    timestamps = [str(int(p.stem)) for p in source]
    np.savez_compressed(
        directory / "trajectories.npz",
        image_names=[p.name for p in paired],
        timestamps=timestamps,
        poses=poses,
        extrinsics=np.linalg.inv(selected)[:, :3].astype(np.float32),
        intrinsics=np.repeat(camera[None], len(indices), axis=0).astype(np.float32),
        source_frame_names=[p.name for p in source],
        source_indices=indices,
    )
    with (directory / "trajectories.txt").open("w", encoding="utf-8") as out:
        out.write("# image_name timestamp tx ty tz qx qy qz qw\n")
        for path, ts, pose in zip(paired, timestamps, poses):
            values = " ".join(f"{v:.8f}" for v in pose)
            out.write(f"{path.name} {ts} {values}\n")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--raw-root", required=True, type=Path, help="contains <split>/<scene>/*.png"
    )
    parser.add_argument(
        "--images-root",
        required=True,
        type=Path,
        help="contains <split>/<scene>/blur/*.png",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("train", "test"))
    parser.add_argument(
        "--scene", help="process one scene (default: all scenes in the split)"
    )
    parser.add_argument(
        "--weights",
        type=Path,
        help="local DA3 snapshot (default: download the pinned revision)",
    )
    args = parser.parse_args()
    split = args.images_root / args.split
    if args.scene:
        if Path(args.scene).name != args.scene:
            parser.error("--scene expects a GoPro scene name")
        scenes = [split / args.scene]
    else:
        scenes = sorted(p for p in split.iterdir() if p.is_dir())
    if not scenes:
        parser.error(f"No scenes in {split}")

    model = None
    for scene in scenes:
        dst = args.output_root / args.split / scene.name
        if dst.exists():
            outputs = [dst / "trajectories.npz", dst / "trajectories.txt"]
            if all(p.is_file() and p.stat().st_size > 0 for p in outputs):
                print(f"{scene.name}: already complete, skipping", flush=True)
                continue
            raise FileExistsError(
                f"Incomplete scene at {dst.resolve()}. Move it aside and rerun."
            )
        raw = png_frames(args.raw_root / args.split / scene.name)
        paired = png_frames(scene / "blur")
        indices = source_indices(scene.name, raw, paired)
        if model is None:
            from depth_anything_3.api import DepthAnything3
            from huggingface_hub import snapshot_download

            snapshot = args.weights or snapshot_download(
                MODEL_ID,
                revision=MODEL_REVISION,
                allow_patterns=["config.json", "model.safetensors"],
            )
            model = DepthAnything3.from_pretrained(str(snapshot)).cuda().eval()
        print(f"{scene.name}: estimating from {len(raw)} raw frames", flush=True)
        c2w, camera = estimate(model, raw)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=dst.parent) as tmp:
            directory = Path(tmp)
            export_scene(directory, raw, paired, indices, c2w, camera)
            os.rename(directory, dst)
        print(f"{dst}: wrote {len(indices)} paired poses", flush=True)


if __name__ == "__main__":
    main()
