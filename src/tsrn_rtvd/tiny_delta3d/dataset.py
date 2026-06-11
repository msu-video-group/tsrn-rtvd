from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .geometry import center_relative_translations


@dataclass(frozen=True)
class TrajectoryRow:
    image_name: str
    timestamp: float
    position: torch.Tensor  # [3]
    quat_xyzw: torch.Tensor  # [4]


@dataclass(frozen=True)
class WindowSample:
    scene: str
    center_image: str
    image_paths: tuple[Path, ...]
    positions: torch.Tensor  # [K, 3]
    quats_xyzw: torch.Tensor  # [K, 4]
    timestamps: torch.Tensor  # [K]


def read_trajectory_txt(path: str | Path) -> dict[str, TrajectoryRow]:
    path = Path(path)
    rows: dict[str, TrajectoryRow] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 9:
                raise ValueError(f"Malformed trajectory line in {path}: {line!r}")
            image_name = parts[0]
            timestamp = float(parts[1])
            tx, ty, tz, qx, qy, qz, qw = [float(v) for v in parts[2:]]
            rows[image_name] = TrajectoryRow(
                image_name=image_name,
                timestamp=timestamp,
                position=torch.tensor([tx, ty, tz], dtype=torch.float32),
                quat_xyzw=torch.tensor([qx, qy, qz, qw], dtype=torch.float32),
            )
    return rows


def _numeric_stem(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError:
        return 0


class GoproTrajectoryDataset(Dataset):
    """K-consecutive blurry-frame windows with K matched trajectory points.

    Directory contract:
      images:       root_images/[train|test]/*SCENE*/[blur|sharp]/*N*.png
      trajectories: root_trajectories/[train|test]/*SCENE*/trajectories.txt

    __getitem__ returns:
      x_rgb: [K, 3, H, W], float32 in [0, 1]
      y:     [K-1, 3], center-to-neighbor translation target
      meta:  dict with scene/window names
    """

    def __init__(
        self,
        root_images: str | Path,
        root_trajectories: str | Path,
        split: str,
        K: int = 3,
        height: int = 108,
        width: int = 192,
        image_subdir: str = "blur",
        target_frame: str = "center_camera",
        stride: int = 1,
        max_samples: int | None = None,
    ) -> None:
        super().__init__()
        if K % 2 != 1:
            raise ValueError("K must be odd.")
        self.root_images = Path(root_images)
        self.root_trajectories = Path(root_trajectories)
        self.split = split
        self.K = K
        self.center = K // 2
        self.neighbor_indices = [i for i in range(K) if i != self.center]
        self.height = height
        self.width = width
        self.image_subdir = image_subdir
        self.target_frame = target_frame
        self.stride = stride

        split_images = self.root_images / split
        split_traj = self.root_trajectories / split
        if not split_images.exists():
            raise FileNotFoundError(
                f"Image split directory does not exist: {split_images}"
            )
        if not split_traj.exists():
            raise FileNotFoundError(
                f"Trajectory split directory does not exist: {split_traj}"
            )

        self.samples = self._index_samples(split_images, split_traj)
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid K={K} windows found in {split_images} with trajectories under {split_traj}."
            )

    def _index_samples(
        self, split_images: Path, split_traj: Path
    ) -> list[WindowSample]:
        samples: list[WindowSample] = []
        scene_dirs = sorted([p for p in split_images.iterdir() if p.is_dir()])
        for scene_dir in scene_dirs:
            scene = scene_dir.name
            image_dir = scene_dir / self.image_subdir
            traj_path = split_traj / scene / "trajectories.txt"
            if not image_dir.exists() or not traj_path.exists():
                continue
            traj = read_trajectory_txt(traj_path)
            image_paths = sorted(image_dir.glob("*.png"), key=_numeric_stem)
            matched = [p for p in image_paths if p.name in traj]
            if len(matched) < self.K:
                continue

            # Consecutive in sorted image order, and every frame must be present in trajectories.
            for start in range(0, len(matched) - self.K + 1, self.stride):
                window = matched[start : start + self.K]
                # Reject silently if numeric stems skip inside the window. GOPRO is normally dense.
                stems = [_numeric_stem(p) for p in window]
                if any((b - a) != 1 for a, b in zip(stems[:-1], stems[1:])):
                    continue
                rows = [traj[p.name] for p in window]
                positions = torch.stack([r.position for r in rows], dim=0)
                quats = torch.stack([r.quat_xyzw for r in rows], dim=0)
                timestamps = torch.tensor(
                    [r.timestamp for r in rows], dtype=torch.float32
                )
                samples.append(
                    WindowSample(
                        scene=scene,
                        center_image=window[self.center].name,
                        image_paths=tuple(window),
                        positions=positions,
                        quats_xyzw=quats,
                        timestamps=timestamps,
                    )
                )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_rgb(self, path: Path) -> torch.Tensor:
        with Image.open(path) as im:
            im = im.convert("RGB")
            # PIL uses (width, height).
            im = im.resize((self.width, self.height), Image.BILINEAR)
            data = np.asarray(im, dtype=np.uint8)
            tensor = torch.from_numpy(data).permute(2, 0, 1).contiguous()
            return tensor.to(dtype=torch.float32).div_(255.0)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        x_rgb = torch.stack([self._load_rgb(p) for p in sample.image_paths], dim=0)
        y = center_relative_translations(
            sample.positions,
            sample.quats_xyzw,
            center=self.center,
            neighbor_indices=self.neighbor_indices,
            target_frame=self.target_frame,
        )
        return {
            "x_rgb": x_rgb,
            "target": y,
            "positions": sample.positions,
            "quats_xyzw": sample.quats_xyzw,
            "timestamps": sample.timestamps,
            "scene": sample.scene,
            "center_image": sample.center_image,
            "image_paths": [str(p) for p in sample.image_paths],
        }


def collate_samples(batch: list[dict]) -> dict:
    out = {
        "x_rgb": torch.stack([b["x_rgb"] for b in batch], dim=0),
        "target": torch.stack([b["target"] for b in batch], dim=0),
        "positions": torch.stack([b["positions"] for b in batch], dim=0),
        "quats_xyzw": torch.stack([b["quats_xyzw"] for b in batch], dim=0),
        "timestamps": torch.stack([b["timestamps"] for b in batch], dim=0),
        "scene": [b["scene"] for b in batch],
        "center_image": [b["center_image"] for b in batch],
        "image_paths": [b["image_paths"] for b in batch],
    }
    return out


def build_dataset(cfg, split: str, train: bool = False) -> GoproTrajectoryDataset:
    max_samples = cfg.data.max_train_samples if train else cfg.data.max_val_samples
    return GoproTrajectoryDataset(
        root_images=cfg.data.root_images,
        root_trajectories=cfg.data.root_trajectories,
        split=split,
        K=cfg.data.K,
        height=cfg.data.height,
        width=cfg.data.width,
        image_subdir=cfg.data.image_subdir,
        target_frame=cfg.data.target_frame,
        stride=cfg.data.stride,
        max_samples=max_samples,
    )
