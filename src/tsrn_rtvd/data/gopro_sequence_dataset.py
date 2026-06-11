"""
Sequence dataset for TS-RNN-Lite-v1 training.

Returns consecutive clips of `seq_len` frames for TBPTT training.

Per item:
  blur_seq : (L, 3, H, W)  consecutive blurry frames
  sharp_seq: (L, 3, H, W)  corresponding sharp frames
  dp_seq   : (L, 2)        (dx, dy) displacement t-1→t in full-res pixels
                            dp_seq[0] = zeros  (no previous frame at clip start)

Convention: dx > 0 = right, dy > 0 = down.
SceneTraj.get_disp(center=t, neigh=t-1) returns (dy_t-1_to_t, dx_t-1_to_t).
We negate because the network wants dp = pose_{t-1}→t, i.e. how much the
camera moved FROM t-1 TO t, to align H_{t-1} features into frame t's viewpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from .gopro_dataset import load_image
from .gopro_window_dataset import SceneTraj

# ─────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────


class GoProSequenceDataset(Dataset):
    """
    Returns overlapping clips of `seq_len` consecutive frames.
    Stride = 1 → every frame appears as a clip centre.
    """

    def __init__(
        self,
        root: str,
        traj_root: str,
        seq_len: int = 8,
        patch_size: int | None = 256,
        augment: bool = True,
        traj_z_ref: float = 10.0,
        traj_motion_scale: float = 1.0,
        traj_sign_x: float = 1.0,
        traj_sign_y: float = 1.0,
        traj_pose_kind: str = "camera_center",
        traj_rotation_convention: str = "c2w",
    ):
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.augment = augment
        self.traj_z_ref = traj_z_ref
        self.traj_motion_scale = traj_motion_scale
        self.traj_sign_x = traj_sign_x
        self.traj_sign_y = traj_sign_y
        self.traj_pose_kind = traj_pose_kind
        self.traj_rotation_convention = traj_rotation_convention

        # (frames list, sharp_dir, start_idx, SceneTraj)
        self._clips: list[tuple[list[Path], Path, int, SceneTraj]] = []
        self._build_index(Path(root), Path(traj_root))

    def _build_index(self, root: Path, traj_root: Path):
        for scene in sorted(d for d in root.iterdir() if d.is_dir()):
            blur_dir = scene / "blur"
            sharp_dir = scene / "sharp"
            if not (blur_dir.exists() and sharp_dir.exists()):
                continue
            frames = sorted(blur_dir.glob("*.png"))
            if len(frames) < self.seq_len:
                continue
            npz = traj_root / scene.name / "trajectories.npz"
            traj = SceneTraj(
                npz if npz.exists() else None,
                z_ref=self.traj_z_ref,
                motion_scale=self.traj_motion_scale,
                sign_x=self.traj_sign_x,
                sign_y=self.traj_sign_y,
                pose_kind=self.traj_pose_kind,
                rotation_convention=self.traj_rotation_convention,
            )
            for start in range(len(frames) - self.seq_len + 1):
                self._clips.append((frames, sharp_dir, start, traj))

    def _augment(self, imgs: list[torch.Tensor]):
        hflip = False
        vflip = False
        if self.patch_size is not None:
            _, H, W = imgs[0].shape
            s = self.patch_size
            top = torch.randint(0, H - s + 1, ()).item()
            left = torch.randint(0, W - s + 1, ()).item()
            imgs = [img[:, top : top + s, left : left + s] for img in imgs]
        if self.augment:
            if torch.rand(()) < 0.5:
                hflip = True
                imgs = [TF.hflip(img) for img in imgs]
            if torch.rand(()) < 0.5:
                vflip = True
                imgs = [TF.vflip(img) for img in imgs]
        return imgs, hflip, vflip

    def __len__(self) -> int:
        return len(self._clips)

    def __getitem__(self, idx: int) -> dict:
        frames, sharp_dir, start, traj = self._clips[idx]
        L = self.seq_len

        window = frames[start : start + L]

        blur_imgs = [load_image(f) for f in window]
        sharp_imgs = [load_image(sharp_dir / f.name) for f in window]

        # Same spatial aug for all frames
        all_imgs, hflip, vflip = self._augment(blur_imgs + sharp_imgs)
        blur_imgs = all_imgs[:L]
        sharp_imgs = all_imgs[L:]

        # dp_seq[t] = displacement from frame t-1 to frame t, in full-res pixels
        # dp_seq[0] = 0 (no previous frame at clip start)
        # SceneTraj.get_disp(center=t, neigh=t-1) → (dy, dx) for neigh→center
        # We want (dx, dy) in the convention dx>0=right, dy>0=down
        dp_seq = np.zeros((L, 2), dtype=np.float32)
        for t in range(1, L):
            stem_t = window[t].stem
            stem_prev = window[t - 1].stem
            # get_disp returns (dy, dx): how much prev moved relative to center
            dy_dx = traj.get_disp(stem_t, stem_prev)
            dp_seq[t, 0] = float(dy_dx[1])  # dx
            dp_seq[t, 1] = float(dy_dx[0])  # dy
        # dp_seq is (dx, dy). Image flips must flip the corresponding motion axis.
        if hflip:
            dp_seq[:, 0] *= -1.0
        if vflip:
            dp_seq[:, 1] *= -1.0
        dp_seq = np.nan_to_num(dp_seq, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            "blur_seq": torch.stack(blur_imgs, dim=0),  # (L, 3, H, W)
            "sharp_seq": torch.stack(sharp_imgs, dim=0),  # (L, 3, H, W)
            "dp_seq": torch.from_numpy(dp_seq),  # (L, 2)
        }


# ─────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────


def build_sequence_dataset(cfg, split: str = "train") -> GoProSequenceDataset:
    d = cfg.data
    return GoProSequenceDataset(
        root=str(Path(d.root) / split),
        traj_root=str(Path(d.traj_root) / split),
        seq_len=d.seq_len,
        patch_size=d.patch_size if split == "train" else None,
        augment=split == "train",
        traj_z_ref=getattr(d, "traj_z_ref", 10.0),
        traj_motion_scale=getattr(d, "traj_motion_scale", 1.0),
        traj_sign_x=getattr(d, "traj_sign_x", 1.0),
        traj_sign_y=getattr(d, "traj_sign_y", 1.0),
        traj_pose_kind=getattr(d, "traj_pose_kind", "camera_center"),
        traj_rotation_convention=getattr(d, "traj_rotation_convention", "c2w"),
    )
