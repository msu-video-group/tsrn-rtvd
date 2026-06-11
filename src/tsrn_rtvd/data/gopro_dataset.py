"""
GoPro clip dataset for causal recurrent video deblurring training.

For each training sample we return a short consecutive clip of `seq_len`
frames from one scene.  The training loop unrolls the ConvGRU across the
clip with TBPTT (hidden state is zero-initialised at the start of every
clip and kept in the computation graph throughout the clip).

At test time we do not use this dataset — test.py iterates over scenes
directly to maintain a persistent hidden state across the entire scene.

Directory layout:
  GOPRO_Large/[train|test]/<scene>/blur/<frame>.png
  GOPRO_Large/[train|test]/<scene>/sharp/<frame>.png
  GOPRO_trajectories/[train|test]/<scene>/trajectories.npz

Trajectory shift fed to the model at each time step t:
  shift[t] = trans[t] - trans[t-1]   (3D, metres)
  shift[0] = zeros  (no previous frame at clip boundary)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

# ─────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────


def load_image(path: Path) -> torch.Tensor:
    """PNG → float32 tensor (3, H, W) in [0, 1]."""
    return TF.to_tensor(Image.open(path).convert("RGB"))


def load_trajectory(npz_path: Path) -> dict[str, np.ndarray]:
    """
    Returns {stem: np.ndarray(3,)} mapping frame stems to (tx, ty, tz).
    """
    data = np.load(str(npz_path), allow_pickle=True)
    names = data["image_names"]  # e.g. ['000001.png', ...]
    poses = data["poses"]  # (N, 7)  tx ty tz qx qy qz qw
    return {
        Path(str(n)).stem: poses[i, :3].astype(np.float32) for i, n in enumerate(names)
    }


# ─────────────────────────────────────────────────────────────────
# Training dataset  (clips)
# ─────────────────────────────────────────────────────────────────


class GoProClipDataset(Dataset):
    """
    Yields consecutive clips of `seq_len` frames for TBPTT training.

    Each item:
      blur_seq:  (seq_len, 3, H, W)  float32 [0,1]
      sharp_seq: (seq_len, 3, H, W)  float32 [0,1]
      traj_seq:  (seq_len, 3)        float32  trans[t]-trans[t-1], 0 at t=0
    """

    def __init__(
        self,
        root: str,
        traj_root: str,
        seq_len: int = 8,
        patch_size: int = 256,
        augment: bool = True,
    ):
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.augment = augment

        # (blur_dir, sharp_dir, [frame_paths], scene_name)
        self._clips: list[tuple[Path, Path, list[Path], str]] = []
        # scene_name → {stem: np.ndarray(3)}
        self._traj: dict[str, dict[str, np.ndarray]] = {}

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

            # Load trajectory
            npz = traj_root / scene.name / "trajectories.npz"
            traj: dict[str, np.ndarray] = {}
            if npz.exists():
                try:
                    traj = load_trajectory(npz)
                except Exception as e:
                    print(f"[WARN] traj load failed for {scene.name}: {e}")
            self._traj[scene.name] = traj

            # Every valid clip start index
            for start in range(len(frames) - self.seq_len + 1):
                self._clips.append(
                    (
                        blur_dir,
                        sharp_dir,
                        frames[start : start + self.seq_len],
                        scene.name,
                    )
                )

    # ── Augmentation ─────────────────────────────────────────────

    def _random_crop(self, imgs: list[torch.Tensor]) -> list[torch.Tensor]:
        _, H, W = imgs[0].shape
        s = self.patch_size
        top = torch.randint(0, H - s + 1, ()).item()
        left = torch.randint(0, W - s + 1, ()).item()
        return [img[:, top : top + s, left : left + s] for img in imgs]

    def _random_flip(self, imgs: list[torch.Tensor]) -> list[torch.Tensor]:
        if torch.rand(()) < 0.5:
            imgs = [TF.hflip(img) for img in imgs]
        if torch.rand(()) < 0.5:
            imgs = [TF.vflip(img) for img in imgs]
        return imgs

    # ── Dataset interface ─────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._clips)

    def __getitem__(self, idx: int) -> dict:
        blur_dir, sharp_dir, frames, scene_name = self._clips[idx]
        traj_map = self._traj[scene_name]

        blur_imgs = [load_image(f) for f in frames]
        sharp_imgs = [load_image(sharp_dir / f.name) for f in frames]

        # Augment (same transform for all frames in the clip)
        all_imgs = blur_imgs + sharp_imgs
        all_imgs = self._random_crop(all_imgs)
        if self.augment:
            all_imgs = self._random_flip(all_imgs)
        blur_imgs = all_imgs[: self.seq_len]
        sharp_imgs = all_imgs[self.seq_len :]

        # Trajectory shifts: shift[t] = trans[t] - trans[t-1]
        # shift[0] = 0  (no previous frame at clip start)
        stems = [f.stem for f in frames]
        traj_seq = np.zeros((self.seq_len, 3), dtype=np.float32)
        for t in range(1, self.seq_len):
            t_curr = traj_map.get(stems[t], np.zeros(3, np.float32))
            t_prev = traj_map.get(stems[t - 1], np.zeros(3, np.float32))
            traj_seq[t] = t_curr - t_prev

        return {
            "blur_seq": torch.stack(blur_imgs, dim=0),  # (L, 3, H, W)
            "sharp_seq": torch.stack(sharp_imgs, dim=0),  # (L, 3, H, W)
            "traj_seq": torch.from_numpy(traj_seq),  # (L, 3)
        }


# ─────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────


def build_train_dataset(cfg) -> GoProClipDataset:
    d = cfg.data
    return GoProClipDataset(
        root=str(Path(d.root) / "train"),
        traj_root=str(Path(d.traj_root) / "train"),
        seq_len=d.seq_len,
        patch_size=d.patch_size,
        augment=True,
    )
