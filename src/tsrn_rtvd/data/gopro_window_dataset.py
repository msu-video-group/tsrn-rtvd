"""
5-frame symmetric window dataset for TrajShiftGS.

For each centre frame t the dataset returns:
  blur_window : (5, 3, H, W)  – [I_{t-2}, I_{t-1}, I_t, I_{t+1}, I_{t+2}]
  sharp_center: (3, H, W)     – sharp I_t
  traj        : (4, 2)        – 2-D pixel displacements (dy, dx) in
                                 full-resolution pixel units for
                                 [I_{t-2}, I_{t-1}, I_{t+1}, I_{t+2}]
                                 relative to the centre frame.

Boundary handling (training): we only use frames with full context
  (index 2 … N-3 in a scene).
Boundary handling (test):     edge frames are replicated.

Trajectory computation
  Given COLMAP poses (tx, ty, tz, qx, qy, qz, qw) and intrinsics (3×3):
    C_j  = -R_j^T t_j          camera centre in world space
    ΔC   = R_0 (C_j – C_0)     relative translation in centre-camera frame
    Δpx  = fx · ΔC_x           pixel shift at unit depth (proxy for scene depth)
    Δpy  = fy · ΔC_y
  Returns (Δpy, Δpx) — (row shift, col shift) convention matching (dy, dx) in
  the shift network.
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
# Geometry helpers
# ─────────────────────────────────────────────────────────────────


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """(qx, qy, qz, qw) → (3, 3) rotation matrix."""
    qx, qy, qz, qw = q[0], q[1], q[2], q[3]
    return np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
        ],
        dtype=np.float64,
    )


def compute_pixel_disp(
    poses: np.ndarray,  # (N, 7)  tx ty tz qx qy qz qw
    intrinsics: np.ndarray,  # (N, 3, 3)
    center_idx: int,
    neigh_idx: int,
    z_ref: float = 1.0,
    motion_scale: float = 1.0,
    sign_x: float = 1.0,
    sign_y: float = 1.0,
    pose_kind: str = "camera_center",
    rotation_convention: str = "c2w",
) -> np.ndarray:
    """
    Approximate 2-D pixel displacement of camera neigh_idx
    relative to camera center_idx, returned as (dy, dx) in pixels.

    Derivation (see module docstring).
    Falls back to zeros if computation fails.
    """
    try:
        p0 = poses[center_idx]
        pj = poses[neigh_idx]

        t0, q0 = p0[:3], p0[3:7]
        tj, qj = pj[:3], pj[3:7]

        R0 = quat_to_rotmat(q0)
        Rj = quat_to_rotmat(qj)

        if pose_kind == "camera_center":
            C0 = t0
            Cj = tj
        elif pose_kind == "colmap_extrinsic":
            C0 = -R0.T @ t0
            Cj = -Rj.T @ tj
        else:
            raise ValueError(f"Unknown pose_kind={pose_kind!r}")

        # Camera translation from neighbor to center.
        delta_world = Cj - C0

        if rotation_convention == "c2w":
            delta_cam = R0.T @ delta_world
        elif rotation_convention == "w2c":
            delta_cam = R0 @ delta_world
        else:
            raise ValueError(f"Unknown rotation_convention={rotation_convention!r}")

        K = intrinsics[center_idx]  # (3, 3)
        fx, fy = float(K[0, 0]), float(K[1, 1])

        if z_ref == 0:
            return np.zeros(2, dtype=np.float32)

        dx = sign_x * motion_scale * (fx / z_ref) * delta_cam[0]
        dy = sign_y * motion_scale * (fy / z_ref) * delta_cam[1]

        result = np.array([dy, dx], dtype=np.float32)
        # Guard: degenerate quaternions (near-zero norm) produce NaN rotation
        # matrices without raising a Python exception.
        if not np.isfinite(result).all():
            return np.zeros(2, dtype=np.float32)
        return result

    except Exception:
        return np.zeros(2, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────


def load_image(path: Path) -> torch.Tensor:
    return TF.to_tensor(Image.open(path).convert("RGB"))  # (3,H,W) float32


def load_traj_npz(npz_path: Path):
    """Returns (image_names, poses, intrinsics) or None on failure."""
    try:
        d = np.load(str(npz_path), allow_pickle=True)
        names = [Path(str(n)).stem for n in d["image_names"]]  # list of stems
        poses = d["poses"].astype(np.float64)  # (N, 7)
        intrs = d["intrinsics"].astype(np.float64)  # (N, 3, 3)
        return names, poses, intrs
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────
# Scene index: maps frame stem → (pose_row, intrinsics_row)
# ─────────────────────────────────────────────────────────────────


class SceneTraj:
    """Holds trajectory data for one scene and answers pixel-disp queries."""

    def __init__(
        self,
        npz_path: Path | None,
        z_ref: float = 1.0,
        motion_scale: float = 1.0,
        sign_x: float = 1.0,
        sign_y: float = 1.0,
        pose_kind: str = "camera_center",
        rotation_convention: str = "c2w",
    ):
        self.valid = False
        self.z_ref = z_ref
        self.motion_scale = motion_scale
        self.sign_x = sign_x
        self.sign_y = sign_y
        self.pose_kind = pose_kind
        self.rotation_convention = rotation_convention
        if npz_path is None or not npz_path.exists():
            return
        result = load_traj_npz(npz_path)
        if result is None:
            return
        names, poses, intrs = result
        self.stem_to_idx = {name: i for i, name in enumerate(names)}
        self.poses = poses
        self.intrinsics = intrs
        self.valid = True

    def get_disp(self, center_stem: str, neigh_stem: str) -> np.ndarray:
        if not self.valid:
            return np.zeros(2, np.float32)
        ci = self.stem_to_idx.get(center_stem)
        ni = self.stem_to_idx.get(neigh_stem)
        if ci is None or ni is None:
            return np.zeros(2, np.float32)
        return compute_pixel_disp(
            self.poses,
            self.intrinsics,
            ci,
            ni,
            z_ref=self.z_ref,
            motion_scale=self.motion_scale,
            sign_x=self.sign_x,
            sign_y=self.sign_y,
            pose_kind=self.pose_kind,
            rotation_convention=self.rotation_convention,
        )


# ─────────────────────────────────────────────────────────────────
# Training dataset — symmetric 5-frame windows
# ─────────────────────────────────────────────────────────────────


class GoProWindowDataset(Dataset):
    """
    Each item is one centre frame with its ±2 context.

    Item dict:
      blur_window:  (5, 3, H, W)  float32 [0,1]
      sharp_center: (3, H, W)     float32 [0,1]
      traj:         (4, 2)        float32 (dy, dx) pixel displacements
                                   order: [I_{-2}, I_{-1}, I_{+1}, I_{+2}]
    """

    RADIUS = 2  # half-window: uses frames t-2 … t+2

    def __init__(
        self,
        root: str,
        traj_root: str,
        patch_size: int | None = 128,
        augment: bool = True,
    ):
        self.patch_size = patch_size
        self.augment = augment

        # (blur_paths[0..N-1], sharp_dir, center_indices, scene_traj)
        self._samples: list[tuple[list[Path], Path, int, SceneTraj]] = []

        self._build_index(Path(root), Path(traj_root))

    def _build_index(self, root: Path, traj_root: Path):
        R = self.RADIUS
        for scene in sorted(d for d in root.iterdir() if d.is_dir()):
            blur_dir = scene / "blur"
            sharp_dir = scene / "sharp"
            if not (blur_dir.exists() and sharp_dir.exists()):
                continue

            frames = sorted(blur_dir.glob("*.png"))
            if len(frames) < 2 * R + 1:
                continue

            npz_path = traj_root / scene.name / "trajectories.npz"
            traj = SceneTraj(npz_path if npz_path.exists() else None)

            for center_idx in range(R, len(frames) - R):
                self._samples.append((frames, sharp_dir, center_idx, traj))

    # ── Augmentation ─────────────────────────────────────────────

    def _apply_aug(self, imgs: list[torch.Tensor]):
        """Identical crop + flip applied to all frames."""
        hflip = False
        vflip = False
        # Random crop
        if self.patch_size is not None:
            _, H, W = imgs[0].shape
            s = self.patch_size
            top = torch.randint(0, H - s + 1, ()).item()
            left = torch.randint(0, W - s + 1, ()).item()
            imgs = [img[:, top : top + s, left : left + s] for img in imgs]

        # Random horizontal / vertical flip (same transform for all frames)
        if self.augment:
            if torch.rand(()) < 0.5:
                hflip = True
                imgs = [TF.hflip(img) for img in imgs]
            if torch.rand(()) < 0.5:
                vflip = True
                imgs = [TF.vflip(img) for img in imgs]

        return imgs, hflip, vflip

    # ── Dataset interface ─────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        frames, sharp_dir, ci, traj = self._samples[idx]
        R = self.RADIUS

        window_paths = frames[ci - R : ci + R + 1]  # 5 paths
        center_path = frames[ci]
        center_stem = center_path.stem

        # Load blurry window + sharp centre
        blur_imgs = [load_image(p) for p in window_paths]
        sharp_img = load_image(sharp_dir / center_path.name)

        # Augment (same for all 5 blurry + 1 sharp)
        all_imgs, hflip, vflip = self._apply_aug(blur_imgs + [sharp_img])
        blur_imgs = all_imgs[:5]
        sharp_img = all_imgs[5]

        # Trajectory displacements for the 4 neighbours
        # Indices in window: 0=t-2, 1=t-1,  2=t (centre), 3=t+1, 4=t+2
        neigh_offsets = [-2, -1, +1, +2]
        traj_disp = np.zeros((4, 2), dtype=np.float32)
        for k, off in enumerate(neigh_offsets):
            neigh_stem = frames[ci + off].stem
            traj_disp[k] = traj.get_disp(center_stem, neigh_stem)
        # traj_disp is (dy, dx). Image flips must flip the corresponding motion axis.
        if hflip:
            traj_disp[:, 1] *= -1.0  # dx
        if vflip:
            traj_disp[:, 0] *= -1.0  # dy

        return {
            "blur_window": torch.stack(blur_imgs, dim=0),  # (5, 3, H, W)
            "sharp_center": sharp_img,  # (3, H, W)
            "traj": torch.from_numpy(traj_disp),  # (4, 2)
        }


# ─────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────


def build_window_dataset(cfg, split: str = "train") -> GoProWindowDataset:
    d = cfg.data
    return GoProWindowDataset(
        root=str(Path(d.root) / split),
        traj_root=str(Path(d.traj_root) / split),
        patch_size=d.patch_size if split == "train" else None,
        augment=(split == "train"),
    )
