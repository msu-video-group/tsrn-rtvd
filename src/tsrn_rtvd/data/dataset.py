"""Minimal data helpers for BT1-only test.py.

Directory layout expected by test.py:
    <data.root>/test/<scene>/blur/*.png
    <data.root>/test/<scene>/sharp/*.png
    <data.traj_root>/test/<scene>/trajectories.npz
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image


def load_image(path: Path) -> torch.Tensor:
    """PNG/JPEG -> float32 tensor (3,H,W) in [0,1], without torchvision."""
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = map(float, q)
    n = qx * qx + qy * qy + qz * qz + qw * qw
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    x, y, z, w = qx, qy, qz, qw
    return np.array(
        [
            [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def compute_pixel_disp(
    poses: np.ndarray,
    intrinsics: np.ndarray,
    center_idx: int,
    neigh_idx: int,
    z_ref: float = 1.0,
    motion_scale: float = 1.0,
    sign_x: float = 1.0,
    sign_y: float = 1.0,
    pose_kind: str = "camera_center",
    rotation_convention: str = "c2w",
) -> np.ndarray:
    """ "Return approximate pixel displacement (dy, dx) for neigh -> center.

    pose_kind:
      "camera_center"  means poses[:, :3] are camera positions in world space.
      "colmap_extrinsic" means poses[:, :3] are world-to-camera extrinsic translations.

    rotation_convention:
      "c2w" means quaternion maps camera coordinates to world coordinates.
      "w2c" means quaternion maps world coordinates to camera coordinates.
    """
    try:
        if z_ref == 0:
            return np.zeros(2, dtype=np.float32)

        p0 = poses[center_idx]
        pj = poses[neigh_idx]
        t0, q0 = p0[:3], p0[3:7]
        tj, qj = pj[:3], pj[3:7]
        R0 = quat_to_rotmat(q0)
        Rj = quat_to_rotmat(qj)

        if pose_kind == "camera_center":
            # Your DA3/MONST3R files: tx,ty,tz are already global camera positions.
            C0 = t0
            Cj = tj
        elif pose_kind == "colmap_extrinsic":
            # COLMAP world-to-camera extrinsic convention: x_cam = R x_world + t.
            C0 = -R0.T @ t0
            Cj = -Rj.T @ tj
        else:
            raise ValueError(f"Unknown pose_kind={pose_kind!r}")

        # Camera translation from neighbor camera center to current/center camera center.
        delta_world = Cj - C0

        if rotation_convention == "c2w":
            # R maps camera -> world, so world -> center-camera is R.T.
            delta_cam = R0.T @ delta_world
        elif rotation_convention == "w2c":
            # R maps world -> camera.
            delta_cam = R0 @ delta_world
        else:
            raise ValueError(f"Unknown rotation_convention={rotation_convention!r}")

        K = intrinsics[center_idx]
        fx, fy = float(K[0, 0]), float(K[1, 1])
        dx = sign_x * motion_scale * (fx / z_ref) * delta_cam[0]
        dy = sign_y * motion_scale * (fy / z_ref) * delta_cam[1]
        out = np.array([dy, dx], dtype=np.float32)
        return out if np.isfinite(out).all() else np.zeros(2, dtype=np.float32)
    except Exception:
        return np.zeros(2, dtype=np.float32)


def load_traj_npz(npz_path: Path):
    try:
        d = np.load(str(npz_path), allow_pickle=True)
        names = [Path(str(n)).stem for n in d["image_names"]]
        poses = d["poses"].astype(np.float64)
        intrs = d["intrinsics"].astype(np.float64)
        return names, poses, intrs
    except Exception:
        return None


class SceneTraj:
    """One-scene trajectory index; missing/invalid files return zero displacement."""

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
        if npz_path is None or not Path(npz_path).exists():
            return
        result = load_traj_npz(Path(npz_path))
        if result is None:
            return
        names, poses, intrs = result
        self.stem_to_idx = {name: i for i, name in enumerate(names)}
        self.poses = poses
        self.intrinsics = intrs
        self.valid = True

    def get_disp(self, center_stem: str, neigh_stem: str) -> np.ndarray:
        if not self.valid:
            return np.zeros(2, dtype=np.float32)
        ci = self.stem_to_idx.get(center_stem)
        ni = self.stem_to_idx.get(neigh_stem)
        if ci is None or ni is None:
            return np.zeros(2, dtype=np.float32)
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


__all__ = ["load_image", "SceneTraj"]
