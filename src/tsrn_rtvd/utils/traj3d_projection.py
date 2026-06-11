from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass
class TrajProjectionConfig:
    """
    Weak-perspective 3D translation -> global 2D pixel displacement.

    Assumes delta3 is translation-like:
      delta3[..., 0] = tx
      delta3[..., 1] = ty
      delta3[..., 2] = tz

    Projection:
      dx = sign_x * fx / z_ref * tx
      dy = sign_y * fy / z_ref * ty

    If your trajectory model was trained with normalized 3D labels,
    use output_mean/output_std for denormalization before projection.
    """

    fx: float = 1.0
    fy: float = 1.0
    z_ref: float = 1.0

    sign_x: float = 1.0
    sign_y: float = 1.0

    output_scale: float = 1.0
    output_order: str = "xyz"  # "xyz", "xzy", "yxz", etc.


def _as_1d_tensor(x, device=None, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(x, dtype=dtype, device=device).flatten()


def load_stats_file(path: str | Path) -> dict:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".npz":
        z = np.load(str(path))
        return {k: z[k] for k in z.files}

    if suffix == ".json":
        with open(path) as f:
            return json.load(f)

    obj = torch.load(str(path), map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"Expected stats checkpoint to be dict-like, got {type(obj)}")
    return obj


def find_mean_std(stats: dict, prefixes: Sequence[str] = ("", "preprocess_", "input_")):
    """
    Accepts common key layouts:
      mean/std
      preprocess_mean/preprocess_std
      input_mean/input_std
      preprocess: {mean, std}
      stats: {mean, std}
    """

    candidates = [stats]

    for nested_key in ["preprocess", "preprocess_stats", "input_stats", "stats"]:
        if isinstance(stats.get(nested_key), dict):
            candidates.append(stats[nested_key])

    for d in candidates:
        for prefix in prefixes:
            mean_key = f"{prefix}mean"
            std_key = f"{prefix}std"
            if mean_key in d and std_key in d:
                return d[mean_key], d[std_key]

    raise KeyError(
        "Could not find mean/std in stats file. "
        "Expected mean/std, preprocess_mean/preprocess_std, input_mean/input_std, "
        "or nested preprocess/stats dict."
    )


def find_output_mean_std(stats: dict):
    """
    Optional output-label denormalization.

    Accepts:
      output_mean/output_std
      label_mean/label_std
      target_mean/target_std
      output_stats: {mean, std}
      label_stats: {mean, std}
    """

    candidates = [stats]

    for nested_key in ["output_stats", "label_stats", "target_stats"]:
        if isinstance(stats.get(nested_key), dict):
            candidates.append(stats[nested_key])

    key_pairs = [
        ("delta_mean", "delta_std"),
        ("output_mean", "output_std"),
        ("label_mean", "label_std"),
        ("target_mean", "target_std"),
        ("mean", "std"),
    ]

    for d in candidates:
        for mk, sk in key_pairs:
            if mk in d and sk in d:
                return d[mk], d[sk]

    return None, None


def strip_state_dict_prefixes(sd: dict) -> dict:
    prefixes = ("module.", "model.", "net.", "trajectory_model.", "traj_model.")

    out = {}
    for k, v in sd.items():
        kk = k
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if kk.startswith(p):
                    kk = kk[len(p) :]
                    changed = True
        out[kk] = v
    return out


def extract_state_dict(ckpt) -> dict:
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model_state_dict", "model", "net"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return strip_state_dict_prefixes(ckpt[key])

        # Raw state dict fallback.
        if all(torch.is_tensor(v) for v in ckpt.values()):
            return strip_state_dict_prefixes(ckpt)

    raise TypeError("Unsupported trajectory checkpoint format.")


class Translation3DToPixelDP(nn.Module):
    """
    Converts TinyDelta3D's 3D output to TSRNN-compatible 2D dp.

    Input:
      delta3: [B, 3]

    Output:
      dp: [B, 2] = [dx, dy]
    """

    def __init__(
        self,
        cfg: TrajProjectionConfig,
        output_mean=None,
        output_std=None,
    ):
        super().__init__()
        if cfg.z_ref == 0:
            raise ValueError("z_ref must be nonzero.")
        if sorted(cfg.output_order) != ["x", "y", "z"]:
            raise ValueError("output_order must be a permutation of 'xyz'.")

        self.cfg = cfg
        self.index_x = cfg.output_order.index("x")
        self.index_y = cfg.output_order.index("y")
        self.index_z = cfg.output_order.index("z")

        if output_mean is None:
            output_mean = torch.zeros(3)
        if output_std is None:
            output_std = torch.ones(3)

        self.register_buffer(
            "output_mean", _as_1d_tensor(output_mean).view(1, 3), persistent=False
        )
        self.register_buffer(
            "output_std", _as_1d_tensor(output_std).view(1, 3), persistent=False
        )

    def forward(self, delta3: torch.Tensor) -> torch.Tensor:
        if delta3.ndim != 2 or delta3.shape[1] < 3:
            raise ValueError(
                f"Expected delta3 shape [B, >=3], got {tuple(delta3.shape)}"
            )

        delta3 = delta3[:, :3]
        delta3 = delta3 * self.output_std.to(delta3.dtype) + self.output_mean.to(
            delta3.dtype
        )
        delta3 = delta3 * self.cfg.output_scale

        tx = delta3[:, self.index_x]
        ty = delta3[:, self.index_y]

        dx = self.cfg.sign_x * (self.cfg.fx / self.cfg.z_ref) * tx
        dy = self.cfg.sign_y * (self.cfg.fy / self.cfg.z_ref) * ty

        return torch.stack([dx, dy], dim=1).float()


class LearnedTrajectory3DAdapter:
    """
    Adapter used by test_tsrnn_new.py.

    TinyDelta3D input:
      [B, 3, 3, H, W] = [left, center, right]

    TinyDelta3D output:
      delta3: [B, 2, 3]
        index 0 = left-neighbor delta
        index 1 = right-neighbor delta

    Returned dp:
      dp_left_to_center:  [B, 2]
      dp_right_to_center: [B, 2]
    """

    def __init__(
        self,
        model: nn.Module,
        preprocess: nn.Module,
        projector: Translation3DToPixelDP,
    ):
        self.model = model.eval()
        self.preprocess = preprocess.eval()
        self.projector = projector.eval()

    @torch.no_grad()
    def predict_left_right(
        self,
        B_left: torch.Tensor,
        B_center: torch.Tensor,
        B_right: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.stack([B_left, B_center, B_right], dim=1)
        x = self.preprocess(x)

        out = self.model(x)
        delta3 = out[0] if isinstance(out, (tuple, list)) else out

        if delta3.ndim != 3 or delta3.shape[1] != 2 or delta3.shape[2] < 3:
            raise ValueError(
                f"Expected TinyDelta3D output [B, 2, >=3], got {tuple(delta3.shape)}"
            )

        dp_left_to_center = self.projector(delta3[:, 0, :3])
        dp_right_to_center = self.projector(delta3[:, 1, :3])

        return dp_left_to_center, dp_right_to_center

    @torch.no_grad()
    def predict_prev_to_cur(
        self,
        B_prev: torch.Tensor,
        B_cur: torch.Tensor,
        B_next: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if B_next is None:
            B_next = B_cur

        dp_prev_to_cur, _ = self.predict_left_right(B_prev, B_cur, B_next)
        return dp_prev_to_cur
