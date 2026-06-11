from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, get_args, get_origin

import yaml


@dataclass
class DataConfig:
    root_images: str = "/mnt/hdd2/deblurring_datasets/GOPRO_Large"
    root_trajectories: str = "/mnt/hdd2/deblurring_datasets/GOPRO_trajectories"
    image_subdir: str = "blur"
    train_split: str = "train"
    val_split: str = "test"
    K: int = 3
    height: int = 108
    width: int = 192
    target_frame: str = "center_camera"  # center_camera | world
    stride: int = 1
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    stats_path: str = "outputs/tinydelta3d_stats.pt"


@dataclass
class ModelConfig:
    in_ch: int = 3
    D: int = 128
    stem_channels: int = 16
    stage_channels: list[int] = field(default_factory=lambda: [24, 48, 96])
    stage_depths: list[int] = field(default_factory=lambda: [2, 3, 4])
    pooling: str = "gap"  # gap | attn_grid
    grid_pool: tuple[int, int] = (2, 3)
    temporal_blocks: int = 2
    temporal_kernel: int = 3
    deploy: bool = False


@dataclass
class LossConfig:
    use_uncertainty: bool = True
    smooth_l1_beta: float = 1.0
    uncertainty_reg: float = 0.05
    log_var_min: float = -6.0
    log_var_max: float = 3.0
    antisymmetry_weight: float = 0.02


@dataclass
class TrainConfig:
    output_dir: str = "outputs/tinydelta3d_k3_108x192"
    epochs: int = 50
    batch_size: int = 64
    num_workers: int = 8
    lr: float = 3.0e-4
    weight_decay: float = 1.0e-4
    grad_clip_norm: float = 1.0
    amp: bool = True
    compile: bool = False
    channels_last: bool = True
    seed: int = 1337
    log_every: int = 50
    val_every: int = 1
    save_every: int = 1
    persistent_workers: bool = True
    prefetch_factor: int = 4


@dataclass
class ExperimentConfig:
    name: str = "tinydelta3d_k3_108x192"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


T = TypeVar("T")


def _coerce_value(value: Any, target_type: Any) -> Any:
    origin = get_origin(target_type)
    args = get_args(target_type)

    if value is None:
        return None
    if origin is list or origin is list:
        inner = args[0] if args else Any
        return [_coerce_value(v, inner) for v in value]
    if origin is tuple or origin is tuple:
        inner = args[0] if args else Any
        return tuple(value)
    if origin is Optional:
        return value
    if origin is type(Optional[int]) or str(target_type).startswith("typing.Optional"):
        return value
    return value


def update_dataclass(obj: Any, updates: dict[str, Any]) -> Any:
    if not is_dataclass(obj):
        raise TypeError(f"Expected dataclass instance, got {type(obj)!r}")
    valid = {f.name: f for f in fields(obj)}
    for key, value in updates.items():
        if key not in valid:
            raise KeyError(f"Unknown config key {key!r} for {type(obj).__name__}")
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            update_dataclass(current, value)
        else:
            setattr(obj, key, _coerce_value(value, valid[key].type))
    return obj


def load_config(path: str | Path) -> ExperimentConfig:
    cfg = ExperimentConfig()
    with open(path, encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    update_dataclass(cfg, payload)
    validate_config(cfg)
    return cfg


def save_config(cfg: ExperimentConfig, path: str | Path) -> None:
    def to_dict(x: Any) -> Any:
        if is_dataclass(x):
            return {f.name: to_dict(getattr(x, f.name)) for f in fields(x)}
        if isinstance(x, tuple):
            return list(x)
        if isinstance(x, list):
            return [to_dict(v) for v in x]
        return x

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(to_dict(cfg), f, sort_keys=False)


def validate_config(cfg: ExperimentConfig) -> None:
    if cfg.data.K % 2 != 1:
        raise ValueError("K must be odd.")
    if cfg.data.K < 3:
        raise ValueError("K must be at least 3.")
    if cfg.model.in_ch != 3:
        raise ValueError(
            "Experiment 1 locks in_ch=3: gray, Sobel magnitude, Laplacian."
        )
    if cfg.model.D <= 0:
        raise ValueError("D must be positive.")
    if len(cfg.model.stage_channels) != len(cfg.model.stage_depths):
        raise ValueError("stage_channels and stage_depths must have the same length.")
    if cfg.data.target_frame not in {"center_camera", "world"}:
        raise ValueError("target_frame must be 'center_camera' or 'world'.")
    if cfg.model.pooling not in {"gap", "attn_grid"}:
        raise ValueError("model.pooling must be 'gap' or 'attn_grid'.")
