"""Minimal utilities for the BT1-only package."""

from __future__ import annotations

import os
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


class Config(dict):
    """dict with attribute access, recursively wrapping nested dicts."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _wrap(x: Any) -> Any:
    if isinstance(x, dict):
        return Config({k: _wrap(v) for k, v in x.items()})
    if isinstance(x, list):
        return [_wrap(v) for v in x]
    return x


def _set_dot(cfg: dict, key: str, value: Any) -> None:
    cur = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def _parse_value(v: str) -> Any:
    try:
        return yaml.safe_load(v)
    except Exception:
        return v


def load_config(path: str, overrides: list[str] | None = None) -> Config:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if overrides:
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"Override must be key=value, got: {item}")
            k, v = item.split("=", 1)
            _set_dot(data, k, _parse_value(v))
    return _wrap(data)


def _to_plain(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _to_plain(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_plain(v) for v in x]
    return x


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    state: dict, output_dir: str, filename: str = "last.pth", is_best: bool = False
):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    torch.save(state, path)
    if is_best:
        shutil.copyfile(path, os.path.join(output_dir, "best.pth"))


def load_checkpoint(path: str, model: torch.nn.Module, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    if optimizer is not None and isinstance(ckpt, dict) and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and isinstance(ckpt, dict) and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if isinstance(ckpt, dict):
        return ckpt.get("epoch", 0), ckpt.get("best_psnr", 0.0)
    return 0, 0.0


def make_output_dir(cfg) -> Path:
    out = Path(cfg.experiment.output_dir) / cfg.experiment.name
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(_to_plain(cfg), f, sort_keys=False, allow_unicode=True)
    return out


class Logger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", buffering=1, encoding="utf-8")

    def __call__(self, msg: str):
        print(msg)
        self._f.write(msg + "\n")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self._f.flush()
        self._f.close()
