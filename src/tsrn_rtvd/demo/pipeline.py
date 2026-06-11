from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .device import autocast_context, resolve_device, synchronize
from .frame_io import bgr_to_tensor, tensor_to_bgr
from .repo import add_repo_to_path


@dataclass(frozen=True)
class ModelOptions:
    repo_root: Path
    config: Path
    checkpoint: Path
    device: str = "auto"
    fast_shift: str = "roll_match"
    channels_last: bool = False
    compile_step: bool = False
    amp: bool = False
    warmup_steps: int = 3


@dataclass(frozen=True)
class TrajectoryOptions:
    enabled: bool
    config: Path
    checkpoint: Path
    stats: Path
    deploy: bool = False
    nonstrict: bool = False
    fx: float = 1000.0
    fy: float = 1000.0
    z_ref: float = 10.0
    sign_x: float = 1.0
    sign_y: float = 1.0
    output_order: str = "xyz"
    output_scale: float = 1.0
    resize_to_config: bool = True


@dataclass(frozen=True)
class ProcessResult:
    source_bgr: np.ndarray
    output_bgr: np.ndarray
    frame_index: int
    model_ms: float
    total_ms: float


class TSRNNWebcamProcessor:
    def __init__(
        self, model_options: ModelOptions, traj_options: TrajectoryOptions
    ) -> None:
        self.options = model_options
        self.traj_options = traj_options
        self.device = resolve_device(model_options.device)
        self.channels_last = model_options.channels_last
        self.amp = model_options.amp

        add_repo_to_path(model_options.repo_root)
        self._load_repo_symbols()

        self.cfg = self.load_config(str(model_options.config), [])
        self.model = self._build_tsrnn_model().to(self.device).eval()
        if self.channels_last:
            self.model = self.to_channels_last(self.model)

        self._load_tsrnn_checkpoint(model_options.checkpoint)
        self.apply_speed_patches(self.model, mode=model_options.fast_shift)
        if model_options.compile_step:
            self.maybe_compile_step(self.model, mode="reduce-overhead")

        self.traj_adapter = self._build_trajectory_adapter(traj_options)
        self.traj_size = self._trajectory_input_size(traj_options)

        self.state: dict | None = None
        self.prev_tensor: torch.Tensor | None = None
        self.cur_tensor: torch.Tensor | None = None
        self.cur_frame_bgr = None
        self.frame_index = 0

        self._warmup(model_options.warmup_steps)

    def _load_repo_symbols(self) -> None:
        from tsrn_rtvd.models.shift_ops import (
            apply_speed_patches,
            maybe_compile_step,
            to_channels_last,
        )
        from tsrn_rtvd.models.tsrnn_bt1 import TSRNN_BT1
        from tsrn_rtvd.tiny_delta3d.model import (
            build_model_from_config as build_traj_model_from_config,
        )
        from tsrn_rtvd.tiny_delta3d.model import (
            make_deploy_copy as make_traj_deploy_copy,
        )
        from tsrn_rtvd.tiny_delta3d.preprocess import FixedDerivativePreprocess
        from tsrn_rtvd.utils import load_config
        from tsrn_rtvd.utils.traj3d_projection import (
            LearnedTrajectory3DAdapter,
            TrajProjectionConfig,
            Translation3DToPixelDP,
            extract_state_dict,
            find_mean_std,
            find_output_mean_std,
            load_stats_file,
        )

        self.TSRNN_BT1 = TSRNN_BT1
        self.apply_speed_patches = apply_speed_patches
        self.maybe_compile_step = maybe_compile_step
        self.to_channels_last = to_channels_last
        self.build_traj_model_from_config = build_traj_model_from_config
        self.make_traj_deploy_copy = make_traj_deploy_copy
        self.FixedDerivativePreprocess = FixedDerivativePreprocess
        self.load_config = load_config
        self.LearnedTrajectory3DAdapter = LearnedTrajectory3DAdapter
        self.TrajProjectionConfig = TrajProjectionConfig
        self.Translation3DToPixelDP = Translation3DToPixelDP
        self.extract_state_dict = extract_state_dict
        self.find_mean_std = find_mean_std
        self.find_output_mean_std = find_output_mean_std
        self.load_stats_file = load_stats_file

    def _build_tsrnn_model(self) -> torch.nn.Module:
        model_cfg = getattr(self.cfg, "model", {})
        getter = (
            model_cfg.get
            if hasattr(model_cfg, "get")
            else lambda key, default=None: getattr(model_cfg, key, default)
        )
        return self.TSRNN_BT1(
            update8_blocks=int(getter("update8_blocks", getter("update8_rbs", 10))),
            update4_blocks=int(getter("update4_blocks", getter("update4_rbs", 10))),
        )

    def _load_tsrnn_checkpoint(self, checkpoint: Path) -> None:
        if not checkpoint.exists():
            raise FileNotFoundError(f"TSRNN checkpoint does not exist: {checkpoint}")

        ckpt = torch.load(str(checkpoint), map_location="cpu")
        if isinstance(ckpt, dict) and "model" in ckpt:
            state_dict = ckpt["model"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif isinstance(ckpt, dict) and all(torch.is_tensor(v) for v in ckpt.values()):
            state_dict = ckpt
        else:
            raise TypeError(f"Unsupported TSRNN checkpoint format: {checkpoint}")

        self.model.load_state_dict(state_dict, strict=True)

    def _build_trajectory_adapter(self, options: TrajectoryOptions):
        if not options.enabled:
            return None

        for path in [options.config, options.checkpoint, options.stats]:
            if not path.exists():
                raise FileNotFoundError(f"Trajectory file does not exist: {path}")

        traj_cfg = self.load_config(str(options.config), [])
        traj_model = self.build_traj_model_from_config(traj_cfg).to(self.device).eval()
        if getattr(traj_model, "K", 3) != 3:
            raise ValueError(f"Expected TinyDelta3D K=3, got {traj_model.K}")

        ckpt = torch.load(str(options.checkpoint), map_location=self.device)
        state_dict = self.extract_state_dict(ckpt)
        incompatible = traj_model.load_state_dict(
            state_dict, strict=not options.nonstrict
        )
        if options.nonstrict:
            print(f"[traj] missing keys: {incompatible.missing_keys}")
            print(f"[traj] unexpected keys: {incompatible.unexpected_keys}")

        if options.deploy:
            traj_model = self.make_traj_deploy_copy(traj_model).to(self.device).eval()

        stats = self.load_stats_file(options.stats)
        prep_mean, prep_std = self.find_mean_std(stats)
        out_mean, out_std = self.find_output_mean_std(stats)

        preprocess = self.FixedDerivativePreprocess().to(self.device).eval()
        preprocess.set_stats(prep_mean, prep_std)

        projection = (
            self.Translation3DToPixelDP(
                self.TrajProjectionConfig(
                    fx=options.fx,
                    fy=options.fy,
                    z_ref=options.z_ref,
                    sign_x=options.sign_x,
                    sign_y=options.sign_y,
                    output_scale=options.output_scale,
                    output_order=options.output_order,
                ),
                output_mean=out_mean,
                output_std=out_std,
            )
            .to(self.device)
            .eval()
        )

        return self.LearnedTrajectory3DAdapter(
            model=traj_model,
            preprocess=preprocess,
            projector=projection,
        )

    def _trajectory_input_size(
        self, options: TrajectoryOptions
    ) -> tuple[int, int] | None:
        if not options.enabled or not options.resize_to_config:
            return None
        cfg = self.load_config(str(options.config), [])
        height = int(getattr(cfg.data, "height", 0))
        width = int(getattr(cfg.data, "width", 0))
        if height <= 0 or width <= 0:
            return None
        return height, width

    def _warmup(self, steps: int) -> None:
        if steps <= 0:
            return
        height, width = 96, 160
        dummy = torch.zeros(1, 3, height, width, device=self.device)
        if self.channels_last:
            dummy = dummy.contiguous(memory_format=torch.channels_last)
        state = self.model.init_state(1, height, width, self.device)
        zero_dp = torch.zeros(1, 2, device=self.device)
        with torch.inference_mode():
            for _ in range(steps):
                with autocast_context(self.device, self.amp):
                    if self.traj_adapter is not None:
                        self._predict_trajectory(dummy, dummy, dummy)
                    _, state = self.model.step(dummy, state, zero_dp, zero_dp)
        synchronize(self.device)

    def reset(self, first_frame_bgr) -> None:
        first_tensor = bgr_to_tensor(first_frame_bgr, self.device, self.channels_last)
        _, _, height, width = first_tensor.shape

        with torch.inference_mode():
            state = self.model.init_state(1, height, width, self.device)
            with autocast_context(self.device, self.amp):
                features = tuple(
                    feat.detach() for feat in self.model.encoder(first_tensor)
                )

        state["feat_prev"] = features
        state["feat_cur"] = features
        state["B_cur"] = first_tensor

        self.state = state
        self.prev_tensor = first_tensor
        self.cur_tensor = first_tensor
        self.cur_frame_bgr = first_frame_bgr.copy()
        self.frame_index = 0

    def process_next(self, next_frame_bgr) -> ProcessResult:
        if self.state is None or self.prev_tensor is None or self.cur_tensor is None:
            raise RuntimeError("Call reset() with the first frame before processing.")

        started = time.perf_counter()
        next_tensor = bgr_to_tensor(next_frame_bgr, self.device, self.channels_last)
        if next_tensor.shape[-2:] != self.cur_tensor.shape[-2:]:
            next_frame_bgr = cv2.resize(
                next_frame_bgr,
                (self.cur_tensor.shape[-1], self.cur_tensor.shape[-2]),
                interpolation=cv2.INTER_AREA,
            )
            next_tensor = bgr_to_tensor(next_frame_bgr, self.device, self.channels_last)

        with torch.inference_mode():
            dp_neg, dp_pos = self._predict_motion(
                self.prev_tensor, self.cur_tensor, next_tensor
            )
            if self.frame_index == 0:
                dp_neg = torch.zeros_like(dp_neg)

            synchronize(self.device)
            model_started = time.perf_counter()
            with autocast_context(self.device, self.amp):
                pred, new_state = self.model.step(
                    next_tensor, self.state, dp_neg, dp_pos
                )
            synchronize(self.device)

        model_ms = (time.perf_counter() - model_started) * 1000.0
        total_ms = (time.perf_counter() - started) * 1000.0

        source_bgr = self.cur_frame_bgr.copy()
        output_bgr = tensor_to_bgr(pred)

        self.state = new_state
        self.prev_tensor = self.cur_tensor.detach()
        self.cur_tensor = next_tensor.detach()
        self.cur_frame_bgr = next_frame_bgr.copy()
        self.frame_index += 1

        return ProcessResult(
            source_bgr=source_bgr,
            output_bgr=output_bgr,
            frame_index=self.frame_index,
            model_ms=model_ms,
            total_ms=total_ms,
        )

    def _predict_motion(
        self,
        prev_tensor: torch.Tensor,
        cur_tensor: torch.Tensor,
        next_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.traj_adapter is None:
            zero = torch.zeros(1, 2, device=self.device)
            return zero, zero

        return self._predict_trajectory(prev_tensor, cur_tensor, next_tensor)

    def _predict_trajectory(
        self,
        prev_tensor: torch.Tensor,
        cur_tensor: torch.Tensor,
        next_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.traj_size is not None:
            prev_tensor = F.interpolate(
                prev_tensor, size=self.traj_size, mode="bilinear", align_corners=False
            )
            cur_tensor = F.interpolate(
                cur_tensor, size=self.traj_size, mode="bilinear", align_corners=False
            )
            next_tensor = F.interpolate(
                next_tensor, size=self.traj_size, mode="bilinear", align_corners=False
            )
        return self.traj_adapter.predict_left_right(
            prev_tensor, cur_tensor, next_tensor
        )
