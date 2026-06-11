from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .hub import DEFAULT_HF_REPO_ID, ArtifactPaths, resolve_artifacts


@dataclass
class TSRNRTVD:
    """Streaming TSRN-RTVD inference wrapper."""

    processor: object
    artifacts: ArtifactPaths

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = DEFAULT_HF_REPO_ID,
        *,
        repo_root: str | Path | None = None,
        device: str = "auto",
        trajectory: bool = True,
        local_files_only: bool = False,
        fast_shift: str = "exact",
        channels_last: bool = False,
        compile_step: bool = False,
        amp: bool = False,
        warmup_steps: int = 3,
        traj_deploy: bool = False,
        traj_nonstrict: bool = False,
        traj_fx: float = 1000.0,
        traj_fy: float = 1000.0,
        traj_z_ref: float = 10.0,
        traj_sign_x: float = 1.0,
        traj_sign_y: float = 1.0,
        traj_output_order: str = "xyz",
        traj_output_scale: float = 1.0,
    ) -> TSRNRTVD:
        artifacts = resolve_artifacts(
            repo_root,
            hf_repo_id=repo_id,
            with_trajectory=trajectory,
            local_files_only=local_files_only,
        )

        from .demo.pipeline import ModelOptions, TrajectoryOptions, TSRNNWebcamProcessor

        model_options = ModelOptions(
            repo_root=artifacts.repo_root or Path.cwd(),
            config=artifacts.config,
            checkpoint=artifacts.checkpoint,
            device=device,
            fast_shift=fast_shift,
            channels_last=channels_last,
            compile_step=compile_step,
            amp=amp,
            warmup_steps=warmup_steps,
        )
        traj_options = TrajectoryOptions(
            enabled=trajectory,
            config=artifacts.traj_config or Path(),
            checkpoint=artifacts.traj_checkpoint or Path(),
            stats=artifacts.traj_stats or Path(),
            deploy=traj_deploy,
            nonstrict=traj_nonstrict,
            fx=traj_fx,
            fy=traj_fy,
            z_ref=traj_z_ref,
            sign_x=traj_sign_x,
            sign_y=traj_sign_y,
            output_order=traj_output_order,
            output_scale=traj_output_scale,
        )
        return cls(
            processor=TSRNNWebcamProcessor(model_options, traj_options),
            artifacts=artifacts,
        )

    @property
    def device(self):
        return self.processor.device

    def reset(self, first_frame_bgr: np.ndarray) -> None:
        self.processor.reset(first_frame_bgr)

    def process_next(self, next_frame_bgr: np.ndarray):
        return self.processor.process_next(next_frame_bgr)
