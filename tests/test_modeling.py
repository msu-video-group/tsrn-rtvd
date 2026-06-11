from __future__ import annotations

import types
from pathlib import Path

import numpy as np

import tsrn_rtvd.modeling as modeling
from tsrn_rtvd.hub import ArtifactPaths
from tsrn_rtvd.modeling import TSRNRTVD


class FakeProcessor:
    def __init__(self, model_options, trajectory_options) -> None:
        self.model_options = model_options
        self.trajectory_options = trajectory_options
        self.device = model_options.device
        self.reset_frame = None

    def reset(self, first_frame_bgr):
        self.reset_frame = first_frame_bgr

    def process_next(self, next_frame_bgr):
        return types.SimpleNamespace(output_bgr=next_frame_bgr)


def test_from_pretrained_builds_processor(monkeypatch, tmp_path: Path) -> None:
    artifacts = ArtifactPaths(
        config=tmp_path / "config.yaml",
        checkpoint=tmp_path / "model.pth",
        traj_config=tmp_path / "traj.yaml",
        traj_checkpoint=tmp_path / "traj.pt",
        traj_stats=tmp_path / "stats.pt",
        repo_root=tmp_path,
    )

    monkeypatch.setattr(
        modeling, "resolve_artifacts", lambda *args, **kwargs: artifacts
    )

    fake_pipeline = types.ModuleType("tsrn_rtvd.demo.pipeline")
    fake_pipeline.ModelOptions = types.SimpleNamespace
    fake_pipeline.TrajectoryOptions = types.SimpleNamespace
    fake_pipeline.TSRNNWebcamProcessor = FakeProcessor
    monkeypatch.setitem(
        __import__("sys").modules, "tsrn_rtvd.demo.pipeline", fake_pipeline
    )

    wrapper = TSRNRTVD.from_pretrained("org/model", device="cpu", warmup_steps=0)

    assert wrapper.artifacts == artifacts
    assert wrapper.device == "cpu"
    assert wrapper.processor.model_options.checkpoint == artifacts.checkpoint
    assert wrapper.processor.trajectory_options.checkpoint == artifacts.traj_checkpoint


def test_wrapper_delegates_streaming_calls(tmp_path: Path) -> None:
    processor = FakeProcessor(
        types.SimpleNamespace(device="cpu"),
        types.SimpleNamespace(),
    )
    wrapper = TSRNRTVD(processor=processor, artifacts=ArtifactPaths(tmp_path, tmp_path))

    frame = np.zeros((4, 5, 3), dtype=np.uint8)
    wrapper.reset(frame)
    result = wrapper.process_next(frame)

    assert processor.reset_frame is frame
    assert result.output_bgr is frame
