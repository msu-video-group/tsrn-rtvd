from __future__ import annotations

from pathlib import Path

from tsrn_rtvd import hub


def test_resolve_artifacts_downloads_from_hf(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, str, bool]] = []

    def fake_download(repo_id: str, filename: str, local_files_only: bool) -> Path:
        calls.append((repo_id, filename, local_files_only))
        return tmp_path / filename.replace("/", "_")

    monkeypatch.setattr(hub, "_hf_download", fake_download)

    artifacts = hub.resolve_artifacts(hf_repo_id="org/model", local_files_only=True)

    assert artifacts.config.name == hub.TSRNN_CONFIG_NAME
    assert artifacts.checkpoint == tmp_path / "checkpoints_tsrnn_bt1_exact_ft.pth"
    assert artifacts.traj_config is not None
    assert artifacts.traj_config.name == hub.TRAJECTORY_CONFIG_NAME
    assert (
        artifacts.traj_checkpoint == tmp_path / "checkpoints_tinydelta3d_k3_108x192.pt"
    )
    assert (
        artifacts.traj_stats == tmp_path / "checkpoints_tinydelta3d_k3_108x192_stats.pt"
    )
    assert artifacts.repo_root is None
    assert calls == [
        ("org/model", hub.HF_TSRNN_CHECKPOINT, True),
        ("org/model", hub.HF_TRAJECTORY_CHECKPOINT, True),
        ("org/model", hub.HF_TRAJECTORY_STATS, True),
    ]


def test_resolve_artifacts_without_trajectory_downloads_only_tsrnn(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, str, bool]] = []

    def fake_download(repo_id: str, filename: str, local_files_only: bool) -> Path:
        calls.append((repo_id, filename, local_files_only))
        return tmp_path / filename.replace("/", "_")

    monkeypatch.setattr(hub, "_hf_download", fake_download)

    artifacts = hub.resolve_artifacts(with_trajectory=False, local_files_only=True)

    assert artifacts.traj_config is None
    assert artifacts.traj_checkpoint is None
    assert artifacts.traj_stats is None
    assert calls == [(hub.DEFAULT_HF_REPO_ID, hub.HF_TSRNN_CHECKPOINT, True)]
