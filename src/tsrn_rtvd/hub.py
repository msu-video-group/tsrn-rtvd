from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Optional

DEFAULT_HF_REPO_ID = os.environ.get(
    "TSRN_RTVD_HF_REPO_ID", "egorchistov/deblurring-tsrn-rtvd"
)

TSRNN_CONFIG_NAME = "tsrnn_bt1_exact_ft.yaml"
TRAJECTORY_CONFIG_NAME = "tinydelta3d_k3_108x192.yaml"

HF_TSRNN_CHECKPOINT = "checkpoints/tsrnn_bt1_exact_ft.pth"
HF_TRAJECTORY_CHECKPOINT = "checkpoints/tinydelta3d_k3_108x192.pt"
HF_TRAJECTORY_STATS = "checkpoints/tinydelta3d_k3_108x192_stats.pt"


@dataclass(frozen=True)
class ArtifactPaths:
    config: Path
    checkpoint: Path
    traj_config: Path | None = None
    traj_checkpoint: Path | None = None
    traj_stats: Path | None = None
    repo_root: Path | None = None


def resolve_artifacts(
    repo_root: str | Path | None = None,
    *,
    hf_repo_id: str | None = None,
    with_trajectory: bool = True,
    local_files_only: bool = False,
) -> ArtifactPaths:
    """Resolve packaged configs and download model artifacts from Hugging Face."""
    repo_id = hf_repo_id or DEFAULT_HF_REPO_ID
    config = package_config_path(TSRNN_CONFIG_NAME)
    checkpoint = _hf_download(repo_id, HF_TSRNN_CHECKPOINT, local_files_only)

    if not with_trajectory:
        return ArtifactPaths(config=config, checkpoint=checkpoint)

    return ArtifactPaths(
        config=config,
        checkpoint=checkpoint,
        traj_config=package_config_path(TRAJECTORY_CONFIG_NAME),
        traj_checkpoint=_hf_download(
            repo_id, HF_TRAJECTORY_CHECKPOINT, local_files_only
        ),
        traj_stats=_hf_download(repo_id, HF_TRAJECTORY_STATS, local_files_only),
    )


def package_config_path(name: str) -> Path:
    ref = resources.files("tsrn_rtvd.configs").joinpath(name)
    with resources.as_file(ref) as path:
        return Path(path)


def _hf_download(repo_id: str, filename: str, local_files_only: bool) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError(
            "huggingface-hub is required to download TSRN-RTVD checkpoints."
        ) from exc

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_files_only=local_files_only,
        )
    )
