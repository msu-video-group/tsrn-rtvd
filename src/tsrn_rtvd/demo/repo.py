from __future__ import annotations

from pathlib import Path


def default_repo_root() -> Path:
    start = Path(__file__).resolve()
    for candidate in (start.parent, *start.parents):
        package_root = candidate / "src" / "tsrn_rtvd"
        if (package_root / "configs" / "tsrnn_bt1_exact_ft.yaml").exists() and (
            package_root / "models"
        ).is_dir():
            return candidate
    return Path.cwd()


def add_repo_to_path(repo_root: Path) -> None:
    return None


def default_path(repo_root: Path, *parts: str) -> Path:
    return repo_root.joinpath(*parts)
