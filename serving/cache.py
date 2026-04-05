"""Local filesystem layout for downloaded model artifacts."""

from __future__ import annotations

import shutil
from pathlib import Path


def version_dir(cache_root: Path, model_name: str, version: str) -> Path:
    safe = model_name.replace("/", "_")
    return cache_root / safe / f"v{version}"


def prepare_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def prune_old_versions(cache_root: Path, model_name: str, keep_versions: int = 3) -> None:
    """Keep only the newest `keep_versions` version directories for a model."""
    safe = model_name.replace("/", "_")
    base = cache_root / safe
    if not base.is_dir():
        return
    dirs = sorted(
        [p for p in base.iterdir() if p.is_dir() and p.name.startswith("v")],
        key=lambda p: p.name,
        reverse=True,
    )
    for stale in dirs[keep_versions:]:
        shutil.rmtree(stale, ignore_errors=True)
