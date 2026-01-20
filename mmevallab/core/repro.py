"""Reproducibility metadata snapshot."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ReproMetadata:
    """Reproducibility metadata for a run."""

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    python_version: str = field(default_factory=lambda: sys.version)
    platform: str = field(default_factory=platform.platform)
    git_commit: str | None = None
    git_dirty: bool = False
    env_vars: dict[str, str] = field(default_factory=dict)
    package_versions: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def capture_repro_metadata(env_prefix: str = "MMEVAL_") -> ReproMetadata:
    """Capture current reproducibility metadata."""
    meta = ReproMetadata()

    # Git info
    try:
        meta.git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        )
        meta.git_dirty = bool(status.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Relevant env vars
    meta.env_vars = {k: v for k, v in os.environ.items() if k.startswith(env_prefix)}

    # Key package versions
    for pkg in ["mmevallab", "torch", "transformers", "datasets", "numpy"]:
        try:
            import importlib.metadata

            meta.package_versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            pass

    return meta
