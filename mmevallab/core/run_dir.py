"""Run directory conventions and utilities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class RunDir:
    """Run directory structure."""

    path: Path

    @property
    def predictions_path(self) -> Path:
        return self.path / "predictions.jsonl"

    @property
    def metrics_path(self) -> Path:
        return self.path / "metrics.json"

    @property
    def config_path(self) -> Path:
        return self.path / "config.yaml"

    @property
    def metadata_path(self) -> Path:
        return self.path / "metadata.json"

    @property
    def logs_path(self) -> Path:
        return self.path / "logs"

    def ensure_dirs(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        self.logs_path.mkdir(exist_ok=True)


def generate_run_id(config: dict[str, Any]) -> str:
    """Generate deterministic run ID from config."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def create_run_dir(base: Path, config: dict[str, Any]) -> RunDir:
    """Create run directory with timestamp and config hash."""
    run_id = generate_run_id(config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_path = base / f"run_{timestamp}_{run_id}"
    run_dir = RunDir(run_path)
    run_dir.ensure_dirs()
    return run_dir
