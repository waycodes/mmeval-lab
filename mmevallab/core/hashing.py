"""Deterministic run-id hashing for reproducibility."""

from __future__ import annotations

import hashlib
import json
import subprocess
from typing import Any


def _canonicalize(obj: Any) -> str:
    """Canonicalize object to deterministic JSON string."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def get_code_version() -> str:
    """Get current git commit hash, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()[:12]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def compute_run_id(
    config: dict[str, Any],
    dataset_id: str,
    code_version: str | None = None,
) -> str:
    """Compute deterministic run ID from config, dataset, and code version.

    Args:
        config: Run configuration dict (will be canonicalized)
        dataset_id: Dataset identifier (name + version + split + content_hash)
        code_version: Git commit hash (auto-detected if None)

    Returns:
        12-character hex hash as run ID
    """
    if code_version is None:
        code_version = get_code_version()

    payload = {
        "config": config,
        "dataset_id": dataset_id,
        "code_version": code_version,
    }

    canonical = _canonicalize(payload)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def compute_dataset_id(
    name: str,
    version: str,
    split: str,
    content_hash: str,
) -> str:
    """Compute dataset identifier string."""
    return f"{name}:{version}:{split}:{content_hash[:8]}"
