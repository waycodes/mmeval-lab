"""Artifact pack export with license-safe redaction."""

from __future__ import annotations

import json
import shutil
import tarfile
from pathlib import Path
from typing import Any
from zipfile import ZipFile

# Files to always include
INCLUDE_FILES = ["config.yaml", "metadata.json", "metrics.json"]

# Files to redact (remove raw data)
REDACT_FILES = ["predictions.jsonl"]

# Patterns to exclude entirely
EXCLUDE_PATTERNS = ["*.pdf", "*.mp4", "*.avi", "*.mov", "*.png", "*.jpg", "*.jpeg"]


def redact_predictions(predictions_path: Path) -> list[dict[str, Any]]:
    """Redact sensitive fields from predictions."""
    redacted = []
    with open(predictions_path) as f:
        for line in f:
            pred = json.loads(line)
            # Keep only non-sensitive fields
            redacted.append({
                "example_id": pred.get("example_id"),
                "is_correct": pred.get("is_correct"),
                "extracted_answer": pred.get("extracted_answer"),
                "latency_ms": pred.get("latency_ms"),
                # Redact raw_output if too long (may contain copyrighted content)
                "raw_output_length": len(pred.get("raw_output", "")),
            })
    return redacted


def export_artifact_pack(
    run_dir: Path | str,
    output_path: Path | str,
    format: str = "tar.gz",
    include_predictions: bool = True,
) -> Path:
    """Export run artifacts as license-safe archive.

    Args:
        run_dir: Path to run directory
        output_path: Output archive path
        format: Archive format (tar.gz or zip)
        include_predictions: Whether to include redacted predictions

    Returns:
        Path to created archive
    """
    run_dir = Path(run_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create temp directory for staging
    staging_dir = output_path.parent / f".staging_{output_path.stem}"
    staging_dir.mkdir(exist_ok=True)

    try:
        # Copy safe files
        for filename in INCLUDE_FILES:
            src = run_dir / filename
            if src.exists():
                shutil.copy(src, staging_dir / filename)

        # Redact and include predictions
        if include_predictions:
            predictions_path = run_dir / "predictions.jsonl"
            if predictions_path.exists():
                redacted = redact_predictions(predictions_path)
                with open(staging_dir / "predictions_redacted.jsonl", "w") as f:
                    for pred in redacted:
                        f.write(json.dumps(pred) + "\n")

        # Create archive
        if format == "tar.gz":
            with tarfile.open(output_path, "w:gz") as tar:
                for file in staging_dir.iterdir():
                    tar.add(file, arcname=file.name)
        elif format == "zip":
            with ZipFile(output_path, "w") as zf:
                for file in staging_dir.iterdir():
                    zf.write(file, arcname=file.name)
        else:
            raise ValueError(f"Unsupported format: {format}")

    finally:
        # Cleanup staging
        shutil.rmtree(staging_dir, ignore_errors=True)

    return output_path
