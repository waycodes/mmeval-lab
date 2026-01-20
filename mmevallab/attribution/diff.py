"""Dataset diff engine for comparing training manifests."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from mmevallab.contamination.manifest import load_manifest


def compute_dataset_diff(
    old_manifest: Path | str,
    new_manifest: Path | str,
) -> dict[str, Any]:
    """Compare two training manifests and compute diff.

    Returns:
        Dict with added, removed, and modified samples
    """
    old_samples = {s.sample_id: s for s in load_manifest(old_manifest)}
    new_samples = {s.sample_id: s for s in load_manifest(new_manifest)}

    old_ids = set(old_samples.keys())
    new_ids = set(new_samples.keys())

    added_ids = new_ids - old_ids
    removed_ids = old_ids - new_ids
    common_ids = old_ids & new_ids

    # Check for modifications in common samples
    modified_ids = []
    for sid in common_ids:
        old_text = old_samples[sid].text or ""
        new_text = new_samples[sid].text or ""
        if old_text != new_text:
            modified_ids.append(sid)

    # Group by modality/source
    added_by_modality: Counter[str] = Counter()
    added_by_source: Counter[str] = Counter()
    for sid in added_ids:
        s = new_samples[sid]
        added_by_modality[s.modality] += 1
        added_by_source[s.source or "unknown"] += 1

    removed_by_modality: Counter[str] = Counter()
    for sid in removed_ids:
        s = old_samples[sid]
        removed_by_modality[s.modality] += 1

    return {
        "old_count": len(old_samples),
        "new_count": len(new_samples),
        "added": len(added_ids),
        "removed": len(removed_ids),
        "modified": len(modified_ids),
        "added_ids": list(added_ids)[:1000],
        "removed_ids": list(removed_ids)[:1000],
        "modified_ids": modified_ids[:1000],
        "added_by_modality": dict(added_by_modality),
        "added_by_source": dict(added_by_source),
        "removed_by_modality": dict(removed_by_modality),
    }


def save_diff_report(diff: dict[str, Any], output_path: Path | str) -> None:
    """Save diff report to JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(diff, f, indent=2)
