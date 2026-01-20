"""MMMU scoring and group breakdowns."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def compute_mmmu_metrics(
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute MMMU metrics with group breakdowns."""
    # Overall
    correct = sum(1 for p in predictions if p.get("is_correct"))
    total = len(predictions)

    # By discipline
    by_discipline: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    # By subject
    by_subject: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})

    for p in predictions:
        meta = p.get("metadata", {})
        disc = meta.get("discipline", "unknown")
        subj = meta.get("subject", "unknown")

        by_discipline[disc]["total"] += 1
        by_subject[subj]["total"] += 1
        if p.get("is_correct"):
            by_discipline[disc]["correct"] += 1
            by_subject[subj]["correct"] += 1

    def acc(d: dict[str, int]) -> float:
        return d["correct"] / d["total"] if d["total"] > 0 else 0.0

    return {
        "overall_accuracy": correct / total if total > 0 else 0.0,
        "correct": correct,
        "total": total,
        "by_discipline": {k: {"accuracy": acc(v), **v} for k, v in by_discipline.items()},
        "by_subject": {k: {"accuracy": acc(v), **v} for k, v in by_subject.items()},
    }
