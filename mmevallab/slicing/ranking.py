"""Slice ranking for worst performers and regressions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SliceRank:
    """Ranked slice with metrics."""

    name: str
    metric: float
    count: int
    rank: int


def rank_worst_slices(
    slice_metrics: dict[str, dict[str, Any]],
    metric_key: str = "accuracy",
    top_k: int = 10,
) -> list[SliceRank]:
    """Rank slices by worst performance."""
    items = [
        (name, data.get(metric_key, 0.0), data.get("count", data.get("total", 0)))
        for name, data in slice_metrics.items()
    ]
    items.sort(key=lambda x: x[1])

    return [
        SliceRank(name=name, metric=metric, count=count, rank=i + 1)
        for i, (name, metric, count) in enumerate(items[:top_k])
    ]


def rank_regressions(
    baseline_metrics: dict[str, dict[str, Any]],
    current_metrics: dict[str, dict[str, Any]],
    metric_key: str = "accuracy",
    top_k: int = 10,
) -> list[tuple[str, float, float, float]]:
    """Rank slices by regression magnitude.

    Returns list of (slice_name, baseline, current, delta).
    """
    regressions = []
    for name in baseline_metrics:
        if name not in current_metrics:
            continue
        baseline = baseline_metrics[name].get(metric_key, 0.0)
        current = current_metrics[name].get(metric_key, 0.0)
        delta = current - baseline
        if delta < 0:
            regressions.append((name, baseline, current, delta))

    regressions.sort(key=lambda x: x[3])
    return regressions[:top_k]
