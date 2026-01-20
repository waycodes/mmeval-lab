"""Correlate data diffs with failure-mode deltas."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from mmevallab.eval.failure_modes import FailureLabel, FailureMode


@dataclass
class FailureModeDelta:
    """Change in failure mode distribution."""

    mode: FailureMode
    baseline_count: int
    current_count: int
    delta: int
    delta_pct: float


def compute_failure_mode_deltas(
    baseline_labels: dict[str, FailureLabel],
    current_labels: dict[str, FailureLabel],
) -> list[FailureModeDelta]:
    """Compute changes in failure mode distribution."""
    baseline_counts: dict[FailureMode, int] = defaultdict(int)
    current_counts: dict[FailureMode, int] = defaultdict(int)

    for label in baseline_labels.values():
        baseline_counts[label.mode] += 1
    for label in current_labels.values():
        current_counts[label.mode] += 1

    deltas = []
    all_modes = set(baseline_counts.keys()) | set(current_counts.keys())

    for mode in all_modes:
        base = baseline_counts[mode]
        curr = current_counts[mode]
        delta = curr - base
        delta_pct = delta / base if base > 0 else float("inf") if delta > 0 else 0.0

        deltas.append(
            FailureModeDelta(
                mode=mode,
                baseline_count=base,
                current_count=curr,
                delta=delta,
                delta_pct=delta_pct,
            )
        )

    return sorted(deltas, key=lambda x: abs(x.delta), reverse=True)


def correlate_with_diff(
    failure_deltas: list[FailureModeDelta],
    added_ids: set[str],
    removed_ids: set[str],
    current_labels: dict[str, FailureLabel],
) -> dict[str, Any]:
    """Correlate failure mode changes with dataset diff."""
    # Count failure modes in added examples
    added_modes: dict[FailureMode, int] = defaultdict(int)
    for eid in added_ids:
        if eid in current_labels:
            added_modes[current_labels[eid].mode] += 1

    return {
        "failure_deltas": [
            {
                "mode": d.mode.value,
                "baseline": d.baseline_count,
                "current": d.current_count,
                "delta": d.delta,
            }
            for d in failure_deltas
        ],
        "added_by_mode": {m.value: c for m, c in added_modes.items()},
        "total_added": len(added_ids),
        "total_removed": len(removed_ids),
    }
