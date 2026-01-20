"""Run delta canonicalization for attribution analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ExampleDelta:
    """Delta for a single example between two runs."""

    example_id: str
    run1_correct: bool | None
    run2_correct: bool | None
    changed: bool
    direction: str  # "improved", "regressed", "unchanged"


@dataclass
class SliceDelta:
    """Delta for a slice between two runs."""

    slice_name: str
    run1_accuracy: float
    run2_accuracy: float
    delta: float
    run1_count: int
    run2_count: int


def load_predictions(run_dir: Path | str) -> dict[str, dict[str, Any]]:
    """Load predictions from run directory."""
    path = Path(run_dir) / "predictions.jsonl"
    preds = {}
    with open(path) as f:
        for line in f:
            p = json.loads(line)
            preds[p["example_id"]] = p
    return preds


def compute_example_deltas(
    run1_dir: Path | str,
    run2_dir: Path | str,
) -> list[ExampleDelta]:
    """Compute per-example deltas between two runs."""
    preds1 = load_predictions(run1_dir)
    preds2 = load_predictions(run2_dir)

    all_ids = set(preds1.keys()) | set(preds2.keys())
    deltas = []

    for eid in all_ids:
        p1 = preds1.get(eid, {})
        p2 = preds2.get(eid, {})

        c1 = p1.get("is_correct")
        c2 = p2.get("is_correct")

        if c1 is False and c2 is True:
            direction = "improved"
        elif c1 is True and c2 is False:
            direction = "regressed"
        else:
            direction = "unchanged"

        deltas.append(
            ExampleDelta(
                example_id=eid,
                run1_correct=c1,
                run2_correct=c2,
                changed=c1 != c2,
                direction=direction,
            )
        )

    return deltas


def compute_slice_deltas(
    example_deltas: list[ExampleDelta],
    slice_assignments: dict[str, list[str]],  # slice_name -> example_ids
) -> list[SliceDelta]:
    """Compute per-slice deltas from example deltas."""
    delta_map = {d.example_id: d for d in example_deltas}
    slice_deltas = []

    for slice_name, example_ids in slice_assignments.items():
        r1_correct = 0
        r1_total = 0
        r2_correct = 0
        r2_total = 0

        for eid in example_ids:
            d = delta_map.get(eid)
            if not d:
                continue
            if d.run1_correct is not None:
                r1_total += 1
                if d.run1_correct:
                    r1_correct += 1
            if d.run2_correct is not None:
                r2_total += 1
                if d.run2_correct:
                    r2_correct += 1

        r1_acc = r1_correct / r1_total if r1_total > 0 else 0
        r2_acc = r2_correct / r2_total if r2_total > 0 else 0

        slice_deltas.append(
            SliceDelta(
                slice_name=slice_name,
                run1_accuracy=r1_acc,
                run2_accuracy=r2_acc,
                delta=r2_acc - r1_acc,
                run1_count=r1_total,
                run2_count=r2_total,
            )
        )

    return sorted(slice_deltas, key=lambda x: x.delta)


def canonicalize_run_delta(
    run1_dir: Path | str,
    run2_dir: Path | str,
) -> dict[str, Any]:
    """Create canonical delta object for attribution."""
    example_deltas = compute_example_deltas(run1_dir, run2_dir)

    improved = [d for d in example_deltas if d.direction == "improved"]
    regressed = [d for d in example_deltas if d.direction == "regressed"]
    unchanged = [d for d in example_deltas if d.direction == "unchanged"]

    return {
        "run1": str(run1_dir),
        "run2": str(run2_dir),
        "total_examples": len(example_deltas),
        "improved": len(improved),
        "regressed": len(regressed),
        "unchanged": len(unchanged),
        "net_change": len(improved) - len(regressed),
        "improved_ids": [d.example_id for d in improved],
        "regressed_ids": [d.example_id for d in regressed],
    }
