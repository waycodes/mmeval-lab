"""Example-level diff view for run comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ExampleDiff:
    """Diff for a single example between two runs."""

    example_id: str
    run_a_correct: bool | None
    run_b_correct: bool | None
    run_a_answer: str
    run_b_answer: str
    ground_truth: str
    change_type: str  # improved, regressed, unchanged, both_wrong, both_correct


def compute_example_diffs(
    run_a: list[dict[str, Any]],
    run_b: list[dict[str, Any]],
) -> list[ExampleDiff]:
    """Compute example-level diffs between two runs."""
    a_by_id = {p["example_id"]: p for p in run_a}
    b_by_id = {p["example_id"]: p for p in run_b}

    all_ids = set(a_by_id.keys()) | set(b_by_id.keys())
    diffs = []

    for eid in sorted(all_ids):
        pa = a_by_id.get(eid, {})
        pb = b_by_id.get(eid, {})

        a_correct = pa.get("is_correct")
        b_correct = pb.get("is_correct")

        if a_correct and b_correct:
            change = "both_correct"
        elif not a_correct and not b_correct:
            change = "both_wrong"
        elif not a_correct and b_correct:
            change = "improved"
        elif a_correct and not b_correct:
            change = "regressed"
        else:
            change = "unchanged"

        diffs.append(
            ExampleDiff(
                example_id=eid,
                run_a_correct=a_correct,
                run_b_correct=b_correct,
                run_a_answer=pa.get("extracted_answer", ""),
                run_b_answer=pb.get("extracted_answer", ""),
                ground_truth=pa.get("ground_truth", pb.get("ground_truth", "")),
                change_type=change,
            )
        )

    return diffs


def summarize_diffs(diffs: list[ExampleDiff]) -> dict[str, int]:
    """Summarize example diffs by change type."""
    summary: dict[str, int] = {}
    for d in diffs:
        summary[d.change_type] = summary.get(d.change_type, 0) + 1
    return summary
