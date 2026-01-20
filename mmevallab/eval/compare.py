"""Run comparison for regression analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_run_metrics(run_dir: Path | str) -> dict[str, Any]:
    """Load metrics from a run directory."""
    run_dir = Path(run_dir)
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path) as f:
        return json.load(f)


def load_run_predictions(run_dir: Path | str) -> list[dict[str, Any]]:
    """Load predictions from a run directory."""
    run_dir = Path(run_dir)
    predictions_path = run_dir / "predictions.jsonl"
    predictions = []
    with open(predictions_path) as f:
        for line in f:
            predictions.append(json.loads(line))
    return predictions


def compare_runs(
    run1_dir: Path | str,
    run2_dir: Path | str,
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    """Compare two evaluation runs.

    Args:
        run1_dir: Path to first run (baseline)
        run2_dir: Path to second run (comparison)
        output_path: Optional path to write comparison results

    Returns:
        Dict with comparison metrics and example-level diffs
    """
    run1_dir = Path(run1_dir)
    run2_dir = Path(run2_dir)

    # Load metrics
    metrics1 = load_run_metrics(run1_dir)
    metrics2 = load_run_metrics(run2_dir)

    # Load predictions
    preds1 = load_run_predictions(run1_dir)
    preds2 = load_run_predictions(run2_dir)

    # Build prediction maps
    pred_map1 = {p["example_id"]: p for p in preds1}
    pred_map2 = {p["example_id"]: p for p in preds2}

    # Compute overall delta
    acc1 = metrics1.get("overall_accuracy", 0)
    acc2 = metrics2.get("overall_accuracy", 0)
    delta = acc2 - acc1

    # Find example-level changes
    correct_to_incorrect = []
    incorrect_to_correct = []
    both_correct = 0
    both_incorrect = 0

    common_ids = set(pred_map1.keys()) & set(pred_map2.keys())
    for example_id in common_ids:
        p1 = pred_map1[example_id]
        p2 = pred_map2[example_id]

        c1 = p1.get("is_correct")
        c2 = p2.get("is_correct")

        if c1 is True and c2 is False:
            correct_to_incorrect.append(
                {
                    "example_id": example_id,
                    "run1_answer": p1.get("extracted_answer"),
                    "run2_answer": p2.get("extracted_answer"),
                    "ground_truth": p1.get("ground_truth"),
                }
            )
        elif c1 is False and c2 is True:
            incorrect_to_correct.append(
                {
                    "example_id": example_id,
                    "run1_answer": p1.get("extracted_answer"),
                    "run2_answer": p2.get("extracted_answer"),
                    "ground_truth": p1.get("ground_truth"),
                }
            )
        elif c1 is True and c2 is True:
            both_correct += 1
        elif c1 is False and c2 is False:
            both_incorrect += 1

    comparison = {
        "run1": str(run1_dir),
        "run2": str(run2_dir),
        "metrics": {
            "run1_accuracy": acc1,
            "run2_accuracy": acc2,
            "delta": delta,
            "delta_pct": f"{delta:+.2%}",
        },
        "example_changes": {
            "correct_to_incorrect": len(correct_to_incorrect),
            "incorrect_to_correct": len(incorrect_to_correct),
            "both_correct": both_correct,
            "both_incorrect": both_incorrect,
            "net_change": len(incorrect_to_correct) - len(correct_to_incorrect),
        },
        "regressions": correct_to_incorrect[:10],  # Top 10
        "improvements": incorrect_to_correct[:10],  # Top 10
    }

    # Write output if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(comparison, f, indent=2)

    return comparison


def format_comparison_report(comparison: dict[str, Any]) -> str:
    """Format comparison results as markdown report."""
    lines = [
        "# Run Comparison Report",
        "",
        f"**Run 1 (baseline):** {comparison['run1']}",
        f"**Run 2 (comparison):** {comparison['run2']}",
        "",
        "## Overall Metrics",
        "",
        "| Metric | Run 1 | Run 2 | Delta |",
        "|--------|-------|-------|-------|",
        f"| Accuracy | {comparison['metrics']['run1_accuracy']:.2%} | "
        f"{comparison['metrics']['run2_accuracy']:.2%} | "
        f"{comparison['metrics']['delta_pct']} |",
        "",
        "## Example-Level Changes",
        "",
        f"- Correct → Incorrect (regressions): "
        f"{comparison['example_changes']['correct_to_incorrect']}",
        f"- Incorrect → Correct (improvements): "
        f"{comparison['example_changes']['incorrect_to_correct']}",
        f"- Both correct: {comparison['example_changes']['both_correct']}",
        f"- Both incorrect: {comparison['example_changes']['both_incorrect']}",
        f"- **Net change:** {comparison['example_changes']['net_change']:+d}",
        "",
    ]

    if comparison.get("regressions"):
        lines.extend(
            [
                "## Top Regressions",
                "",
                "| Example | Run 1 | Run 2 | Ground Truth |",
                "|---------|-------|-------|--------------|",
            ]
        )
        for r in comparison["regressions"]:
            eid, r1, r2, gt = r["example_id"], r["run1_answer"], r["run2_answer"], r["ground_truth"]
            lines.append(f"| {eid} | {r1} | {r2} | {gt} |")
        lines.append("")

    if comparison.get("improvements"):
        lines.extend(
            [
                "## Top Improvements",
                "",
                "| Example | Run 1 | Run 2 | Ground Truth |",
                "|---------|-------|-------|--------------|",
            ]
        )
        for r in comparison["improvements"]:
            eid, r1, r2, gt = r["example_id"], r["run1_answer"], r["run2_answer"], r["ground_truth"]
            lines.append(f"| {eid} | {r1} | {r2} | {gt} |")
        lines.append("")

    return "\n".join(lines)
