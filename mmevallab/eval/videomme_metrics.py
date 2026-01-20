"""Video-MME official evaluation format and metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def export_videomme_predictions(
    predictions: list[dict[str, Any]],
    output_path: Path | str,
) -> None:
    """Export predictions in Video-MME official JSON format.

    Args:
        predictions: List of prediction dicts with video_id, question_id, answer
        output_path: Path to write JSON file
    """
    # Format for official evaluation
    formatted = []
    for pred in predictions:
        formatted.append(
            {
                "video_id": pred.get("video_id", ""),
                "question_id": pred.get("question_id", 0),
                "answer": pred.get("answer", ""),
            }
        )

    with open(output_path, "w") as f:
        json.dump(formatted, f, indent=2)


def load_videomme_ground_truth(gt_path: Path | str) -> dict[str, str]:
    """Load ground truth answers from Video-MME format.

    Args:
        gt_path: Path to ground truth JSON

    Returns:
        Dict mapping "video_id_qN" to answer letter
    """
    with open(gt_path) as f:
        data = json.load(f)

    gt_map = {}
    for item in data:
        video_id = item.get("video_id", "")
        question_id = item.get("question_id", 0)
        answer = item.get("answer", "")
        key = f"{video_id}_q{question_id}"
        gt_map[key] = answer

    return gt_map


def evaluate_videomme(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]] | None = None,
    gt_path: Path | str | None = None,
) -> dict[str, Any]:
    """Evaluate Video-MME predictions.

    Args:
        predictions: List of prediction dicts
        ground_truths: List of ground truth dicts (alternative to gt_path)
        gt_path: Path to ground truth JSON

    Returns:
        Dict with accuracy metrics and breakdowns
    """
    # Load ground truth
    if gt_path:
        gt_map = load_videomme_ground_truth(gt_path)
    elif ground_truths:
        gt_map = {}
        for item in ground_truths:
            video_id = item.get("video_id", "")
            question_id = item.get("question_id", 0)
            answer = item.get("answer", "")
            key = f"{video_id}_q{question_id}"
            gt_map[key] = answer
    else:
        raise ValueError("Must provide ground_truths or gt_path")

    # Compute metrics
    correct = 0
    total = 0
    by_duration: dict[str, dict[str, int]] = {}
    by_domain: dict[str, dict[str, int]] = {}

    for pred in predictions:
        video_id = pred.get("video_id", "")
        question_id = pred.get("question_id", 0)
        pred_answer = pred.get("answer", "").strip().upper()

        key = f"{video_id}_q{question_id}"
        gt_answer = gt_map.get(key, "").strip().upper()

        if not gt_answer:
            continue

        total += 1
        is_correct = pred_answer == gt_answer
        if is_correct:
            correct += 1

        # Track by duration
        duration = pred.get("duration", "unknown")
        if duration not in by_duration:
            by_duration[duration] = {"correct": 0, "total": 0}
        by_duration[duration]["total"] += 1
        if is_correct:
            by_duration[duration]["correct"] += 1

        # Track by domain
        domain = pred.get("domain", "unknown")
        if domain not in by_domain:
            by_domain[domain] = {"correct": 0, "total": 0}
        by_domain[domain]["total"] += 1
        if is_correct:
            by_domain[domain]["correct"] += 1

    # Compute accuracies
    overall_accuracy = correct / total if total > 0 else 0.0

    duration_accuracy = {}
    for dur, counts in by_duration.items():
        if counts["total"] > 0:
            duration_accuracy[dur] = counts["correct"] / counts["total"]

    domain_accuracy = {}
    for dom, counts in by_domain.items():
        if counts["total"] > 0:
            domain_accuracy[dom] = counts["correct"] / counts["total"]

    return {
        "overall_accuracy": overall_accuracy,
        "correct": correct,
        "total": total,
        "by_duration": duration_accuracy,
        "by_domain": domain_accuracy,
    }
