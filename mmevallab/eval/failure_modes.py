"""Failure-mode taxonomy and rule-based labels."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


class FailureMode(str, Enum):
    """Failure mode taxonomy."""

    CORRECT = "correct"
    WRONG_OPTION = "wrong_option"
    REFUSAL = "refusal"
    HALLUCINATION = "hallucination"
    FORMAT_ERROR = "format_error"
    EMPTY = "empty"
    UNKNOWN = "unknown"


@dataclass
class FailureLabel:
    """Failure label for an example."""

    example_id: str
    mode: FailureMode
    confidence: float
    reason: str


def classify_failure(
    prediction: str,
    ground_truth: str,
    is_correct: bool,
    options: list[str] | None = None,
) -> FailureLabel:
    """Classify failure mode using rule-based heuristics."""
    pred = prediction.strip()

    if is_correct:
        return FailureLabel("", FailureMode.CORRECT, 1.0, "Correct answer")

    if not pred:
        return FailureLabel("", FailureMode.EMPTY, 1.0, "Empty response")

    # Check for refusal patterns
    refusal_patterns = [
        r"i cannot",
        r"i can't",
        r"i'm unable",
        r"i am unable",
        r"sorry",
        r"apologize",
    ]
    if any(re.search(p, pred.lower()) for p in refusal_patterns):
        return FailureLabel("", FailureMode.REFUSAL, 0.9, "Refusal detected")

    # Check for format errors (no valid option letter)
    if options:
        valid_letters = {chr(65 + i) for i in range(len(options))}
        first_char = pred[0].upper() if pred else ""
        if first_char not in valid_letters:
            return FailureLabel("", FailureMode.FORMAT_ERROR, 0.8, "No valid option letter")

    # Default to wrong option
    return FailureLabel("", FailureMode.WRONG_OPTION, 0.7, "Incorrect option selected")


def label_failures(
    predictions: list[dict[str, Any]],
) -> dict[str, FailureLabel]:
    """Label failure modes for all predictions."""
    labels = {}
    for p in predictions:
        label = classify_failure(
            prediction=p.get("extracted_answer", ""),
            ground_truth=p.get("ground_truth", ""),
            is_correct=p.get("is_correct", False),
            options=p.get("metadata", {}).get("options"),
        )
        label.example_id = p.get("example_id", "")
        labels[label.example_id] = label
    return labels
