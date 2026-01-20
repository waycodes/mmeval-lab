"""Optional: MMMU submission export for test set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def export_mmmu_submission(
    predictions: list[dict[str, Any]],
    output_path: Path,
    model_name: str = "anonymous",
) -> Path:
    """Export predictions in MMMU submission format.

    Format: JSON with example_id -> answer mapping
    """
    submission = {}
    for p in predictions:
        example_id = p.get("example_id", "")
        answer = p.get("extracted_answer", "")
        # MMMU expects single letter answers
        if answer and answer[0].upper() in "ABCDE":
            submission[example_id] = answer[0].upper()
        else:
            submission[example_id] = answer

    output = {
        "model_name": model_name,
        "predictions": submission,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    return output_path


def validate_submission(submission_path: Path) -> tuple[bool, list[str]]:
    """Validate MMMU submission format."""
    errors = []

    try:
        with open(submission_path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON: {e}"]

    if "predictions" not in data:
        errors.append("Missing 'predictions' key")
        return False, errors

    preds = data["predictions"]
    for eid, answer in preds.items():
        if not isinstance(answer, str):
            errors.append(f"{eid}: answer must be string")
        elif len(answer) != 1 or answer.upper() not in "ABCDE":
            errors.append(f"{eid}: answer must be single letter A-E")

    return len(errors) == 0, errors
