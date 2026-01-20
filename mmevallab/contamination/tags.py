"""Contamination slice tags for examples."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ContaminationLevel(str, Enum):
    """Contamination risk level."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXACT = "exact"


@dataclass
class ContaminationTag:
    """Contamination tag for an example."""

    example_id: str
    level: ContaminationLevel
    match_type: str  # exact_text, near_dup, image_match
    match_score: float
    matched_source: str | None = None


def tag_contamination(
    example_id: str,
    text_match_score: float | None = None,
    image_match_score: float | None = None,
    matched_source: str | None = None,
) -> ContaminationTag:
    """Generate contamination tag for an example."""
    # Determine level based on scores
    max_score = max(text_match_score or 0, image_match_score or 0)

    if max_score >= 1.0:
        level = ContaminationLevel.EXACT
        match_type = "exact_text" if text_match_score == 1.0 else "exact_image"
    elif max_score >= 0.9:
        level = ContaminationLevel.HIGH
        match_type = "near_dup"
    elif max_score >= 0.7:
        level = ContaminationLevel.MEDIUM
        match_type = "near_dup"
    elif max_score >= 0.5:
        level = ContaminationLevel.LOW
        match_type = "partial"
    else:
        level = ContaminationLevel.NONE
        match_type = "none"

    return ContaminationTag(
        example_id=example_id,
        level=level,
        match_type=match_type,
        match_score=max_score,
        matched_source=matched_source,
    )


def filter_by_contamination(
    predictions: list[dict[str, Any]],
    tags: dict[str, ContaminationTag],
    max_level: ContaminationLevel = ContaminationLevel.LOW,
) -> list[dict[str, Any]]:
    """Filter predictions to exclude contaminated examples."""
    levels = list(ContaminationLevel)
    max_idx = levels.index(max_level)

    return [
        p
        for p in predictions
        if p["example_id"] not in tags
        or levels.index(tags[p["example_id"]].level) <= max_idx
    ]
