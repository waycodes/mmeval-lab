"""Similarity-based attribution baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Attribution:
    """Attribution of a test example to training data."""

    test_id: str
    train_id: str
    similarity: float
    method: str


def compute_text_similarity(text1: str, text2: str) -> float:
    """Compute simple text similarity (Jaccard on words)."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return intersection / union


def attribute_by_similarity(
    test_examples: list[dict[str, Any]],
    train_examples: list[dict[str, Any]],
    text_field: str = "question",
    top_k: int = 5,
    threshold: float = 0.3,
) -> dict[str, list[Attribution]]:
    """Attribute test examples to training data by text similarity."""
    attributions: dict[str, list[Attribution]] = {}

    for test in test_examples:
        test_id = test.get("example_id", test.get("id", ""))
        test_text = test.get(text_field, "")

        scores = []
        for train in train_examples:
            train_id = train.get("example_id", train.get("id", ""))
            train_text = train.get(text_field, "")
            sim = compute_text_similarity(test_text, train_text)
            if sim >= threshold:
                scores.append((train_id, sim))

        scores.sort(key=lambda x: -x[1])
        attributions[test_id] = [
            Attribution(test_id=test_id, train_id=tid, similarity=sim, method="jaccard")
            for tid, sim in scores[:top_k]
        ]

    return attributions
