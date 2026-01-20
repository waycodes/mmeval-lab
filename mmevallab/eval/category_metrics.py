"""Category breakdown metrics computation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def compute_category_breakdown(
    predictions: list[dict[str, Any]],
    category_field: str,
) -> dict[str, dict[str, Any]]:
    """Compute metrics breakdown by category."""
    by_cat: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})

    for p in predictions:
        meta = p.get("metadata", {})
        cat = meta.get(category_field, "unknown")
        by_cat[cat]["total"] += 1
        if p.get("is_correct"):
            by_cat[cat]["correct"] += 1

    result = {}
    for cat, data in by_cat.items():
        acc = data["correct"] / data["total"] if data["total"] > 0 else 0.0
        result[cat] = {"accuracy": acc, **data}
    return result


def compute_multi_category_breakdown(
    predictions: list[dict[str, Any]],
    category_fields: list[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Compute metrics breakdown by multiple category fields."""
    return {field: compute_category_breakdown(predictions, field) for field in category_fields}
