"""OmniDocBench slice breakdown computation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def compute_omnidoc_slices(
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute OmniDocBench metrics by slice."""
    slices: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"scores": []})

    for p in predictions:
        meta = p.get("metadata", {})
        score = p.get("score", 0.0)

        # By doc type
        doc_type = meta.get("doc_type", "unknown")
        slices[f"doc_type:{doc_type}"]["scores"].append(score)

        # By language
        lang = meta.get("language", "unknown")
        slices[f"language:{lang}"]["scores"].append(score)

        # By layout
        layout = meta.get("layout_type", "unknown")
        slices[f"layout:{layout}"]["scores"].append(score)

        # By content type
        if meta.get("has_formula"):
            slices["content:formula"]["scores"].append(score)
        if meta.get("has_table"):
            slices["content:table"]["scores"].append(score)

    result = {}
    for name, data in slices.items():
        scores = data["scores"]
        result[name] = {
            "mean": sum(scores) / len(scores) if scores else 0.0,
            "count": len(scores),
        }
    return result
