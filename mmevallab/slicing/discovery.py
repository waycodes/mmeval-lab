"""Automated slice discovery via feature conjunction search."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Any


@dataclass
class DiscoveredSlice:
    """A discovered slice with performance metrics."""

    features: tuple[tuple[str, Any], ...]
    accuracy: float
    count: int
    delta_from_overall: float


def discover_slices(
    predictions: list[dict[str, Any]],
    feature_fields: list[str],
    min_count: int = 10,
    max_conjunction: int = 2,
    top_k: int = 20,
) -> list[DiscoveredSlice]:
    """Discover underperforming slices via feature conjunction search."""
    # Compute overall accuracy
    total_correct = sum(1 for p in predictions if p.get("is_correct"))
    overall_acc = total_correct / len(predictions) if predictions else 0.0

    # Extract feature values
    feature_values: dict[str, set[Any]] = defaultdict(set)
    for p in predictions:
        meta = p.get("metadata", {})
        for field in feature_fields:
            if field in meta:
                feature_values[field].add(meta[field])

    # Generate conjunctions
    discovered = []
    for conj_size in range(1, max_conjunction + 1):
        for fields in combinations(feature_fields, conj_size):
            # Generate all value combinations
            value_lists = [list(feature_values[f]) for f in fields]
            for values in _product(*value_lists):
                features = tuple(zip(fields, values))
                # Filter predictions matching this conjunction
                matching = [
                    p
                    for p in predictions
                    if all(p.get("metadata", {}).get(f) == v for f, v in features)
                ]
                if len(matching) < min_count:
                    continue

                correct = sum(1 for p in matching if p.get("is_correct"))
                acc = correct / len(matching)
                delta = acc - overall_acc

                discovered.append(
                    DiscoveredSlice(
                        features=features,
                        accuracy=acc,
                        count=len(matching),
                        delta_from_overall=delta,
                    )
                )

    # Sort by delta (worst first)
    discovered.sort(key=lambda x: x.delta_from_overall)
    return discovered[:top_k]


def _product(*iterables: list[Any]) -> list[tuple[Any, ...]]:
    """Simple cartesian product."""
    if not iterables:
        return [()]
    result = []
    for item in iterables[0]:
        for rest in _product(*iterables[1:]):
            result.append((item,) + rest)
    return result
