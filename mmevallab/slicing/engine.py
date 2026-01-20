"""Slice specification language and evaluator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SliceSpec:
    """Specification for a data slice."""

    name: str
    field: str
    op: str = "eq"  # eq, ne, in, contains, gt, lt, gte, lte
    value: Any = None
    values: list[Any] = field(default_factory=list)


@dataclass
class SliceResult:
    """Result of slice evaluation."""

    name: str
    count: int
    example_ids: list[str]


def parse_slice_spec(spec: dict[str, Any]) -> SliceSpec:
    """Parse a slice spec from dict/YAML format."""
    return SliceSpec(
        name=spec["name"],
        field=spec["field"],
        op=spec.get("op", "eq"),
        value=spec.get("value"),
        values=spec.get("values", []),
    )


def evaluate_condition(item: dict[str, Any], spec: SliceSpec) -> bool:
    """Evaluate if an item matches a slice condition."""
    val = item.get(spec.field)
    if val is None:
        # Check nested in metadata
        val = item.get("metadata", {}).get(spec.field)

    if spec.op == "eq":
        return val == spec.value
    elif spec.op == "ne":
        return val != spec.value
    elif spec.op == "in":
        return val in spec.values
    elif spec.op == "contains":
        return spec.value in str(val) if val else False
    elif spec.op == "gt":
        return val > spec.value if val is not None else False
    elif spec.op == "lt":
        return val < spec.value if val is not None else False
    elif spec.op == "gte":
        return val >= spec.value if val is not None else False
    elif spec.op == "lte":
        return val <= spec.value if val is not None else False
    return False


def evaluate_slice(
    items: list[dict[str, Any]],
    spec: SliceSpec,
) -> SliceResult:
    """Evaluate a slice spec against a list of items."""
    matching_ids = []
    for item in items:
        if evaluate_condition(item, spec):
            matching_ids.append(item.get("example_id", ""))
    return SliceResult(name=spec.name, count=len(matching_ids), example_ids=matching_ids)


def evaluate_slices(
    items: list[dict[str, Any]],
    specs: list[SliceSpec],
) -> list[SliceResult]:
    """Evaluate multiple slice specs."""
    return [evaluate_slice(items, spec) for spec in specs]


def compute_slice_metrics(
    predictions: list[dict[str, Any]],
    slice_result: SliceResult,
) -> dict[str, Any]:
    """Compute metrics for a slice."""
    pred_map = {p["example_id"]: p for p in predictions}
    correct = 0
    total = 0
    for eid in slice_result.example_ids:
        pred = pred_map.get(eid)
        if pred and pred.get("is_correct") is not None:
            total += 1
            if pred["is_correct"]:
                correct += 1
    accuracy = correct / total if total > 0 else 0.0
    return {
        "slice_name": slice_result.name,
        "count": slice_result.count,
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
    }
