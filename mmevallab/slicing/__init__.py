"""Slice system for built-in and discovered slices."""

from mmevallab.slicing.engine import (
    SliceResult,
    SliceSpec,
    compute_slice_metrics,
    evaluate_condition,
    evaluate_slice,
    evaluate_slices,
    parse_slice_spec,
)

__all__ = [
    "SliceResult",
    "SliceSpec",
    "compute_slice_metrics",
    "evaluate_condition",
    "evaluate_slice",
    "evaluate_slices",
    "parse_slice_spec",
]
