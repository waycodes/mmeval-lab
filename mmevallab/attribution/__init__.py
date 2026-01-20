"""Data-diff attribution for slice deltas."""

from mmevallab.attribution.delta import (
    ExampleDelta,
    SliceDelta,
    canonicalize_run_delta,
    compute_example_deltas,
    compute_slice_deltas,
)
from mmevallab.attribution.diff import compute_dataset_diff, save_diff_report

__all__ = [
    "ExampleDelta",
    "SliceDelta",
    "canonicalize_run_delta",
    "compute_dataset_diff",
    "compute_example_deltas",
    "compute_slice_deltas",
    "save_diff_report",
]
