"""Core datamodels, schemas, and utilities."""

from mmevallab.core.datamodel import Example, MediaRef, Prediction
from mmevallab.core.hashing import compute_dataset_id, compute_run_id, get_code_version
from mmevallab.core.registry import (
    Benchmark,
    ModelRunner,
    benchmark_registry,
    model_registry,
    register_benchmark,
    register_model,
)
from mmevallab.core.schema import (
    DatasetVersion,
    ExampleOutput,
    RunArtifact,
    RunConfig,
    RunMetadata,
    RunMetrics,
    SliceAnnotation,
    SliceMetrics,
)

__all__ = [
    "Benchmark",
    "DatasetVersion",
    "Example",
    "ExampleOutput",
    "MediaRef",
    "ModelRunner",
    "Prediction",
    "RunArtifact",
    "RunConfig",
    "RunMetadata",
    "RunMetrics",
    "SliceAnnotation",
    "SliceMetrics",
    "benchmark_registry",
    "compute_dataset_id",
    "compute_run_id",
    "get_code_version",
    "model_registry",
    "register_benchmark",
    "register_model",
]
