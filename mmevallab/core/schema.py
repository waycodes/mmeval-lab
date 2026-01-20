"""Run artifact schema definitions."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class DatasetVersion(BaseModel):
    """Dataset version identifier for reproducibility."""

    name: str
    version: str
    split: str
    num_examples: int
    content_hash: str = Field(description="Hash of dataset content")


class RunConfig(BaseModel):
    """Configuration used for an evaluation run."""

    benchmark: str
    model: str
    split: str
    prompt_template: str
    prompt_template_hash: str
    decoding_params: dict[str, Any] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)


class RunMetadata(BaseModel):
    """Metadata for an evaluation run."""

    run_id: str
    created_at: datetime
    config: RunConfig
    dataset_version: DatasetVersion
    code_version: str = Field(description="Git commit hash")
    python_version: str
    package_version: str


class SliceAnnotation(BaseModel):
    """Slice membership for an example."""

    slice_name: str
    slice_value: str


class ExampleOutput(BaseModel):
    """Per-example output from evaluation."""

    example_id: str
    raw_output: str
    extracted_answer: str | None = None
    is_correct: bool | None = None
    latency_ms: float
    tokens_in: int | None = None
    tokens_out: int | None = None
    slices: list[SliceAnnotation] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SliceMetrics(BaseModel):
    """Metrics for a single slice."""

    slice_name: str
    slice_value: str
    count: int
    accuracy: float
    metrics: dict[str, float] = Field(default_factory=dict)


class RunMetrics(BaseModel):
    """Aggregated metrics for a run."""

    overall_accuracy: float
    total_examples: int
    correct: int
    slice_metrics: list[SliceMetrics] = Field(default_factory=list)
    extra_metrics: dict[str, float] = Field(default_factory=dict)


class RunArtifact(BaseModel):
    """Complete run artifact combining all outputs."""

    metadata: RunMetadata
    metrics: RunMetrics
    outputs: list[ExampleOutput] = Field(default_factory=list)
