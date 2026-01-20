"""Example and Prediction datamodels for benchmark evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class MediaRef(BaseModel):
    """Reference to media content (image, video, PDF page)."""

    type: str = Field(description="Media type: image, video, pdf_page")
    path: Path | None = None
    url: str | None = None
    page_num: int | None = Field(default=None, description="Page number for PDFs")
    frame_indices: list[int] | None = Field(default=None, description="Frame indices for video")


class Example(BaseModel):
    """A single evaluation example from a benchmark.

    This is the input to a model runner. All benchmark adapters must produce
    Example objects with stable example_ids.
    """

    example_id: str = Field(description="Unique, stable identifier")
    inputs: dict[str, Any] = Field(
        description="Input fields: question, options, etc.",
        default_factory=dict,
    )
    media: list[MediaRef] = Field(
        default_factory=list,
        description="References to images, video frames, or PDF pages",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Benchmark-specific metadata for slicing",
    )
    ground_truth: str | None = Field(
        default=None,
        description="Ground truth answer (None if withheld)",
    )


class Prediction(BaseModel):
    """Model prediction for a single example.

    This is the output from a model runner, before scoring.
    """

    example_id: str = Field(description="Matches Example.example_id")
    raw_output: str = Field(description="Raw model output text")
    extracted_answer: str | None = Field(
        default=None,
        description="Parsed/extracted answer (e.g., 'B' for MCQ)",
    )
    latency_ms: float = Field(description="Inference latency in milliseconds")
    tokens_in: int | None = Field(default=None, description="Input token count")
    tokens_out: int | None = Field(default=None, description="Output token count")
    model_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Model-specific metadata (logprobs, etc.)",
    )
