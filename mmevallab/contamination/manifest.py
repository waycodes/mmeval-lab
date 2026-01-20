"""Training manifest schema and loader for contamination scanning."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class TrainingSample(BaseModel):
    """A single training sample."""

    sample_id: str
    modality: str = Field(description="text, image, video, multimodal")
    text: str | None = None
    image_ref: str | None = None
    video_ref: str | None = None
    source: str | None = None
    license: str | None = None
    timestamp: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def load_manifest_jsonl(path: Path | str) -> Iterator[TrainingSample]:
    """Load training manifest from JSONL file."""
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            yield TrainingSample.model_validate(data)


def load_manifest_parquet(path: Path | str) -> Iterator[TrainingSample]:
    """Load training manifest from Parquet file."""
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("Install pandas: pip install pandas") from e

    df = pd.read_parquet(path)
    for _, row in df.iterrows():
        yield TrainingSample(
            sample_id=str(row.get("sample_id", "")),
            modality=str(row.get("modality", "text")),
            text=row.get("text"),
            image_ref=row.get("image_ref"),
            video_ref=row.get("video_ref"),
            source=row.get("source"),
            license=row.get("license"),
            timestamp=row.get("timestamp"),
        )


def load_manifest(path: Path | str) -> Iterator[TrainingSample]:
    """Load training manifest from file (auto-detect format)."""
    path = Path(path)
    if path.suffix == ".jsonl":
        yield from load_manifest_jsonl(path)
    elif path.suffix == ".parquet":
        yield from load_manifest_parquet(path)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")


def stream_manifest(path: Path | str, batch_size: int = 1000) -> Iterator[list[TrainingSample]]:
    """Stream manifest in batches for memory efficiency."""
    batch: list[TrainingSample] = []
    for sample in load_manifest(path):
        batch.append(sample)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
