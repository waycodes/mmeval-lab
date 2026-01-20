"""Sample logging artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class SampleLogger:
    """Log sample-level artifacts during evaluation."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._samples_path = output_dir / "samples.jsonl"

    def log_sample(
        self,
        example_id: str,
        inputs: dict[str, Any],
        prediction: str,
        ground_truth: str | None = None,
        is_correct: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        record = {
            "example_id": example_id,
            "inputs": {k: v for k, v in inputs.items() if k != "images"},
            "prediction": prediction,
            "ground_truth": ground_truth,
            "is_correct": is_correct,
            "metadata": metadata,
        }
        with open(self._samples_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def log_image(self, example_id: str, image_bytes: bytes, suffix: str = ".png") -> Path:
        img_dir = self.output_dir / "images"
        img_dir.mkdir(exist_ok=True)
        path = img_dir / f"{example_id}{suffix}"
        path.write_bytes(image_bytes)
        return path
