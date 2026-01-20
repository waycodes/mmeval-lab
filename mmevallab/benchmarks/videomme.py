"""Video-MME benchmark adapter."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from mmevallab.core.datamodel import Example, MediaRef, Prediction
from mmevallab.core.registry import Benchmark, register_benchmark

# License gate error message
LICENSE_GATE_MSG = """
Video-MME requires explicit license acceptance.

To use Video-MME, you must:
1. Accept the Video-MME license terms
2. Set accept_license=True when creating the benchmark

Example:
    benchmark = VideoMMEBenchmark(data_dir="/path/to/data", accept_license=True)

Note: Video files must not be redistributed. Only use locally downloaded videos.
"""


class LicenseNotAcceptedError(Exception):
    """Raised when Video-MME license has not been accepted."""

    pass


@register_benchmark("videomme")
class VideoMMEBenchmark(Benchmark):
    """Video-MME: Video understanding benchmark with MCQ questions."""

    def __init__(
        self,
        data_dir: str | Path | None = None,
        accept_license: bool = False,
    ) -> None:
        """Initialize Video-MME adapter.

        Args:
            data_dir: Path to Video-MME data directory
            accept_license: Must be True to use this benchmark

        Raises:
            LicenseNotAcceptedError: If accept_license is False
        """
        if not accept_license:
            raise LicenseNotAcceptedError(LICENSE_GATE_MSG)

        self._data_dir = Path(data_dir) if data_dir else None
        self._license_accepted = True

    @property
    def name(self) -> str:
        return "videomme"

    def _check_license(self) -> None:
        """Verify license was accepted."""
        if not self._license_accepted:
            raise LicenseNotAcceptedError(LICENSE_GATE_MSG)

    def load(self, split: str, **kwargs: Any) -> Iterator[Example]:
        """Load Video-MME examples.

        Args:
            split: Split name (e.g., 'test')
            **kwargs: Additional options (limit, data_dir override)

        Yields:
            Example objects with video references

        Raises:
            LicenseNotAcceptedError: If license not accepted
        """
        self._check_license()

        data_dir = kwargs.get("data_dir") or self._data_dir
        if data_dir is None:
            raise ValueError("data_dir must be provided")
        data_dir = Path(data_dir)

        # Load annotations
        annot_path = data_dir / "annotations" / f"{split}.json"
        if not annot_path.exists():
            raise FileNotFoundError(f"Annotations not found: {annot_path}")

        with open(annot_path) as f:
            annotations = json.load(f)

        limit = kwargs.get("limit")
        for i, item in enumerate(annotations):
            if limit and i >= limit:
                break

            video_id = item.get("video_id", f"video_{i}")
            question_id = item.get("question_id", 0)
            video_name = item.get("video_name", f"{video_id}.mp4")

            video_path = data_dir / "videos" / video_name
            example_id = f"{video_id}_q{question_id}"

            # Create media reference for video
            media = [
                MediaRef(
                    type="video",
                    path=video_path if video_path.exists() else None,
                )
            ]

            # Build options
            options = []
            for opt_key in ["A", "B", "C", "D"]:
                opt = item.get(opt_key)
                if opt:
                    options.append(f"{opt_key}. {opt}")

            # Build inputs
            inputs = {
                "question": item.get("question", ""),
                "options": options,
                "video_path": str(video_path),
            }

            # Subtitles if available
            subtitles = item.get("subtitles", [])
            if subtitles:
                inputs["subtitles"] = subtitles

            # Metadata for slicing
            metadata = {
                "video_id": video_id,
                "duration": item.get("duration", "unknown"),
                "domain": item.get("domain", "unknown"),
                "subcategory": item.get("subcategory", "unknown"),
                "task_type": item.get("task_type", "unknown"),
                "split": split,
            }

            # Ground truth
            ground_truth = item.get("answer")

            yield Example(
                example_id=example_id,
                inputs=inputs,
                media=media,
                metadata=metadata,
                ground_truth=ground_truth,
            )

    def score(self, example: Example, prediction: Prediction) -> dict[str, Any]:
        """Score prediction against ground truth."""
        self._check_license()

        if example.ground_truth is None:
            return {"is_correct": None, "accuracy": None}

        gt = example.ground_truth.strip().upper()
        pred = (prediction.extracted_answer or "").strip().upper()

        is_correct = pred == gt
        return {
            "is_correct": is_correct,
            "accuracy": 1.0 if is_correct else 0.0,
            "ground_truth": gt,
            "predicted": pred,
        }
