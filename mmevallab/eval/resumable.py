"""Resume-safe inference execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from mmevallab.core.datamodel import Example, Prediction


class ResumableRunner:
    """Runner that can resume from partial results."""

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self._completed: set[str] = set()
        self._load_completed()

    def _load_completed(self) -> None:
        if self.output_path.exists():
            with open(self.output_path) as f:
                for line in f:
                    rec = json.loads(line)
                    self._completed.add(rec["example_id"])

    def is_completed(self, example_id: str) -> bool:
        return example_id in self._completed

    def pending(self, examples: list[Example]) -> Iterator[Example]:
        for ex in examples:
            if not self.is_completed(ex.example_id):
                yield ex

    def record(self, prediction: Prediction, extra: dict[str, Any] | None = None) -> None:
        record = {
            "example_id": prediction.example_id,
            "raw_output": prediction.raw_output,
            "extracted_answer": prediction.extracted_answer,
            "latency_ms": prediction.latency_ms,
            **(extra or {}),
        }
        with open(self.output_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        self._completed.add(prediction.example_id)

    @property
    def num_completed(self) -> int:
        return len(self._completed)
