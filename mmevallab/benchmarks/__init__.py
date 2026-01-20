"""Benchmark adapters for MMMU, OmniDocBench, Video-MME."""

from collections.abc import Iterator
from typing import Any

# Import benchmark implementations to trigger registration
from mmevallab.benchmarks import mmmu as _mmmu  # noqa: F401
from mmevallab.core.datamodel import Example, Prediction
from mmevallab.core.registry import Benchmark, register_benchmark


@register_benchmark("dummy")
class DummyBenchmark(Benchmark):
    """Dummy benchmark for testing."""

    @property
    def name(self) -> str:
        return "dummy"

    def load(self, split: str, **kwargs: Any) -> Iterator[Example]:
        """Yield a few dummy examples."""
        for i in range(3):
            yield Example(
                example_id=f"dummy_{split}_{i}",
                inputs={"question": f"What is {i}+1?", "options": ["A", "B", "C", "D"]},
                metadata={"split": split},
                ground_truth=str(i + 1),
            )

    def score(self, example: Example, prediction: Prediction) -> dict[str, Any]:
        """Score by exact match."""
        is_correct = prediction.extracted_answer == example.ground_truth
        return {"is_correct": is_correct, "accuracy": 1.0 if is_correct else 0.0}
