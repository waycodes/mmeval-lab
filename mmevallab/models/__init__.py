"""Model runners and backends."""

import re
import time

from mmevallab.core.datamodel import Example, Prediction
from mmevallab.core.registry import ModelRunner, register_model


@register_model("dummy")
class DummyModel(ModelRunner):
    """Dummy model that returns fixed answers for testing."""

    @property
    def name(self) -> str:
        return "dummy"

    def generate(self, example: Example) -> Prediction:
        """Generate a dummy prediction."""
        start = time.perf_counter()
        question = example.inputs.get("question", "")
        # Extract "What is X+1?" pattern
        match = re.search(r"What is (\d+)\+1\?", question)
        answer = str(int(match.group(1)) + 1) if match else "1"

        latency = (time.perf_counter() - start) * 1000
        return Prediction(
            example_id=example.example_id,
            raw_output=f"The answer is {answer}",
            extracted_answer=answer,
            latency_ms=latency,
        )
