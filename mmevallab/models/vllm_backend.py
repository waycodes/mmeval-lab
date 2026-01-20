"""Optional: vLLM backend for high-throughput inference."""

from __future__ import annotations

import time
from typing import Any

from mmevallab.core.datamodel import Example, Prediction
from mmevallab.core.registry import ModelRunner, register_model


@register_model("vllm")
class VLLMBackend(ModelRunner):
    """vLLM backend for high-throughput inference."""

    def __init__(
        self,
        model: str,
        tensor_parallel_size: int = 1,
        max_tokens: int = 256,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.tensor_parallel_size = tensor_parallel_size
        self.max_tokens = max_tokens
        self._llm: Any = None

    @property
    def name(self) -> str:
        return f"vllm:{self.model}"

    def _load(self) -> None:
        if self._llm is not None:
            return
        try:
            from vllm import LLM

            self._llm = LLM(
                model=self.model,
                tensor_parallel_size=self.tensor_parallel_size,
            )
        except ImportError as e:
            raise ImportError("Install vllm: pip install vllm") from e

    def generate(self, example: Example) -> Prediction:
        self._load()
        start = time.perf_counter()

        from vllm import SamplingParams

        prompt = example.inputs.get("question", "")
        params = SamplingParams(max_tokens=self.max_tokens)

        outputs = self._llm.generate([prompt], params)
        raw_output = outputs[0].outputs[0].text

        latency = (time.perf_counter() - start) * 1000

        return Prediction(
            example_id=example.example_id,
            raw_output=raw_output,
            extracted_answer=raw_output.strip(),
            latency_ms=latency,
        )

    def generate_batch(self, examples: list[Example]) -> list[Prediction]:
        """Batch generation for higher throughput."""
        self._load()
        start = time.perf_counter()

        from vllm import SamplingParams

        prompts = [ex.inputs.get("question", "") for ex in examples]
        params = SamplingParams(max_tokens=self.max_tokens)

        outputs = self._llm.generate(prompts, params)
        total_latency = (time.perf_counter() - start) * 1000
        per_example_latency = total_latency / len(examples)

        return [
            Prediction(
                example_id=ex.example_id,
                raw_output=out.outputs[0].text,
                extracted_answer=out.outputs[0].text.strip(),
                latency_ms=per_example_latency,
            )
            for ex, out in zip(examples, outputs)
        ]
