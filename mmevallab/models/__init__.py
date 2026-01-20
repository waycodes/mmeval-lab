"""Model runners and backends."""

import re
import time
from typing import Any

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


@register_model("hf_vlm")
class HFVLMBackend(ModelRunner):
    """HuggingFace Transformers VLM backend."""

    def __init__(
        self,
        model_id: str = "llava-hf/llava-1.5-7b-hf",
        device: str = "auto",
        torch_dtype: str = "auto",
        **kwargs: Any,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        self._model: Any = None
        self._processor: Any = None

    @property
    def name(self) -> str:
        return f"hf_vlm:{self.model_id}"

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor

            dtype = getattr(torch, self.torch_dtype) if self.torch_dtype != "auto" else "auto"
            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = AutoModelForVision2Seq.from_pretrained(
                self.model_id, torch_dtype=dtype, device_map=self.device
            )
        except ImportError as e:
            raise ImportError("Install transformers: pip install transformers torch") from e

    def generate(self, example: Example) -> Prediction:
        self._load()
        start = time.perf_counter()

        images = example.inputs.get("images", [])
        question = example.inputs.get("question", "")

        inputs = self._processor(text=question, images=images or None, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        outputs = self._model.generate(**inputs, max_new_tokens=256)
        raw_output = self._processor.decode(outputs[0], skip_special_tokens=True)

        latency = (time.perf_counter() - start) * 1000
        return Prediction(
            example_id=example.example_id,
            raw_output=raw_output,
            extracted_answer=raw_output.strip(),
            latency_ms=latency,
        )
