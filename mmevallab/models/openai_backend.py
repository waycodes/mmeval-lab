"""Optional: OpenAI-compatible API backend."""

from __future__ import annotations

import base64
import time
from typing import Any

from mmevallab.core.datamodel import Example, Prediction
from mmevallab.core.registry import ModelRunner, register_model


@register_model("openai")
class OpenAIBackend(ModelRunner):
    """OpenAI-compatible API backend."""

    def __init__(
        self,
        model: str = "gpt-4-vision-preview",
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 256,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.max_tokens = max_tokens
        self._client: Any = None

    @property
    def name(self) -> str:
        return f"openai:{self.model}"

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            except ImportError as e:
                raise ImportError("Install openai: pip install openai") from e
        return self._client

    def generate(self, example: Example) -> Prediction:
        client = self._get_client()
        start = time.perf_counter()

        messages = self._build_messages(example)

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
        )

        raw_output = response.choices[0].message.content or ""
        latency = (time.perf_counter() - start) * 1000

        return Prediction(
            example_id=example.example_id,
            raw_output=raw_output,
            extracted_answer=raw_output.strip(),
            latency_ms=latency,
        )

    def _build_messages(self, example: Example) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []

        # Add images
        images = example.inputs.get("images", [])
        for img in images:
            if hasattr(img, "tobytes"):
                import io

                buf = io.BytesIO()
                img.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode()
                content.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                )

        # Add text
        question = example.inputs.get("question", "")
        content.append({"type": "text", "text": question})

        return [{"role": "user", "content": content}]
