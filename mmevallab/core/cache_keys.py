"""Request canonicalization and cache key generation."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonicalize_request(
    model_id: str,
    prompt: str,
    images: list[bytes] | None = None,
    generation_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Canonicalize a model request for caching."""
    image_hashes = []
    if images:
        for img in images:
            image_hashes.append(hashlib.sha256(img).hexdigest()[:16])

    return {
        "model_id": model_id,
        "prompt": prompt,
        "image_hashes": image_hashes,
        "generation_params": generation_params or {},
    }


def compute_cache_key(request: dict[str, Any]) -> str:
    """Compute cache key from canonicalized request."""
    canonical = json.dumps(request, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()
