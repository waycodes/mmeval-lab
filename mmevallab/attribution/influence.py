"""Optional: TracIn-like influence attribution prototype."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class InfluenceScore:
    """Influence score of a training example on a test example."""

    test_id: str
    train_id: str
    influence: float
    checkpoint: str | None = None


def compute_tracin_influence(
    test_gradients: dict[str, Any],
    train_gradients: dict[str, Any],
    checkpoints: list[str] | None = None,
) -> float:
    """Compute TracIn-style influence (dot product of gradients).

    This is a simplified prototype - real implementation requires:
    - Gradient checkpoints during training
    - Efficient gradient storage/retrieval
    - GPU acceleration for large-scale computation
    """
    # Placeholder: would compute dot product of gradients
    # influence = sum(test_grad * train_grad for all parameters)
    return 0.0


def rank_influential_examples(
    test_id: str,
    train_influences: list[InfluenceScore],
    top_k: int = 10,
    proponents: bool = True,
) -> list[InfluenceScore]:
    """Rank training examples by influence on test example.

    Args:
        test_id: Test example ID
        train_influences: List of influence scores
        top_k: Number of examples to return
        proponents: If True, return most helpful (positive influence)
                   If False, return most harmful (negative influence)
    """
    filtered = [s for s in train_influences if s.test_id == test_id]
    filtered.sort(key=lambda x: x.influence, reverse=proponents)
    return filtered[:top_k]
