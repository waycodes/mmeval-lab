"""Bootstrap confidence intervals for metrics."""

from __future__ import annotations

import random
from typing import Callable


def bootstrap_ci(
    values: list[float],
    statistic: Callable[[list[float]], float] = lambda x: sum(x) / len(x),
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval.

    Returns:
        Tuple of (point_estimate, lower_bound, upper_bound)
    """
    if not values:
        return 0.0, 0.0, 0.0

    rng = random.Random(seed)
    n = len(values)
    point = statistic(values)

    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = [rng.choice(values) for _ in range(n)]
        bootstrap_stats.append(statistic(sample))

    bootstrap_stats.sort()
    alpha = 1 - confidence
    lower_idx = int(alpha / 2 * n_bootstrap)
    upper_idx = int((1 - alpha / 2) * n_bootstrap) - 1

    return point, bootstrap_stats[lower_idx], bootstrap_stats[upper_idx]


def accuracy_with_ci(
    correct: list[bool],
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
) -> dict[str, float]:
    """Compute accuracy with confidence interval."""
    values = [1.0 if c else 0.0 for c in correct]
    point, lower, upper = bootstrap_ci(values, confidence=confidence, n_bootstrap=n_bootstrap)
    return {"accuracy": point, "ci_lower": lower, "ci_upper": upper, "confidence": confidence}
