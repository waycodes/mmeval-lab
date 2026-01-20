"""Attribution sanity checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mmevallab.attribution.similarity import Attribution


@dataclass
class SanityCheckResult:
    """Result of an attribution sanity check."""

    check_name: str
    passed: bool
    message: str
    details: dict[str, Any] | None = None


def check_self_attribution(
    attributions: dict[str, list[Attribution]],
    test_ids: set[str],
) -> SanityCheckResult:
    """Check that test examples don't attribute to themselves."""
    violations = []
    for test_id, attrs in attributions.items():
        for attr in attrs:
            if attr.train_id in test_ids:
                violations.append((test_id, attr.train_id))

    msg = f"Found {len(violations)} self-attributions" if violations else "No self-attributions"
    return SanityCheckResult(
        check_name="self_attribution",
        passed=len(violations) == 0,
        message=msg,
        details={"violations": violations[:10]} if violations else None,
    )


def check_attribution_coverage(
    attributions: dict[str, list[Attribution]],
    min_coverage: float = 0.5,
) -> SanityCheckResult:
    """Check that enough test examples have attributions."""
    total = len(attributions)
    with_attrs = sum(1 for attrs in attributions.values() if attrs)
    coverage = with_attrs / total if total > 0 else 0.0

    return SanityCheckResult(
        check_name="attribution_coverage",
        passed=coverage >= min_coverage,
        message=f"Coverage: {coverage:.1%} (min: {min_coverage:.1%})",
        details={"coverage": coverage, "with_attrs": with_attrs, "total": total},
    )


def run_sanity_checks(
    attributions: dict[str, list[Attribution]],
    test_ids: set[str],
) -> list[SanityCheckResult]:
    """Run all attribution sanity checks."""
    return [
        check_self_attribution(attributions, test_ids),
        check_attribution_coverage(attributions),
    ]
