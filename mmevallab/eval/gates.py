"""Slice regression gates for CI/CD."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RegressionGate:
    """A regression gate for a slice."""

    slice_name: str
    metric: str
    threshold: float
    direction: str = "gte"  # gte = greater than or equal, lte = less than or equal


@dataclass
class GateResult:
    """Result of evaluating a regression gate."""

    gate: RegressionGate
    value: float
    passed: bool
    message: str


def evaluate_gates(
    metrics: dict[str, dict[str, float]],
    gates: list[RegressionGate],
) -> list[GateResult]:
    """Evaluate regression gates against metrics."""
    results = []
    for gate in gates:
        slice_metrics = metrics.get(gate.slice_name, {})
        value = slice_metrics.get(gate.metric, 0.0)

        if gate.direction == "gte":
            passed = value >= gate.threshold
            op = ">="
        else:
            passed = value <= gate.threshold
            op = "<="

        msg = f"{gate.slice_name}.{gate.metric}: {value:.4f} {op} {gate.threshold:.4f}"
        results.append(GateResult(gate=gate, value=value, passed=passed, message=msg))

    return results


def check_all_gates(results: list[GateResult]) -> tuple[bool, list[str]]:
    """Check if all gates passed."""
    failures = [r.message for r in results if not r.passed]
    return len(failures) == 0, failures
