"""Prompt drift detection for run comparison."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PromptDrift:
    """Detected prompt drift between runs."""

    run_a_hash: str
    run_b_hash: str
    run_a_version: str
    run_b_version: str
    drifted: bool


def detect_prompt_drift(
    run_a_metadata: dict[str, str],
    run_b_metadata: dict[str, str],
) -> PromptDrift:
    """Detect if prompt templates differ between runs."""
    a_hash = run_a_metadata.get("prompt_template_hash", "")
    b_hash = run_b_metadata.get("prompt_template_hash", "")
    a_ver = run_a_metadata.get("prompt_template_version", "unknown")
    b_ver = run_b_metadata.get("prompt_template_version", "unknown")

    return PromptDrift(
        run_a_hash=a_hash,
        run_b_hash=b_hash,
        run_a_version=a_ver,
        run_b_version=b_ver,
        drifted=a_hash != b_hash,
    )
