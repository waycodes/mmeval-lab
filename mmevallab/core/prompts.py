"""Prompt templates for benchmarks."""

from __future__ import annotations

import hashlib
from typing import Any


def _compute_template_hash(template: str) -> str:
    """Compute hash of template for reproducibility tracking."""
    return hashlib.sha256(template.encode()).hexdigest()[:8]


class PromptTemplate:
    """A versioned prompt template."""

    def __init__(self, name: str, version: str, template: str) -> None:
        self.name = name
        self.version = version
        self.template = template
        self.hash = _compute_template_hash(template)

    def format(self, **kwargs: Any) -> str:
        """Format the template with provided values."""
        return self.template.format(**kwargs)

    def __repr__(self) -> str:
        return f"PromptTemplate({self.name}:{self.version}, hash={self.hash})"


# MMMU Prompt Templates
MMMU_V1_TEMPLATE = """{question}

{options}

Answer with the letter of the correct option."""

MMMU_V2_TEMPLATE = """{question}

{options}

Respond with ONLY the letter (A, B, C, D, or E) of the correct answer.
Do not include any explanation."""

MMMU_TEMPLATES = {
    "v1": PromptTemplate("mmmu", "v1", MMMU_V1_TEMPLATE),
    "v2": PromptTemplate("mmmu", "v2", MMMU_V2_TEMPLATE),
}


def get_mmmu_template(version: str = "v1") -> PromptTemplate:
    """Get MMMU prompt template by version."""
    if version not in MMMU_TEMPLATES:
        available = ", ".join(MMMU_TEMPLATES.keys())
        raise ValueError(f"Unknown MMMU template version: {version}. Available: {available}")
    return MMMU_TEMPLATES[version]


def format_mmmu_prompt(
    question: str,
    options: list[str],
    template_version: str = "v1",
) -> tuple[str, str]:
    """Format MMMU prompt and return (prompt, template_hash).

    Args:
        question: The question text
        options: List of options like ["A. option1", "B. option2", ...]
        template_version: Template version to use

    Returns:
        Tuple of (formatted_prompt, template_hash)
    """
    template = get_mmmu_template(template_version)
    options_str = "\n".join(options)
    prompt = template.format(question=question, options=options_str)
    return prompt, template.hash
