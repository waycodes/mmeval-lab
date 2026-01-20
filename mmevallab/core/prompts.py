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


# OmniDocBench templates
OMNIDOC_TEMPLATES = {
    "e2e_markdown": PromptTemplate(
        "omnidoc_e2e", "v1", "Convert this document page to Markdown format."
    ),
    "ocr_text": PromptTemplate(
        "omnidoc_ocr", "v1", "Extract all text from this document image."
    ),
    "formula": PromptTemplate(
        "omnidoc_formula", "v1", "Extract the mathematical formula as LaTeX."
    ),
    "table": PromptTemplate(
        "omnidoc_table", "v1", "Extract the table as HTML."
    ),
}


# Video-MME templates
VIDEOMME_TEMPLATES = {
    "v1": PromptTemplate(
        "videomme", "v1",
        "{question}\n\n{options}\n\nAnswer with the letter of the correct option.",
    ),
}


def get_template(benchmark: str, task: str = "default", version: str = "v1") -> PromptTemplate:
    """Get prompt template for any benchmark."""
    if benchmark == "mmmu":
        return MMMU_TEMPLATES[version]
    elif benchmark == "omnidocbench":
        return OMNIDOC_TEMPLATES.get(task, OMNIDOC_TEMPLATES["e2e_markdown"])
    elif benchmark == "videomme":
        return VIDEOMME_TEMPLATES[version]
    raise ValueError(f"Unknown benchmark: {benchmark}")
