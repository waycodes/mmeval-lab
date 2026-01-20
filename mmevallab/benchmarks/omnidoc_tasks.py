"""OmniDocBench task variant definitions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class OmniDocTask(str, Enum):
    """OmniDocBench task variants."""

    E2E_MARKDOWN = "e2e_markdown"
    OCR_TEXT = "ocr_text"
    FORMULA = "formula"
    TABLE = "table"
    LAYOUT = "layout"


@dataclass
class TaskConfig:
    """Configuration for a task variant."""

    name: str
    prompt_template: str
    metric: str
    output_format: str


TASK_CONFIGS: dict[OmniDocTask, TaskConfig] = {
    OmniDocTask.E2E_MARKDOWN: TaskConfig(
        name="End-to-end Markdown",
        prompt_template="Convert this document page to Markdown format.",
        metric="edit_similarity",
        output_format="markdown",
    ),
    OmniDocTask.OCR_TEXT: TaskConfig(
        name="OCR Text Extraction",
        prompt_template="Extract all text from this document image.",
        metric="edit_similarity",
        output_format="text",
    ),
    OmniDocTask.FORMULA: TaskConfig(
        name="Formula Recognition",
        prompt_template="Extract the mathematical formula from this image as LaTeX.",
        metric="edit_similarity",
        output_format="latex",
    ),
    OmniDocTask.TABLE: TaskConfig(
        name="Table Recognition",
        prompt_template="Extract the table from this image as HTML.",
        metric="teds",
        output_format="html",
    ),
    OmniDocTask.LAYOUT: TaskConfig(
        name="Layout Analysis",
        prompt_template="Identify the layout regions in this document.",
        metric="iou",
        output_format="json",
    ),
}


def get_task_config(task: str | OmniDocTask) -> TaskConfig:
    """Get configuration for a task variant."""
    if isinstance(task, str):
        task = OmniDocTask(task)
    return TASK_CONFIGS[task]
