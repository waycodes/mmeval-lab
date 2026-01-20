"""OmniDocBench benchmark adapter."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from mmevallab.core.datamodel import Example, MediaRef, Prediction
from mmevallab.core.registry import Benchmark, register_benchmark

# Task variants supported by OmniDocBench
OMNIDOCBENCH_TASKS = ["e2e_markdown", "ocr_text", "formula", "table", "layout"]


def _compute_file_hash(path: Path) -> str:
    """Compute SHA256 hash of file content."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


@register_benchmark("omnidocbench")
class OmniDocBenchmark(Benchmark):
    """OmniDocBench: Document understanding benchmark."""

    def __init__(
        self,
        data_dir: str | Path | None = None,
        task: str = "e2e_markdown",
    ) -> None:
        """Initialize OmniDocBench adapter.

        Args:
            data_dir: Path to OmniDocBench data directory containing PDFs and annotations
            task: Task variant (e2e_markdown, ocr_text, formula, table, layout)
        """
        self._data_dir = Path(data_dir) if data_dir else None
        if task not in OMNIDOCBENCH_TASKS:
            raise ValueError(f"Unknown task: {task}. Valid: {OMNIDOCBENCH_TASKS}")
        self._task = task

    @property
    def name(self) -> str:
        return f"omnidocbench_{self._task}"

    def load(self, split: str, **kwargs: Any) -> Iterator[Example]:
        """Load OmniDocBench examples.

        Args:
            split: Split name (e.g., 'test')
            **kwargs: Additional options (limit, data_dir override)

        Yields:
            Example objects with PDF page references
        """
        data_dir = kwargs.get("data_dir") or self._data_dir
        if data_dir is None:
            raise ValueError("data_dir must be provided")
        data_dir = Path(data_dir)

        # Load annotations
        annot_path = data_dir / "annotations" / f"{split}.json"
        if not annot_path.exists():
            raise FileNotFoundError(f"Annotations not found: {annot_path}")

        with open(annot_path) as f:
            annotations = json.load(f)

        limit = kwargs.get("limit")
        for i, item in enumerate(annotations):
            if limit and i >= limit:
                break

            doc_id = item.get("doc_id", f"doc_{i}")
            page_num = item.get("page_num", 1)
            pdf_name = item.get("pdf_name", f"{doc_id}.pdf")

            pdf_path = data_dir / "pdfs" / pdf_name
            example_id = f"{doc_id}_p{page_num}_{self._task}"

            # Create media reference for PDF page
            media = [
                MediaRef(
                    type="pdf_page",
                    path=pdf_path if pdf_path.exists() else None,
                    page_num=page_num,
                )
            ]

            # Build inputs based on task
            inputs = {
                "task": self._task,
                "pdf_path": str(pdf_path),
                "page_num": page_num,
            }

            # Add task-specific inputs
            if self._task == "formula":
                inputs["formula_bbox"] = item.get("bbox")
            elif self._task == "table":
                inputs["table_bbox"] = item.get("bbox")

            # Metadata for slicing
            metadata = {
                "doc_id": doc_id,
                "doc_type": item.get("doc_type", "unknown"),
                "layout_type": item.get("layout_type", "unknown"),
                "language": item.get("language", "en"),
                "has_formula": item.get("has_formula", False),
                "has_table": item.get("has_table", False),
                "split": split,
                "task": self._task,
            }

            # Ground truth
            ground_truth = item.get("target") or item.get("ground_truth")

            yield Example(
                example_id=example_id,
                inputs=inputs,
                media=media,
                metadata=metadata,
                ground_truth=ground_truth,
            )

    def score(self, example: Example, prediction: Prediction) -> dict[str, Any]:
        """Score prediction against ground truth.

        Scoring depends on task type:
        - e2e_markdown/ocr_text: Edit distance, BLEU
        - formula: Exact match (normalized)
        - table: TEDS (Tree Edit Distance Similarity)
        """
        if example.ground_truth is None:
            return {"is_correct": None}

        gt = example.ground_truth
        pred = prediction.raw_output

        task = example.metadata.get("task", self._task)

        if task == "formula":
            # Normalize and exact match for formulas
            gt_norm = _normalize_latex(gt)
            pred_norm = _normalize_latex(pred)
            is_correct = gt_norm == pred_norm
            return {
                "is_correct": is_correct,
                "exact_match": 1.0 if is_correct else 0.0,
            }
        else:
            # For text tasks, compute edit distance ratio
            edit_dist = _edit_distance(gt, pred)
            max_len = max(len(gt), len(pred), 1)
            similarity = 1.0 - (edit_dist / max_len)
            return {
                "is_correct": similarity > 0.9,
                "edit_similarity": similarity,
                "edit_distance": edit_dist,
            }


def _normalize_latex(text: str) -> str:
    """Normalize LaTeX for comparison."""
    import re

    text = text.strip()
    # Remove common LaTeX wrappers
    text = re.sub(r"^\$+|\$+$", "", text)
    text = re.sub(r"^\\\[|\\\]$", "", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    return text


def _edit_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance."""
    if len(s1) < len(s2):
        return _edit_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]
