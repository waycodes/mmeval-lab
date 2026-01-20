"""OmniDocBench official evaluation metrics wrapper."""

from __future__ import annotations

from typing import Any


def compute_teds(pred_html: str, gt_html: str) -> float:
    """Compute Tree Edit Distance Similarity for tables.

    TEDS measures structural similarity between HTML tables.
    Score ranges from 0 (completely different) to 1 (identical).

    Args:
        pred_html: Predicted table HTML
        gt_html: Ground truth table HTML

    Returns:
        TEDS score between 0 and 1
    """
    # Simplified TEDS implementation
    # In production, use the official TEDS implementation from:
    # https://github.com/ibm-aur-nlp/PubTabNet
    from html.parser import HTMLParser

    class TableParser(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.cells: list[str] = []
            self._current: list[str] = []
            self._in_cell = False

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            if tag in ("td", "th"):
                self._in_cell = True
                self._current = []

        def handle_endtag(self, tag: str) -> None:
            if tag in ("td", "th"):
                self._in_cell = False
                self.cells.append("".join(self._current).strip())

        def handle_data(self, data: str) -> None:
            if self._in_cell:
                self._current.append(data)

    def extract_cells(html: str) -> list[str]:
        parser = TableParser()
        try:
            parser.feed(html)
        except Exception:
            return []
        return parser.cells

    pred_cells = extract_cells(pred_html)
    gt_cells = extract_cells(gt_html)

    if not gt_cells:
        return 1.0 if not pred_cells else 0.0

    # Simple cell-level matching
    matches = sum(1 for p, g in zip(pred_cells, gt_cells) if p == g)
    max_len = max(len(pred_cells), len(gt_cells))

    return matches / max_len if max_len > 0 else 1.0


def compute_edit_distance(pred: str, gt: str) -> int:
    """Compute Levenshtein edit distance."""
    if len(pred) < len(gt):
        return compute_edit_distance(gt, pred)

    if len(gt) == 0:
        return len(pred)

    prev_row = list(range(len(gt) + 1))
    for i, c1 in enumerate(pred):
        curr_row = [i + 1]
        for j, c2 in enumerate(gt):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def compute_edit_similarity(pred: str, gt: str) -> float:
    """Compute edit distance similarity (1 - normalized edit distance)."""
    if not gt and not pred:
        return 1.0
    edit_dist = compute_edit_distance(pred, gt)
    max_len = max(len(pred), len(gt))
    return 1.0 - (edit_dist / max_len) if max_len > 0 else 1.0


def compute_bleu(pred: str, gt: str, n: int = 4) -> float:
    """Compute BLEU score for text comparison.

    Args:
        pred: Predicted text
        gt: Ground truth text
        n: Maximum n-gram order

    Returns:
        BLEU score between 0 and 1
    """
    import math
    from collections import Counter

    def get_ngrams(text: str, n: int) -> Counter[tuple[str, ...]]:
        words = text.split()
        if len(words) < n:
            return Counter()
        return Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))

    pred_words = pred.split()
    gt_words = gt.split()

    if not pred_words or not gt_words:
        return 0.0

    # Brevity penalty
    if len(pred_words) >= len(gt_words):
        bp = 1.0
    else:
        bp = math.exp(1 - len(gt_words) / len(pred_words))

    # N-gram precisions (use smoothing for short texts)
    precisions = []
    for i in range(1, min(n + 1, len(pred_words) + 1)):
        pred_ngrams = get_ngrams(pred, i)
        gt_ngrams = get_ngrams(gt, i)

        if not pred_ngrams:
            continue

        matches = sum((pred_ngrams & gt_ngrams).values())
        total = sum(pred_ngrams.values())
        # Add-1 smoothing
        precisions.append((matches + 1) / (total + 1))

    if not precisions:
        return 0.0

    # Geometric mean of precisions
    log_precisions = [math.log(p) for p in precisions]
    return bp * math.exp(sum(log_precisions) / len(log_precisions))


def evaluate_omnidocbench(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    task: str = "e2e_markdown",
) -> dict[str, float]:
    """Evaluate OmniDocBench predictions against ground truth.

    Args:
        predictions: List of prediction dicts with 'example_id' and 'output'
        ground_truths: List of ground truth dicts with 'example_id' and 'target'
        task: Task type for metric selection

    Returns:
        Dictionary of metric scores
    """
    # Build lookup
    gt_map = {gt["example_id"]: gt["target"] for gt in ground_truths}

    scores: dict[str, list[float]] = {
        "edit_similarity": [],
        "bleu": [],
    }

    if task == "table":
        scores["teds"] = []

    for pred in predictions:
        example_id = pred["example_id"]
        pred_text = pred.get("output", "")
        gt_text = gt_map.get(example_id, "")

        scores["edit_similarity"].append(compute_edit_similarity(pred_text, gt_text))
        scores["bleu"].append(compute_bleu(pred_text, gt_text))

        if task == "table":
            scores["teds"].append(compute_teds(pred_text, gt_text))

    # Aggregate
    results = {}
    for metric, values in scores.items():
        if values:
            results[metric] = sum(values) / len(values)
            results[f"{metric}_count"] = len(values)

    return results
