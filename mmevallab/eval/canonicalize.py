"""Output canonicalization for consistent comparison."""

from __future__ import annotations

import re


def canonicalize_mcq(answer: str) -> str:
    """Canonicalize MCQ answer to single letter."""
    answer = answer.strip().upper()
    match = re.search(r"^([A-Z])", answer)
    return match.group(1) if match else answer


def canonicalize_text(text: str) -> str:
    """Canonicalize text output."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def canonicalize_latex(latex: str) -> str:
    """Canonicalize LaTeX formula."""
    latex = latex.strip()
    latex = re.sub(r"^\$+|\$+$", "", latex)
    latex = re.sub(r"^\\\[|\\\]$", "", latex)
    latex = re.sub(r"\s+", " ", latex)
    return latex


def canonicalize_html_table(html: str) -> str:
    """Canonicalize HTML table."""
    html = re.sub(r"\s+", " ", html.strip())
    html = re.sub(r">\s+<", "><", html)
    return html.lower()
