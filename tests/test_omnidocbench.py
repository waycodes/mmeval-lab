"""Tests for OmniDocBench schema and normalization."""

import pytest

from mmevallab.benchmarks.omnidoc_schema import (
    BoundingBox,
    OmniDocAnnotation,
    parse_omnidoc_annotation,
    validate_omnidoc_schema,
)
from mmevallab.eval.omnidoc_metrics import compute_edit_similarity, compute_teds


class TestOmniDocSchema:
    """Tests for OmniDocBench schema parsing."""

    def test_parse_minimal_annotation(self) -> None:
        data = {"doc_id": "doc1", "page_num": 1, "pdf_name": "doc1.pdf"}
        annot = parse_omnidoc_annotation(data)
        assert annot.doc_id == "doc1"
        assert annot.page_num == 1
        assert annot.pdf_name == "doc1.pdf"

    def test_parse_full_annotation(self) -> None:
        data = {
            "doc_id": "doc1",
            "page_num": 2,
            "pdf_name": "doc1.pdf",
            "target": "# Title\n\nContent",
            "doc_type": "academic",
            "layout_type": "single_column",
            "language": "en",
            "has_formula": True,
            "has_table": False,
        }
        annot = parse_omnidoc_annotation(data)
        assert annot.doc_type == "academic"
        assert annot.has_formula is True
        assert annot.has_table is False

    def test_parse_with_bbox(self) -> None:
        data = {
            "doc_id": "doc1",
            "page_num": 1,
            "pdf_name": "doc1.pdf",
            "bbox": [10.0, 20.0, 100.0, 50.0],
        }
        annot = parse_omnidoc_annotation(data)
        assert annot.bbox == [10.0, 20.0, 100.0, 50.0]

    def test_invalid_page_num(self) -> None:
        data = {"doc_id": "doc1", "page_num": 0, "pdf_name": "doc1.pdf"}
        with pytest.raises(ValueError):
            parse_omnidoc_annotation(data)

    def test_invalid_bbox(self) -> None:
        data = {"doc_id": "doc1", "page_num": 1, "pdf_name": "doc1.pdf", "bbox": [1, 2, 3]}
        with pytest.raises(ValueError):
            parse_omnidoc_annotation(data)


class TestSchemaValidation:
    """Tests for schema validation."""

    def test_valid_data(self) -> None:
        data = {"doc_id": "doc1", "page_num": 1, "pdf_name": "doc1.pdf"}
        errors = validate_omnidoc_schema(data)
        assert errors == []

    def test_missing_required_fields(self) -> None:
        data = {"doc_id": "doc1"}
        errors = validate_omnidoc_schema(data)
        assert len(errors) == 1
        assert "missing required keys" in errors[0]

    def test_invalid_page_num_type(self) -> None:
        data = {"doc_id": "doc1", "page_num": "one", "pdf_name": "doc1.pdf"}
        errors = validate_omnidoc_schema(data)
        assert any("page_num must be integer" in e for e in errors)

    def test_batch_validation(self) -> None:
        data = [
            {"doc_id": "doc1", "page_num": 1, "pdf_name": "doc1.pdf"},
            {"doc_id": "doc2"},  # Missing fields
        ]
        errors = validate_omnidoc_schema(data)
        assert len(errors) == 1
        assert "Item 1" in errors[0]


class TestBoundingBox:
    """Tests for BoundingBox model."""

    def test_from_list(self) -> None:
        bbox = BoundingBox.from_list([10.0, 20.0, 100.0, 50.0])
        assert bbox.x == 10.0
        assert bbox.y == 20.0
        assert bbox.width == 100.0
        assert bbox.height == 50.0

    def test_from_list_invalid(self) -> None:
        with pytest.raises(ValueError):
            BoundingBox.from_list([1, 2, 3])


class TestNormalization:
    """Tests for output normalization metrics."""

    def test_edit_similarity_identical(self) -> None:
        assert compute_edit_similarity("hello", "hello") == 1.0

    def test_edit_similarity_empty(self) -> None:
        assert compute_edit_similarity("", "") == 1.0

    def test_edit_similarity_partial(self) -> None:
        sim = compute_edit_similarity("hello", "hallo")
        assert 0.7 < sim < 0.9

    def test_teds_identical_tables(self) -> None:
        html = "<table><tr><td>A</td><td>B</td></tr></table>"
        assert compute_teds(html, html) == 1.0

    def test_teds_different_tables(self) -> None:
        html1 = "<table><tr><td>A</td></tr></table>"
        html2 = "<table><tr><td>B</td></tr></table>"
        assert compute_teds(html1, html2) == 0.0

    def test_teds_partial_match(self) -> None:
        html1 = "<table><tr><td>A</td><td>B</td></tr></table>"
        html2 = "<table><tr><td>A</td><td>C</td></tr></table>"
        teds = compute_teds(html1, html2)
        assert 0.4 < teds < 0.6


class TestLatexNormalization:
    """Tests for LaTeX formula normalization."""

    def test_normalize_basic(self) -> None:
        from mmevallab.benchmarks.omnidocbench import _normalize_latex

        assert _normalize_latex("$x^2$") == "x^2"
        assert _normalize_latex("$$y = mx + b$$") == "y = mx + b"

    def test_normalize_whitespace(self) -> None:
        from mmevallab.benchmarks.omnidocbench import _normalize_latex

        assert _normalize_latex("x  +  y") == "x + y"
