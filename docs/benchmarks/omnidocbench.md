# OmniDocBench Benchmark Contract

## Overview

OmniDocBench evaluates document understanding across diverse document types including scientific papers, financial reports, textbooks, and forms.

## Dataset

- **Source**: Official OmniDocBench release
- **Format**: PDFs + JSON annotations
- **License**: Research use only (check individual document licenses)

## Input Schema

| Field | Type | Description |
|-------|------|-------------|
| `doc_id` | str | Document identifier |
| `page_num` | int | Page number (1-indexed) |
| `pdf_path` | str | Path to source PDF |
| `task` | str | Task variant (see below) |
| `target` | str | Expected output |

## Task Variants

| Task | Description | Output Format |
|------|-------------|---------------|
| `e2e_markdown` | Full page to markdown | Markdown text |
| `ocr_text` | Text extraction | Plain text |
| `formula` | LaTeX formula extraction | LaTeX string |
| `table` | Table extraction | Markdown/HTML table |
| `layout` | Layout element detection | Bounding boxes |

## Metadata Fields

| Field | Values |
|-------|--------|
| `doc_type` | academic, financial, textbook, form, slide |
| `layout_type` | single_column, multi_column, mixed |
| `language` | en, zh, multi |
| `has_formula` | bool |
| `has_table` | bool |

## Output Schema

```json
{
  "example_id": "doc001_page3_e2e",
  "raw_output": "# Title\n\nParagraph text...",
  "task": "e2e_markdown",
  "latency_ms": 1234
}
```

## Scoring

- **TEDS**: Table Edit Distance Similarity (tables)
- **BLEU/Edit Distance**: Text extraction
- **Exact Match**: Formula extraction (normalized)
- **Breakdowns**: By doc_type, layout_type, language

## Rendering

- Default DPI: 144
- Cache key: `(pdf_hash, page_num, dpi)`
- Format: PNG

## License Constraints

- PDFs must not be committed to repository
- Rendered images are derivative works
- Check per-document licenses before redistribution
