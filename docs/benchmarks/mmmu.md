# MMMU Benchmark Contract

## Overview

MMMU (Massive Multi-discipline Multimodal Understanding) evaluates multimodal models on expert-level tasks across 30 subjects and 6 disciplines.

## Dataset

- **Source**: HuggingFace `MMMU/MMMU`
- **Splits**: `dev` (150), `validation` (900), `test` (10,500 - labels withheld)
- **License**: CC BY-NC-SA 4.0

## Input Schema

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | Unique example identifier |
| `question` | str | Question text with `<image N>` placeholders |
| `options` | list[str] | Answer choices (A-D or A-E) |
| `image_1..7` | bytes | Up to 7 images per question |
| `answer` | str | Correct option letter (dev/val only) |

## Metadata Fields

| Field | Values |
|-------|--------|
| `discipline` | Art & Design, Business, Health & Medicine, Humanities & Social Science, Science, Tech & Engineering |
| `subject` | 30 subjects (e.g., Accounting, Biology, Chemistry) |
| `subfield` | Fine-grained topic within subject |
| `image_type` | Diagram, Chart, Photo, Table, etc. |

## Output Schema

```json
{
  "example_id": "validation_Art_and_Design_0",
  "raw_output": "The answer is B because...",
  "extracted_answer": "B",
  "is_correct": true
}
```

## Scoring

- **Primary metric**: Accuracy (exact match on extracted letter)
- **Breakdowns**: By discipline, subject, subfield, image_type
- **Extraction**: Robust MCQ parser handles verbose outputs

## Prompt Template (v1)

```
{question}

Options:
{options}

Answer with the letter of the correct option.
```

## License Constraints

- Non-commercial use only
- Attribution required
- No redistribution of raw images
