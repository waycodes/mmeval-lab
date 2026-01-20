# Run Artifact Schema

## Overview

Each evaluation run produces a structured artifact containing metadata, metrics, and per-example outputs.

## Directory Structure

```
runs/{run_id}/
├── config.yaml          # Run configuration
├── metadata.json        # Run metadata + dataset version
├── predictions.jsonl    # Per-example outputs
├── metrics.json         # Aggregated metrics
├── slices.parquet       # Slice annotations (optional)
└── report.html          # Human-readable report (optional)
```

## Schema Components

### RunMetadata

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | str | Deterministic hash of config + dataset + code |
| `created_at` | datetime | Run start timestamp |
| `config` | RunConfig | Full configuration |
| `dataset_version` | DatasetVersion | Dataset identifier + hash |
| `code_version` | str | Git commit hash |
| `python_version` | str | Python version |
| `package_version` | str | mmevallab version |

### DatasetVersion

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Dataset name (e.g., "MMMU") |
| `version` | str | Version string |
| `split` | str | Split name (dev/val/test) |
| `num_examples` | int | Number of examples |
| `content_hash` | str | SHA256 of dataset content |

### ExampleOutput

| Field | Type | Description |
|-------|------|-------------|
| `example_id` | str | Unique example identifier |
| `raw_output` | str | Model's raw text output |
| `extracted_answer` | str? | Parsed answer (e.g., "B") |
| `is_correct` | bool? | Correctness (null if labels withheld) |
| `latency_ms` | float | Inference latency |
| `tokens_in` | int? | Input token count |
| `tokens_out` | int? | Output token count |
| `slices` | list | Slice memberships |
| `metadata` | dict | Additional per-example data |

### SliceMetrics

| Field | Type | Description |
|-------|------|-------------|
| `slice_name` | str | Slice dimension (e.g., "discipline") |
| `slice_value` | str | Slice value (e.g., "Science") |
| `count` | int | Examples in slice |
| `accuracy` | float | Accuracy on slice |
| `metrics` | dict | Additional metrics |

## JSON Schema

See `docs/schemas/run_artifact.schema.json` for the complete JSON Schema.

## Usage

```python
from mmevallab.core.schema import RunArtifact, RunMetadata, ExampleOutput

# Load existing run
artifact = RunArtifact.model_validate_json(path.read_text())

# Create new run
artifact = RunArtifact(
    metadata=RunMetadata(...),
    metrics=RunMetrics(...),
    outputs=[ExampleOutput(...), ...]
)

# Export
path.write_text(artifact.model_dump_json(indent=2))
```
