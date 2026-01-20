# Quickstart Guide

MMEvalLab is a unified multimodal regression harness for MMMU, OmniDocBench, and Video-MME benchmarks.

## Installation

```bash
# Basic install
uv pip install -e .

# With all extras
uv pip install -e ".[all]"
```

## Running Evaluations

### MMMU

```bash
mmeval run --benchmark mmmu --model hf_vlm --split validation --limit 100
```

### OmniDocBench

```bash
mmeval run --benchmark omnidocbench --model hf_vlm --split test --limit 50
```

### Video-MME

```bash
mmeval run --benchmark videomme --model hf_vlm --split test --limit 30
```

## Comparing Runs

```bash
mmeval compare runs/run_a runs/run_b --output comparison.json
```

## Contamination Scanning

```bash
mmeval contam --benchmark mmmu --manifest training_data.jsonl --output contam_report.json
```

## Exporting Results

```bash
mmeval export runs/run_abc123 --output results.zip
```

## Configuration Files

Use YAML configs for reproducible runs:

```yaml
# config.yaml
benchmark: mmmu
model:
  name: hf_vlm
  model_id: llava-hf/llava-1.5-7b-hf
split: validation
limit: 1000
```

```bash
mmeval run --config config.yaml
```

## Output Structure

Each run creates a directory with:
- `config.yaml` - Run configuration
- `predictions.jsonl` - Per-example predictions
- `metrics.json` - Aggregate metrics
- `metadata.json` - Reproducibility metadata
