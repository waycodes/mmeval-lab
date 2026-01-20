# MMEvalLab

**A unified regression harness for multimodal AI evaluation across MMMU, OmniDocBench, and Video-MME benchmarks.**

[![CI](https://github.com/waycodes/mmeval-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/waycodes/mmeval-lab/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

---

## The Problem

Evaluating vision-language models (VLMs) is fragmented and error-prone:

- **Three benchmarks, three codebases** — MMMU, OmniDocBench, and Video-MME each have separate evaluation scripts with incompatible interfaces
- **No regression tracking** — Teams ship model updates without knowing which capabilities improved or degraded
- **Hidden data contamination** — Training data overlap with test sets inflates metrics silently
- **Irreproducible results** — Missing metadata makes it impossible to debug why two runs differ

## The Solution

MMEvalLab provides a single, production-grade harness that:

| Capability | What it does |
|------------|--------------|
| **Unified Interface** | One CLI for all three benchmarks with consistent output formats |
| **Regression Detection** | Compare runs to find exactly which examples flipped correct→wrong |
| **Slice Analysis** | Automatically discover underperforming subgroups (e.g., "math + diagrams") |
| **Contamination Scanning** | Detect training data overlap via text fingerprinting and perceptual hashing |
| **Full Reproducibility** | Every run captures git commit, package versions, and prompt template hashes |

---

## Quick Start

### Installation

```bash
# Clone and install
git clone https://github.com/waycodes/mmeval-lab.git
cd mmeval-lab
uv pip install -e ".[all]"
```

### Run Your First Evaluation

```bash
# Evaluate on MMMU with a HuggingFace model
mmeval run --benchmark mmmu --model hf_vlm --split validation --limit 100

# Compare two runs to find regressions
mmeval compare runs/run_abc123 runs/run_def456 --output comparison.json

# Scan for training data contamination
mmeval contam --benchmark mmmu --manifest training_data.jsonl
```

---

## Example Workflows

### 1. Pre-Release Regression Check

Before shipping a model update, verify no capabilities regressed:

```bash
# Run baseline
mmeval run --benchmark mmmu --model hf_vlm --split validation --output runs/baseline

# Run candidate
mmeval run --benchmark mmmu --model hf_vlm --split validation --output runs/candidate \
  --model-kwargs '{"model_id": "my-org/new-model"}'

# Compare and generate report
mmeval compare runs/baseline runs/candidate --html report.html
```

The comparison report shows:
- Overall accuracy delta with confidence intervals
- Per-slice regressions ranked by severity
- Example-level diffs (which questions flipped from correct to wrong)

### 2. Slice-Based Debugging

Find where your model struggles:

```python
from mmevallab.slicing import discover_slices, rank_worst_slices

# Automatically discover underperforming feature combinations
slices = discover_slices(predictions, ["discipline", "image_type", "difficulty"])

# Output: [("discipline:science + image_type:diagram", accuracy=0.42, n=87), ...]
```

### 3. Contamination Audit

Before publishing results, verify test set integrity:

```bash
mmeval contam --benchmark mmmu --manifest training_data.jsonl --output contam_report.json
```

Detects:
- **Exact matches** — identical question text
- **Near-duplicates** — MinHash/LSH similarity > 0.9
- **Image matches** — PDQ perceptual hash collisions

---

## Supported Benchmarks

| Benchmark | Domain | Tasks | Examples |
|-----------|--------|-------|----------|
| [MMMU](https://mmmu-benchmark.github.io/) | Academic knowledge | MCQ across 30 subjects | 11.5K |
| [OmniDocBench](https://github.com/opendatalab/OmniDocBench) | Document understanding | OCR, table extraction, layout | 981 |
| [Video-MME](https://video-mme.github.io/) | Video comprehension | MCQ with temporal reasoning | 2.7K |

---

## Architecture

```
mmevallab/
├── benchmarks/      # Benchmark adapters (MMMU, OmniDocBench, Video-MME)
├── models/          # Model backends (HuggingFace, OpenAI, vLLM)
├── eval/            # Scoring, metrics, regression gates
├── slicing/         # Slice definitions and discovery
├── contamination/   # Fingerprinting and deduplication
├── attribution/     # Data diff and contamination risk linking
└── reporting/       # HTML reports and artifact export
```

---

## Documentation

- **[Quickstart Guide](docs/quickstart.md)** — Get running in 5 minutes
- **[Adding Benchmarks & Models](docs/adding_components.md)** — Extend MMEvalLab with custom components

---

## Key Features

### For Engineers

- **Resume-safe execution** — Interrupted runs pick up where they left off
- **Two-layer caching** — Memory + disk cache for expensive model calls
- **Distributed sharding** — Split evaluation across multiple GPUs/nodes

### For Researchers

- **Bootstrap confidence intervals** — Know if your improvement is statistically significant
- **Prompt template versioning** — Track exactly which prompt produced which results
- **Failure mode taxonomy** — Automatic labeling of error types (refusal, hallucination, format error)

### For Teams

- **CI regression gates** — Block merges if slice accuracy drops below threshold
- **HTML reports** — Shareable dashboards for stakeholder review
- **Artifact export** — License-safe redaction for external sharing

---

## Installation Options

```bash
# Minimal (core evaluation)
uv pip install -e ".[eval]"

# With video support
uv pip install -e ".[eval,video]"

# Full development setup
uv pip install -e ".[all]"
```

| Extra | Dependencies | Use Case |
|-------|--------------|----------|
| `eval` | datasets, pillow, numpy | Core evaluation |
| `video` | av, opencv-python | Video-MME benchmark |
| `pdf` | pymupdf | OmniDocBench PDF rendering |
| `faiss` | faiss-cpu | Similarity search for contamination |
| `dev` | pytest, ruff, mypy | Development and testing |

---

## Why MMEvalLab?

| Without MMEvalLab | With MMEvalLab |
|-------------------|----------------|
| Copy-paste evaluation scripts between projects | Single `mmeval run` command |
| "It worked on my machine" | Full reproducibility metadata |
| Ship regressions unknowingly | CI gates catch slice degradations |
| Manual spreadsheet comparisons | Automated HTML diff reports |
| Hope training data doesn't overlap | Systematic contamination scanning |

---

## Contributing

We welcome contributions! See our [development setup](docs/quickstart.md) to get started.

```bash
# Run tests
pytest tests/ -v

# Run linter
ruff check mmevallab/

# Type check
mypy mmevallab/ --ignore-missing-imports
```

---

## License

Apache 2.0 — See [LICENSE](LICENSE) for details.

---

<p align="center">
  <i>Built for teams who ship multimodal AI with confidence.</i>
</p>
