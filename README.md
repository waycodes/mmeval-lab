# MMEvalLab

Unified multimodal regression harness for MMMU, OmniDocBench, and Video-MME benchmarks.

## Installation

```bash
# Basic install
uv pip install -e .

# With all extras
uv pip install -e ".[all]"
```

## Extras

- `eval`: Core evaluation dependencies (datasets, pillow, numpy)
- `video`: Video processing (av, opencv-python)
- `pdf`: PDF rendering (pymupdf)
- `faiss`: Vector similarity search
- `dev`: Development tools (pytest, ruff, mypy, pre-commit)

## Usage

```bash
mmeval run --config config.yaml
mmeval compare run1 run2
```
