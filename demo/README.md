# MMEvalLab Executive Demo

A live demonstration of MMEvalLab's capabilities for technical leadership.

## Quick Start

```bash
# From repository root
python demo/run_demo.py
```

## What the Demo Shows

### 1. Unified Evaluation Interface
- Run evaluation on synthetic MMMU-like data
- See metrics computed using actual `mmevallab` modules
- View accuracy breakdown by discipline

### 2. Regression Detection
- Compare baseline vs candidate model
- Identify example-level changes (improved/regressed/unchanged)
- Quantify the impact of model updates

### 3. Slice-Based Failure Analysis
- Find which slices regressed most
- Discover underperforming feature combinations
- Guide targeted debugging and data collection

### 4. Contamination Scanning
- Detect exact text matches between train/test
- Find near-duplicates using MinHash/LSH
- Validate result integrity before publication

### 5. Professional Reporting
- Generate HTML reports for stakeholders
- Include confidence intervals and metadata
- Export artifacts for external sharing

## Demo Flow

The demo is interactive—press Enter to advance between sections. This allows the presenter to explain each capability before showing the next.

## Regenerating Fixtures

```bash
python demo/generate_fixtures.py
```

This creates:
- `fixtures/mmmu_samples.json` — 50 synthetic MMMU-like examples
- `fixtures/training_manifest.jsonl` — 100 training samples (some contaminated)

## Key Talking Points

**For VPs/Directors:**
- "This catches regressions before they ship to production"
- "One tool replaces three separate evaluation codebases"
- "Reports are stakeholder-ready, not just raw numbers"

**For Principal Scientists:**
- "Slice discovery reveals hidden failure patterns"
- "Contamination scanning ensures published results are valid"
- "Full reproducibility metadata for every run"

**For Engineering Leads:**
- "CI gates can block merges on slice regressions"
- "Resume-safe execution handles interruptions"
- "Two-layer caching minimizes redundant computation"
