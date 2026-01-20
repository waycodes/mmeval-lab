"""Contamination scanner for detecting benchmark overlap with training data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mmevallab.contamination.fingerprint import TextFingerprintIndex
from mmevallab.contamination.manifest import load_manifest


def build_training_index(
    manifest_path: Path | str,
    text_field: str = "text",
) -> TextFingerprintIndex:
    """Build fingerprint index from training manifest."""
    index = TextFingerprintIndex()
    for sample in load_manifest(manifest_path):
        text = sample.text or ""
        if text:
            index.add(sample.sample_id, text)
    return index


def scan_benchmark(
    benchmark_examples: list[dict[str, Any]],
    index: TextFingerprintIndex,
    text_field: str = "question",
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Scan benchmark examples against training index.

    Returns:
        Dict with contamination report
    """
    exact_matches = []
    near_matches = []
    clean = []

    for example in benchmark_examples:
        example_id = example.get("example_id", "")
        text = example.get("inputs", {}).get(text_field, "")
        if not text:
            text = example.get(text_field, "")

        # Check exact
        exact = index.find_exact(text)
        if exact:
            exact_matches.append({
                "example_id": example_id,
                "matched_samples": exact,
                "match_type": "exact",
            })
            continue

        # Check near
        near = index.find_near(text, threshold=threshold)
        if near:
            near_matches.append({
                "example_id": example_id,
                "matched_samples": [m[0] for m in near[:5]],
                "similarity": near[0][1] if near else 0,
                "match_type": "near",
            })
        else:
            clean.append(example_id)

    return {
        "total_examples": len(benchmark_examples),
        "exact_matches": len(exact_matches),
        "near_matches": len(near_matches),
        "clean": len(clean),
        "contamination_rate": (len(exact_matches) + len(near_matches)) / len(benchmark_examples)
        if benchmark_examples
        else 0,
        "exact_match_details": exact_matches,
        "near_match_details": near_matches[:100],  # Limit details
    }


def run_contamination_scan(
    benchmark_name: str,
    manifest_path: Path | str,
    output_path: Path | str | None = None,
    split: str = "validation",
    limit: int | None = None,
) -> dict[str, Any]:
    """Run full contamination scan."""
    import mmevallab.benchmarks  # noqa: F401
    from mmevallab.core import benchmark_registry

    # Load benchmark
    benchmark = benchmark_registry.create(benchmark_name)
    examples = [
        {
            "example_id": ex.example_id,
            "inputs": ex.inputs,
            "metadata": ex.metadata,
        }
        for ex in benchmark.load(split, limit=limit)
    ]

    # Build index
    index = build_training_index(manifest_path)

    # Scan
    report = scan_benchmark(examples, index)
    report["benchmark"] = benchmark_name
    report["split"] = split
    report["manifest"] = str(manifest_path)

    # Write output
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

    return report
