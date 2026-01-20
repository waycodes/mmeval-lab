"""Evaluation runner for executing benchmark evaluations."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from mmevallab.core import (
    benchmark_registry,
    compute_dataset_id,
    compute_run_id,
    get_code_version,
    model_registry,
)


def run_evaluation(
    benchmark_name: str,
    model_name: str,
    split: str = "validation",
    output_dir: Path | str | None = None,
    limit: int | None = None,
    benchmark_kwargs: dict[str, Any] | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run evaluation on a benchmark with a model.

    Args:
        benchmark_name: Name of registered benchmark
        model_name: Name of registered model
        split: Dataset split to evaluate
        output_dir: Directory to write run artifacts (default: runs/{run_id})
        limit: Limit number of examples (for testing)
        benchmark_kwargs: Additional kwargs for benchmark creation
        model_kwargs: Additional kwargs for model creation

    Returns:
        Dict with run_id, metrics, and output paths
    """
    # Import benchmarks and models to trigger registration
    import mmevallab.benchmarks  # noqa: F401
    import mmevallab.models  # noqa: F401

    benchmark_kwargs = benchmark_kwargs or {}
    model_kwargs = model_kwargs or {}

    # Create benchmark and model
    benchmark = benchmark_registry.create(benchmark_name, **benchmark_kwargs)
    model = model_registry.create(model_name, **model_kwargs)

    # Build config for run ID
    config = {
        "benchmark": benchmark_name,
        "model": model_name,
        "split": split,
        "benchmark_kwargs": benchmark_kwargs,
        "model_kwargs": model_kwargs,
    }

    # Load examples
    examples = list(benchmark.load(split, limit=limit))
    num_examples = len(examples)

    # Compute dataset ID (simplified - in production would hash content)
    dataset_id = compute_dataset_id(
        name=benchmark_name,
        version="1.0",
        split=split,
        content_hash=f"{num_examples:08x}",
    )

    # Compute run ID
    run_id = compute_run_id(config, dataset_id)

    # Setup output directory
    if output_dir is None:
        output_dir = Path("runs") / run_id
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run inference and scoring
    predictions: list[dict[str, Any]] = []
    correct = 0
    total = 0

    for example in examples:
        # Generate prediction
        prediction = model.generate(example)

        # Score
        result = benchmark.score(example, prediction)

        # Record
        pred_record = {
            "example_id": example.example_id,
            "raw_output": prediction.raw_output,
            "extracted_answer": prediction.extracted_answer,
            "latency_ms": prediction.latency_ms,
            "is_correct": result.get("is_correct"),
            "ground_truth": example.ground_truth,
            "metadata": example.metadata,
        }
        predictions.append(pred_record)

        if result.get("is_correct") is True:
            correct += 1
        if result.get("is_correct") is not None:
            total += 1

    # Compute metrics
    accuracy = correct / total if total > 0 else 0.0
    metrics = {
        "overall_accuracy": accuracy,
        "correct": correct,
        "total": total,
    }

    # Write artifacts
    # 1. Config
    config_path = output_dir / "config.yaml"
    import yaml

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # 2. Predictions
    predictions_path = output_dir / "predictions.jsonl"
    with open(predictions_path, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")

    # 3. Metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # 4. Metadata
    metadata = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "benchmark": benchmark_name,
        "model": model_name,
        "split": split,
        "num_examples": num_examples,
        "code_version": get_code_version(),
    }
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "metrics": metrics,
        "num_examples": num_examples,
    }
