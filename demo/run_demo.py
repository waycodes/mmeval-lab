#!/usr/bin/env python3
"""
MMEvalLab Executive Demo
========================

A live demonstration of MMEvalLab's capabilities for technical leadership.

This demo showcases:
1. Unified evaluation across benchmarks
2. Regression detection between model versions
3. Slice-based failure analysis
4. Contamination scanning
5. Professional reporting

Run: python demo/run_demo.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()

DEMO_DIR = Path(__file__).parent
FIXTURES = DEMO_DIR / "fixtures"
OUTPUTS = DEMO_DIR / "outputs"

# Non-interactive mode for CI/testing
INTERACTIVE = "--no-pause" not in sys.argv


def pause(message: str = "Press Enter to continue...") -> None:
    """Pause for presenter to explain."""
    if not INTERACTIVE:
        return
    console.print(f"\n[dim]{message}[/dim]")
    input()


def section(title: str) -> None:
    """Print section header."""
    console.print()
    console.rule(f"[bold blue]{title}[/bold blue]")
    console.print()


def demo_intro() -> None:
    """Introduction and problem statement."""
    console.print(Panel.fit(
        "[bold]MMEvalLab[/bold]\n"
        "[dim]Unified Multimodal Regression Harness[/dim]\n\n"
        "A production-grade evaluation framework for vision-language models\n"
        "across MMMU, OmniDocBench, and Video-MME benchmarks.",
        title="🔬 Executive Demo",
        border_style="blue",
    ))

    pause()

    section("The Problem We're Solving")

    problems = Table(show_header=False, box=None, padding=(0, 2))
    problems.add_column(style="red")
    problems.add_column()
    problems.add_row("❌", "Three benchmarks = three incompatible evaluation scripts")
    problems.add_row("❌", "No systematic regression tracking between model versions")
    problems.add_row("❌", "Hidden training data contamination inflates metrics")
    problems.add_row("❌", "Results are irreproducible across teams")
    console.print(problems)

    pause()


def demo_evaluation() -> None:
    """Demonstrate unified evaluation."""
    section("1. Unified Evaluation Interface")

    console.print("Running evaluation on synthetic MMMU data...\n")

    # Load fixtures
    with open(FIXTURES / "mmmu_samples.json") as f:
        samples = json.load(f)

    # Import and use actual mmevallab modules
    from mmevallab.eval.mmmu_metrics import compute_mmmu_metrics

    # Simulate evaluation with progress
    predictions = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Evaluating examples...", total=len(samples))

        for sample in samples:
            # Simulate model prediction (deterministic for demo)
            is_correct = sample["is_correct_baseline"]
            predictions.append({
                "example_id": sample["id"],
                "extracted_answer": sample["answer"] if is_correct else "X",
                "ground_truth": sample["answer"],
                "is_correct": is_correct,
                "metadata": sample["metadata"],
            })
            progress.advance(task)
            time.sleep(0.02)  # Visual effect

    # Compute metrics using actual module
    metrics = compute_mmmu_metrics(predictions)

    # Display results
    results = Table(title="Baseline Evaluation Results")
    results.add_column("Metric", style="cyan")
    results.add_column("Value", style="green")
    results.add_row("Overall Accuracy", f"{metrics['overall_accuracy']:.1%}")
    results.add_row("Correct / Total", f"{metrics['correct']} / {metrics['total']}")
    console.print(results)

    # Show discipline breakdown
    console.print("\n[bold]Accuracy by Discipline:[/bold]")
    disc_table = Table()
    disc_table.add_column("Discipline")
    disc_table.add_column("Accuracy", justify="right")
    disc_table.add_column("Count", justify="right")

    for disc, data in sorted(metrics["by_discipline"].items()):
        disc_table.add_row(disc, f"{data['accuracy']:.1%}", str(data["total"]))
    console.print(disc_table)

    # Save for later comparison
    OUTPUTS.mkdir(exist_ok=True)
    with open(OUTPUTS / "baseline_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    pause()
    return predictions, metrics


def demo_regression(baseline_preds: list) -> None:
    """Demonstrate regression detection."""
    section("2. Regression Detection")

    console.print("Now let's compare against a [bold]candidate model[/bold] update...\n")

    # Load candidate predictions (has regressions)
    with open(FIXTURES / "mmmu_samples.json") as f:
        samples = json.load(f)

    from mmevallab.eval.example_diff import compute_example_diffs, summarize_diffs
    from mmevallab.eval.mmmu_metrics import compute_mmmu_metrics

    # Build candidate predictions (with intentional regressions)
    candidate_preds = []
    for sample in samples:
        is_correct = sample["is_correct_candidate"]
        candidate_preds.append({
            "example_id": sample["id"],
            "extracted_answer": sample["answer"] if is_correct else "X",
            "ground_truth": sample["answer"],
            "is_correct": is_correct,
            "metadata": sample["metadata"],
        })

    # Compute metrics
    baseline_metrics = compute_mmmu_metrics(baseline_preds)
    candidate_metrics = compute_mmmu_metrics(candidate_preds)

    # Show comparison
    delta = candidate_metrics["overall_accuracy"] - baseline_metrics["overall_accuracy"]
    delta_style = "red" if delta < 0 else "green"

    compare = Table(title="Model Comparison")
    compare.add_column("", style="bold")
    compare.add_column("Baseline", justify="right")
    compare.add_column("Candidate", justify="right")
    compare.add_column("Delta", justify="right")
    compare.add_row(
        "Accuracy",
        f"{baseline_metrics['overall_accuracy']:.1%}",
        f"{candidate_metrics['overall_accuracy']:.1%}",
        f"[{delta_style}]{delta:+.1%}[/{delta_style}]",
    )
    console.print(compare)

    pause("Let's dig deeper into what changed...")

    # Example-level diffs
    diffs = compute_example_diffs(baseline_preds, candidate_preds)
    summary = summarize_diffs(diffs)

    console.print("\n[bold]Example-Level Changes:[/bold]")
    changes = Table()
    changes.add_column("Change Type")
    changes.add_column("Count", justify="right")
    for change_type, count in sorted(summary.items()):
        if change_type == "regressed":
            changes.add_row(f"[red]{change_type}[/red]", str(count))
        elif change_type == "improved":
            changes.add_row(f"[green]{change_type}[/green]", str(count))
        else:
            changes.add_row(change_type, str(count))
    console.print(changes)

    # Show specific regressions
    regressed = [d for d in diffs if d.change_type == "regressed"]
    if regressed:
        console.print(f"\n[bold red]⚠️  Found {len(regressed)} regressions![/bold red]")
        console.print("\nSample regressions:")
        for d in regressed[:3]:
            console.print(f"  • {d.example_id}: was correct, now wrong")

    pause()
    return baseline_metrics, candidate_metrics, candidate_preds


def demo_slice_analysis(
    baseline_metrics: dict, candidate_metrics: dict, candidate_preds: list
) -> None:
    """Demonstrate slice-based debugging."""
    section("3. Slice-Based Failure Analysis")

    console.print("Where exactly is the model failing? Let's analyze by slice...\n")

    from mmevallab.slicing.ranking import rank_regressions

    # Find slice regressions
    regressions = rank_regressions(
        baseline_metrics["by_discipline"],
        candidate_metrics["by_discipline"],
        metric_key="accuracy",
        top_k=5,
    )

    if regressions:
        console.print("[bold]Slice Regressions (by discipline):[/bold]")
        reg_table = Table()
        reg_table.add_column("Slice")
        reg_table.add_column("Baseline", justify="right")
        reg_table.add_column("Candidate", justify="right")
        reg_table.add_column("Delta", justify="right")

        for name, base, curr, delta in regressions:
            reg_table.add_row(
                name,
                f"{base:.1%}",
                f"{curr:.1%}",
                f"[red]{delta:+.1%}[/red]",
            )
        console.print(reg_table)

    pause("Let's discover more granular failure patterns...")

    # Automated slice discovery
    from mmevallab.slicing.discovery import discover_slices

    discovered = discover_slices(
        candidate_preds,
        feature_fields=["discipline", "image_type"],
        min_count=3,
        max_conjunction=2,
        top_k=5,
    )

    if discovered:
        console.print("\n[bold]Discovered Underperforming Slices:[/bold]")
        disc_table = Table()
        disc_table.add_column("Feature Combination")
        disc_table.add_column("Accuracy", justify="right")
        disc_table.add_column("Count", justify="right")
        disc_table.add_column("vs Overall", justify="right")

        for s in discovered:
            features_str = " + ".join(f"{k}={v}" for k, v in s.features)
            disc_table.add_row(
                features_str,
                f"{s.accuracy:.1%}",
                str(s.count),
                f"[red]{s.delta_from_overall:+.1%}[/red]",
            )
        console.print(disc_table)

        console.print("\n[yellow]💡 Insight:[/yellow] Model struggles with specific combinations.")
        console.print("   This guides targeted data collection and fine-tuning.")

    pause()


def demo_contamination() -> None:
    """Demonstrate contamination scanning."""
    section("4. Contamination Scanning")

    console.print("Before publishing results, we must verify test set integrity...\n")

    from mmevallab.contamination.fingerprint import TextFingerprintIndex
    from mmevallab.contamination.minhash import LSHIndex, MinHash

    # Load test questions
    with open(FIXTURES / "mmmu_samples.json") as f:
        test_samples = json.load(f)

    # Load training manifest (some contaminated)
    with open(FIXTURES / "training_manifest.jsonl") as f:
        train_samples = [json.loads(line) for line in f]

    n_test, n_train = len(test_samples), len(train_samples)
    console.print(f"Scanning {n_test} test examples against {n_train} training samples...\n")

    # Build fingerprint index
    fp_index = TextFingerprintIndex()
    for sample in train_samples:
        fp_index.add(sample["id"], sample["text"])

    # Scan for exact matches
    exact_matches = []
    for test in test_samples:
        matches = fp_index.find_exact(test["question"])
        if matches:
            exact_matches.append((test["id"], matches[0]))

    # Build LSH index for near-duplicates
    lsh = LSHIndex(num_bands=8, rows_per_band=8)
    train_hashes = {}
    for sample in train_samples:
        mh = MinHash(num_hashes=64).compute(sample["text"])
        train_hashes[sample["id"]] = mh
        lsh.add(sample["id"], mh)

    # Scan for near-duplicates
    near_dups = []
    for test in test_samples:
        mh = MinHash(num_hashes=64).compute(test["question"])
        matches = lsh.query(mh, threshold=0.7)
        if matches:
            near_dups.append((test["id"], matches[0][0], matches[0][1]))

    # Report findings
    results = Table(title="Contamination Scan Results")
    results.add_column("Check", style="cyan")
    results.add_column("Found", justify="right")
    results.add_column("Status")

    exact_status = "[red]⚠️  CONTAMINATED[/red]" if exact_matches else "[green]✓ Clean[/green]"
    near_status = "[yellow]⚠️  Review needed[/yellow]" if near_dups else "[green]✓ Clean[/green]"

    results.add_row("Exact text matches", str(len(exact_matches)), exact_status)
    results.add_row("Near-duplicates (>70%)", str(len(near_dups)), near_status)
    console.print(results)

    if exact_matches:
        console.print("\n[bold red]Exact matches found:[/bold red]")
        for test_id, train_id in exact_matches[:3]:
            console.print(f"  • Test {test_id} ↔ Train {train_id}")

    if near_dups:
        console.print("\n[bold yellow]Near-duplicates found:[/bold yellow]")
        for test_id, train_id, sim in near_dups[:3]:
            console.print(f"  • Test {test_id} ↔ Train {train_id} ({sim:.0%} similar)")

    pause()


def demo_reporting() -> None:
    """Demonstrate report generation."""
    section("5. Professional Reporting")

    console.print("Generating stakeholder-ready HTML report...\n")

    from mmevallab.reporting.html_report import generate_run_report

    # Load metrics
    with open(OUTPUTS / "baseline_metrics.json") as f:
        metrics = json.load(f)

    # Generate single-run report
    report_path = OUTPUTS / "evaluation_report.html"
    generate_run_report(
        run_id="demo_baseline_v1",
        metrics=metrics,
        breakdown=metrics["by_discipline"],
        output_path=report_path,
    )

    console.print(f"[green]✓[/green] Generated: {report_path}")
    console.print("  Open in browser to view interactive dashboard\n")

    # Show what's in the report
    features = Table(title="Report Contents")
    features.add_column("Section")
    features.add_column("Description")
    features.add_row("Overall Metrics", "Accuracy with confidence intervals")
    features.add_row("Category Breakdown", "Performance by discipline/subject")
    features.add_row("Worst Slices", "Underperforming subgroups ranked")
    features.add_row("Metadata", "Git commit, timestamps, reproducibility info")
    console.print(features)

    pause()


def demo_conclusion() -> None:
    """Wrap up and key takeaways."""
    section("Summary")

    takeaways = Table(show_header=False, box=None, padding=(0, 2))
    takeaways.add_column(style="green")
    takeaways.add_column()
    takeaways.add_row("✓", "Unified interface across MMMU, OmniDocBench, Video-MME")
    takeaways.add_row("✓", "Automatic regression detection with example-level diffs")
    takeaways.add_row("✓", "Slice discovery reveals hidden failure patterns")
    takeaways.add_row("✓", "Contamination scanning ensures result integrity")
    takeaways.add_row("✓", "Professional reports for stakeholder communication")
    console.print(takeaways)

    console.print("\n[bold]Impact:[/bold]")
    console.print("  • Catch regressions [bold]before[/bold] they ship to production")
    console.print("  • Debug failures [bold]systematically[/bold] instead of guessing")
    console.print("  • Publish results with [bold]confidence[/bold] in their integrity")

    console.print()
    console.print(Panel.fit(
        "[bold]Questions?[/bold]\n\n"
        "Documentation: docs/quickstart.md\n"
        "Source: github.com/waycodes/mmeval-lab",
        border_style="blue",
    ))


def main() -> None:
    """Run the full demo."""
    console.clear()

    try:
        demo_intro()
        baseline_preds, baseline_metrics = demo_evaluation()
        baseline_metrics, candidate_metrics, candidate_preds = demo_regression(baseline_preds)
        demo_slice_analysis(baseline_metrics, candidate_metrics, candidate_preds)
        demo_contamination()
        demo_reporting()
        demo_conclusion()

    except KeyboardInterrupt:
        console.print("\n[dim]Demo interrupted.[/dim]")
        sys.exit(0)


if __name__ == "__main__":
    main()
