"""MMEvalLab CLI entry point."""

from pathlib import Path
from typing import Optional

import click


@click.group()
@click.version_option()
def main() -> None:
    """MMEvalLab: Unified multimodal regression harness."""
    pass


@main.command()
@click.option("--config", "-c", type=click.Path(exists=True), help="Config file path")
@click.option("--benchmark", "-b", type=str, help="Benchmark name")
@click.option("--model", "-m", type=str, help="Model name")
@click.option("--split", "-s", type=str, default="validation", help="Dataset split")
@click.option("--output", "-o", type=click.Path(), help="Output directory")
@click.option("--limit", type=int, help="Limit number of examples")
def run(
    config: Optional[str],
    benchmark: Optional[str],
    model: Optional[str],
    split: str,
    output: Optional[str],
    limit: Optional[int],
) -> None:
    """Run evaluation on a benchmark."""
    from mmevallab.eval.runner import run_evaluation

    if not benchmark or not model:
        click.echo("Error: --benchmark and --model are required", err=True)
        raise SystemExit(1)

    click.echo(f"Running evaluation: benchmark={benchmark}, model={model}, split={split}")

    result = run_evaluation(
        benchmark_name=benchmark,
        model_name=model,
        split=split,
        output_dir=output,
        limit=limit,
    )

    click.echo(f"Run ID: {result['run_id']}")
    click.echo(f"Output: {result['output_dir']}")
    click.echo(f"Examples: {result['num_examples']}")
    click.echo(f"Accuracy: {result['metrics']['overall_accuracy']:.2%}")


@main.command()
@click.argument("run1", type=click.Path(exists=True))
@click.argument("run2", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output path for comparison")
@click.option("--format", "-f", type=click.Choice(["json", "md"]), default="json")
def compare(run1: str, run2: str, output: Optional[str], format: str) -> None:
    """Compare two evaluation runs."""
    from mmevallab.eval.compare import compare_runs, format_comparison_report

    click.echo(f"Comparing: {run1} vs {run2}")

    comparison = compare_runs(run1, run2, output_path=output if format == "json" else None)

    # Print summary
    metrics = comparison["metrics"]
    click.echo(f"Run 1 accuracy: {metrics['run1_accuracy']:.2%}")
    click.echo(f"Run 2 accuracy: {metrics['run2_accuracy']:.2%}")
    click.echo(f"Delta: {metrics['delta_pct']}")

    changes = comparison["example_changes"]
    click.echo(f"Regressions: {changes['correct_to_incorrect']}")
    click.echo(f"Improvements: {changes['incorrect_to_correct']}")
    click.echo(f"Net change: {changes['net_change']:+d}")

    if output and format == "md":
        report = format_comparison_report(comparison)
        Path(output).write_text(report)
        click.echo(f"Report written to: {output}")


@main.command("contam")
@click.option("--benchmark", "-b", type=str, required=True, help="Benchmark to scan")
@click.option("--manifest", "-m", type=click.Path(exists=True), required=True, help="Manifest")
@click.option("--output", "-o", type=click.Path(), help="Output report path")
@click.option("--split", "-s", type=str, default="validation", help="Benchmark split")
@click.option("--limit", type=int, help="Limit examples")
def contamination(
    benchmark: str, manifest: str, output: Optional[str], split: str, limit: Optional[int]
) -> None:
    """Scan for contamination between training data and benchmark."""
    from mmevallab.contamination.scanner import run_contamination_scan

    click.echo(f"Scanning {benchmark} ({split}) against {manifest}")

    report = run_contamination_scan(
        benchmark_name=benchmark,
        manifest_path=manifest,
        output_path=output,
        split=split,
        limit=limit,
    )

    click.echo(f"Total examples: {report['total_examples']}")
    click.echo(f"Exact matches: {report['exact_matches']}")
    click.echo(f"Near matches: {report['near_matches']}")
    click.echo(f"Clean: {report['clean']}")
    click.echo(f"Contamination rate: {report['contamination_rate']:.2%}")

    if output:
        click.echo(f"Report written to: {output}")


@main.command()
@click.argument("run_dir", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output path for slice report")
def slices(run_dir: str, output: Optional[str]) -> None:
    """Analyze slices in a run."""
    click.echo(f"mmeval slices: {run_dir}")
    click.echo("Not yet implemented")


@main.command("attrib")
@click.argument("run1", type=click.Path(exists=True))
@click.argument("run2", type=click.Path(exists=True))
@click.option("--diff", "-d", type=click.Path(exists=True), help="Dataset diff manifest")
@click.option("--output", "-o", type=click.Path(), help="Output path for attribution")
def attribution(
    run1: str, run2: str, diff: Optional[str], output: Optional[str]
) -> None:
    """Attribute slice changes to data differences."""
    click.echo(f"mmeval attrib: {run1} vs {run2}")
    click.echo("Not yet implemented")


@main.command()
@click.argument("run_dir", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), required=True, help="Output archive path")
@click.option("--format", "-f", "fmt", type=click.Choice(["tar.gz", "zip"]), default="tar.gz")
def export(run_dir: str, output: str, fmt: str) -> None:
    """Export run artifacts (license-safe)."""
    from mmevallab.reporting.export import export_artifact_pack

    click.echo(f"Exporting {run_dir} to {output}")
    result = export_artifact_pack(run_dir, output, format=fmt)
    click.echo(f"Created: {result}")


if __name__ == "__main__":
    main()
