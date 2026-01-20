"""MMEvalLab CLI entry point."""

import click


@click.group()
@click.version_option()
def main() -> None:
    """MMEvalLab: Unified multimodal regression harness."""
    pass


@main.command()
def run() -> None:
    """Run evaluation on a benchmark."""
    click.echo("mmeval run: not yet implemented")


@main.command()
def compare() -> None:
    """Compare two evaluation runs."""
    click.echo("mmeval compare: not yet implemented")


if __name__ == "__main__":
    main()
