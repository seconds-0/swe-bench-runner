"""CLI interface for SWE-bench Runner."""

import sys
from pathlib import Path
from typing import NoReturn

import click

from . import __version__
from .docker_run import run_evaluation


@click.group()
@click.version_option(version=__version__)
def cli() -> None:
    """SWE-bench evaluation runner."""
    pass


@cli.command()
@click.option(
    "--patches",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to JSONL file containing patches",
)
def run(patches: Path) -> NoReturn:
    """Run SWE-bench evaluation."""
    # Validate that the patches file is not empty
    if patches.stat().st_size == 0:
        click.echo("Error: Patches file is empty", err=True)
        sys.exit(1)

    # Basic validation that it's a file (not a directory)
    if not patches.is_file():
        click.echo(f"Error: {patches} is not a file", err=True)
        sys.exit(1)

    # Run the evaluation
    try:
        result = run_evaluation(str(patches))
        if result.passed:
            click.echo(f"✅ {result.instance_id}: PASSED")
            sys.exit(0)
        else:
            click.echo(f"❌ {result.instance_id}: FAILED")
            if result.error:
                click.echo(f"   Error: {result.error}")
            sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
