"""CLI interface for SWE-bench Runner."""

import sys
from pathlib import Path
from typing import NoReturn

import click

from . import __version__


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

    # MVP: Just show what would be run
    click.echo(f"Would run evaluation with {patches}")

    # Success exit code
    sys.exit(0)


if __name__ == "__main__":
    cli()
