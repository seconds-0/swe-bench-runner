"""CLI interface for SWE-bench Runner."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import NoReturn

import click

from . import __version__, exit_codes
from .bootstrap import (
    check_and_prompt_first_run,
    show_docker_setup_help,
    show_success_message,
    suggest_patches_file,
)
from .cache import clean_cache, get_cache_usage
from .docker_run import run_evaluation


@click.group()
@click.version_option(version=__version__)
def cli() -> None:
    """SWE-bench evaluation runner."""
    pass


@cli.command()
@click.option(
    "--patches",
    type=click.Path(exists=True, path_type=Path),
    help="Path to JSONL file containing patches",
)
@click.option(
    "--patches-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Directory containing .patch files named {instance_id}.patch",
)
@click.option(
    "--no-input",
    is_flag=True,
    help="CI mode: fail on prompts instead of waiting for user input",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Output results as JSON to stdout",
)
@click.option(
    "--max-patch-size",
    default=5,
    type=int,
    help="Maximum patch size in MB (default: 5)",
)
def run(
    patches: Path | None,
    patches_dir: Path | None,
    no_input: bool,
    json_output: bool,
    max_patch_size: int,
) -> NoReturn:
    """Run SWE-bench evaluation."""
    # Auto-detect patches file if none provided
    if patches is None and patches_dir is None:
        if not no_input:
            suggested_file = suggest_patches_file()
            if suggested_file:
                patches = suggested_file

        if patches is None and patches_dir is None:
            click.echo(
                "Error: Must provide either --patches or --patches-dir", err=True
            )
            sys.exit(exit_codes.GENERAL_ERROR)

    if patches is not None and patches_dir is not None:
        click.echo("Error: Cannot provide both --patches and --patches-dir", err=True)
        sys.exit(exit_codes.GENERAL_ERROR)

    # Validate patches file if provided
    if patches is not None:
        if patches.stat().st_size == 0:
            click.echo("Error: Patches file is empty", err=True)
            sys.exit(exit_codes.GENERAL_ERROR)

        if not patches.is_file():
            click.echo(f"Error: {patches} is not a file", err=True)
            sys.exit(exit_codes.GENERAL_ERROR)

        patch_source = str(patches)
    else:
        # patches_dir is provided
        assert patches_dir is not None
        patch_files = list(patches_dir.glob("*.patch"))
        if not patch_files:
            click.echo(f"Error: No .patch files found in {patches_dir}", err=True)
            sys.exit(exit_codes.GENERAL_ERROR)

        patch_source = str(patches_dir)

    # Check for first-time setup after argument validation
    is_first_run = check_and_prompt_first_run(no_input=no_input)

    # Run the evaluation
    try:
        result = run_evaluation(
            patch_source, no_input=no_input, max_patch_size_mb=max_patch_size
        )

        if json_output:
            # Output JSON to stdout
            output = {
                "instance_id": result.instance_id,
                "passed": result.passed,
                "error": result.error
            }
            click.echo(json.dumps(output))
        else:
            # Regular output
            if result.passed:
                show_success_message(result.instance_id, is_first_success=is_first_run)
            else:
                click.echo(f"❌ {result.instance_id}: FAILED")
                if result.error:
                    click.echo(f"   Error: {result.error}")

        # Map errors to appropriate exit codes
        if result.error:
            error_lower = result.error.lower()

            # Check for specific error types and map to exit codes
            # Check network errors first (includes "failed to pull")
            if any(term in error_lower for term in [
                "network", "connection", "unreachable", "registry", "pull",
                "resolve", "dns", "connection refused", "failed to pull",
                "pull access denied"
            ]):
                sys.exit(exit_codes.NETWORK_ERROR)
            elif any(term in error_lower for term in ["timeout", "timed out"]):
                sys.exit(exit_codes.GENERAL_ERROR)  # Timeouts are general errors
            elif "docker" in error_lower and any(term in error_lower for term in [
                "not found", "not running", "daemon"
            ]):
                sys.exit(exit_codes.DOCKER_NOT_FOUND)
            elif any(term in error_lower for term in [
                "disk", "space", "memory", "ram"
            ]):
                sys.exit(exit_codes.RESOURCE_ERROR)
            else:
                sys.exit(exit_codes.GENERAL_ERROR)
        else:
            sys.exit(exit_codes.SUCCESS if result.passed else exit_codes.GENERAL_ERROR)
    except Exception as e:
        if json_output:
            error_output = {
                "error": str(e),
                "passed": False
            }
            click.echo(json.dumps(error_output))
        else:
            click.echo(f"Error: {e}", err=True)
        sys.exit(exit_codes.GENERAL_ERROR)


@cli.command()
@click.option(
    "--datasets",
    is_flag=True,
    help="Clean downloaded datasets",
)
@click.option(
    "--logs",
    is_flag=True,
    help="Clean log files",
)
@click.option(
    "--results",
    is_flag=True,
    help="Clean result files",
)
@click.option(
    "--all",
    "clean_all",
    is_flag=True,
    help="Clean all cache directories",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be removed without actually removing",
)
def clean(
    datasets: bool, logs: bool, results: bool, clean_all: bool, dry_run: bool
) -> None:
    """Clean cache directories to free up disk space."""
    # If no specific options given, show usage
    if not (datasets or logs or results or clean_all):
        # Show current cache usage
        usage = get_cache_usage()
        click.echo("Cache usage:")
        for category, size_bytes in usage.items():
            size_mb = size_bytes / (1024 * 1024)
            click.echo(f"  {category}: {size_mb:.1f} MB")

        total_mb = sum(usage.values()) / (1024 * 1024)
        click.echo(f"  Total: {total_mb:.1f} MB")
        click.echo()
        click.echo(
            "Use --datasets, --logs, --results, or --all to clean specific areas"
        )
        click.echo("Add --dry-run to see what would be removed")
        return

    # Set flags based on options
    if clean_all:
        datasets = logs = results = True

    # Perform cleaning
    removed = clean_cache(
        clean_datasets=datasets,
        clean_logs=logs,
        clean_results=results,
        dry_run=dry_run
    )

    # Show results
    action = "Would remove" if dry_run else "Removed"
    total_removed = 0
    for category, size_bytes in removed.items():
        if size_bytes > 0:
            size_mb = size_bytes / (1024 * 1024)
            click.echo(f"{action} {size_mb:.1f} MB from {category}")
            total_removed += size_bytes

    if total_removed > 0:
        total_mb = total_removed / (1024 * 1024)
        click.echo(f"{action} {total_mb:.1f} MB total")
    else:
        click.echo("No files to remove")


@cli.command()
def setup() -> None:
    """Interactive setup wizard for first-time users."""
    click.echo("🔧 SWE-bench Runner Setup")
    click.echo()

    # Check if Docker is available
    try:
        import docker
        client = docker.from_env()
        client.ping()
        click.echo("✅ Docker is running")
    except Exception:
        click.echo("❌ Docker not available")
        show_docker_setup_help()
        return

    # Check first run status
    if check_and_prompt_first_run(no_input=False):
        click.echo("✅ First-time setup completed")
    else:
        click.echo("✅ SWE-bench Runner is already set up")

    click.echo()
    click.echo("You're ready to run evaluations!")
    click.echo("Try: swebench run --patches your_patches.jsonl")


if __name__ == "__main__":
    cli()
