"""CLI interface for SWE-bench Runner."""

from __future__ import annotations

import json
import sys
import uuid
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
from .cache import clean_cache, get_cache_dir, get_cache_usage
from .docker_run import run_evaluation
from .error_utils import classify_error
from .output import detect_patches_file, display_result


def load_failed_instances(results_dir: Path) -> list[str]:
    """Load instance IDs that failed from a previous run."""
    failed_instances = []

    # Check for summary.json files
    for summary_file in results_dir.rglob("summary.json"):
        try:
            with open(summary_file) as f:
                data = json.load(f)
                if not data.get('passed', True):
                    failed_instances.append(data['instance_id'])
        except (json.JSONDecodeError, KeyError):
            # Skip malformed files
            continue

    # Alternative: check for FAILED status files
    if not failed_instances:
        for failed_file in results_dir.rglob("FAILED"):
            instance_id = failed_file.parent.name
            failed_instances.append(instance_id)

    return failed_instances


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
    "-d", "--dataset",
    type=click.Choice(['lite', 'verified', 'full']),
    help="Use SWE-bench dataset (auto-downloads from HuggingFace)"
)
@click.option(
    "--instances",
    help="Comma-separated list of specific instance IDs to run"
)
@click.option(
    "--count",
    type=int,
    help="Number of instances to run (random sample)"
)
@click.option(
    "--sample",
    help="Random percentage of instances (e.g., '10%' or 'random-seed=42')"
)
@click.option(
    "--subset",
    help="Filter instances by pattern (e.g., 'django__*', 'requests__*')"
)
@click.option(
    "--regex",
    is_flag=True,
    help="Treat --subset as regex instead of glob pattern"
)
@click.option(
    "--rerun-failed",
    type=click.Path(exists=True, path_type=Path),
    help="Rerun failed instances from a previous run directory"
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
    dataset: str | None,
    instances: str | None,
    count: int | None,
    sample: str | None,
    subset: str | None,
    regex: bool,
    rerun_failed: Path | None,
    no_input: bool,
    json_output: bool,
    max_patch_size: int,
) -> NoReturn:
    """Run SWE-bench evaluation."""
    # Handle --rerun-failed first
    if rerun_failed:
        failed_instances = load_failed_instances(rerun_failed)
        if not failed_instances:
            if not json_output:
                click.echo("✅ No failed instances to rerun")
            sys.exit(exit_codes.SUCCESS)
        # Convert to instance list
        instances = ','.join(failed_instances)
        if not json_output:
            click.echo(f"🔄 Rerunning {len(failed_instances)} failed instances")

    # Priority: explicit patches file > dataset selection > auto-detection
    if not patches and not patches_dir and not dataset and not rerun_failed:
        # Try auto-detection
        detected = detect_patches_file()
        if detected:
            patches = detected
            if not json_output:
                click.echo(f"💡 Using {patches}")
        elif not no_input:
            # Fall back to interactive suggestion
            suggested_file = suggest_patches_file()
            if suggested_file:
                patches = suggested_file

        if patches is None and patches_dir is None and dataset is None:
            click.echo(
                "Error: Must provide --patches, --patches-dir, or --dataset", err=True
            )
            sys.exit(exit_codes.GENERAL_ERROR)

    # Validate mutual exclusivity
    provided_sources = sum(bool(x) for x in [patches, patches_dir, dataset])
    if provided_sources > 1:
        click.echo(
            "Error: Cannot provide multiple sources (--patches, --patches-dir, --dataset)", 
            err=True
        )
        sys.exit(exit_codes.GENERAL_ERROR)

    # Handle dataset fetching
    if dataset and not patches and not patches_dir:
        try:
            from .datasets import DatasetManager
        except ImportError:
            click.echo(
                "Error: Dataset features require additional dependencies", err=True
            )
            click.echo(
                "Install with: pip install 'swebench-runner[datasets]'", err=True
            )
            sys.exit(exit_codes.GENERAL_ERROR)

        if not json_output:
            click.echo(f"📥 Loading {dataset} dataset from HuggingFace...")

        try:
            manager = DatasetManager(get_cache_dir())

            # Parse --sample flag
            sample_percent = None
            random_seed = None
            if sample:
                if sample.endswith('%'):
                    sample_percent = float(sample[:-1])
                elif '=' in sample:
                    # Handle "random-seed=42" format
                    parts = sample.split('=')
                    if parts[0] == 'random-seed':
                        random_seed = int(parts[1])

            # Parse --instances flag
            instance_list = None
            if instances:
                instance_list = [i.strip() for i in instances.split(',')]

            dataset_instances = manager.get_instances(
                dataset_name=dataset,
                instances=instance_list,
                count=count,
                sample_percent=sample_percent,
                subset_pattern=subset,
                use_regex=regex,
                random_seed=random_seed
            )

            if not dataset_instances:
                click.echo("❌ No instances matched your criteria", err=True)
                sys.exit(exit_codes.GENERAL_ERROR)

            # Clean up old temp files first to ensure directory exists
            manager.cleanup_temp_files()

            # Save to temporary JSONL for compatibility with existing code
            temp_patches = (
                get_cache_dir() / "temp" / f"{dataset}_{uuid.uuid4().hex[:8]}.jsonl"
            )
            manager.save_as_jsonl(dataset_instances, temp_patches)
            patches = temp_patches

            if not json_output:
                click.echo(
                    f"✅ Loaded {len(dataset_instances)} instances from {dataset} dataset"
                )
                if instance_list:
                    instance_preview = ', '.join(instance_list[:3])
                    suffix = '...' if len(instance_list) > 3 else ''
                    click.echo(f"   Specific instances: {instance_preview}{suffix}")
                if sample_percent:
                    click.echo(f"   Random {sample_percent}% sample")
                if count:
                    click.echo(f"   Limited to {count} instances")
                if subset:
                    filter_type = 'regex' if regex else 'pattern'
                    click.echo(f"   Filtered by {filter_type}: {subset}")

        except Exception as e:
            if "401" in str(e):
                click.echo("❌ Authentication required for this dataset", err=True)
                click.echo(
                    "   Set HF_TOKEN environment variable with your HuggingFace token",
                    err=True
                )
                click.echo(
                    "   Get a token at: https://huggingface.co/settings/tokens", err=True
                )
            elif "Connection" in str(e) or "Network" in str(e):
                click.echo("❌ Network error downloading dataset", err=True)
                click.echo("   Check your internet connection", err=True)
            else:
                click.echo(f"❌ Failed to load dataset: {e}", err=True)
            sys.exit(exit_codes.NETWORK_ERROR)

    # Validate patches file if provided
    if patches is not None:
        if patches.stat().st_size == 0:
            click.echo("Error: Patches file is empty", err=True)
            sys.exit(exit_codes.GENERAL_ERROR)

        if not patches.is_file():
            click.echo(f"Error: {patches} is not a file", err=True)
            sys.exit(exit_codes.GENERAL_ERROR)

        patch_source = str(patches)
    elif patches_dir is not None:
        # patches_dir is provided
        patch_files = list(patches_dir.glob("*.patch"))
        if not patch_files:
            click.echo(f"Error: No .patch files found in {patches_dir}", err=True)
            sys.exit(exit_codes.GENERAL_ERROR)

        patch_source = str(patches_dir)
    else:
        # This shouldn't happen due to validation above
        click.echo("Error: No patch source specified", err=True)
        sys.exit(exit_codes.GENERAL_ERROR)

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
            # Use our enhanced display function
            output_dir = Path(f"results/{result.instance_id}")
            display_result(result, output_dir)

            # Show success celebration if it passed
            if result.passed and is_first_run:
                show_success_message(result.instance_id, is_first_success=True)

        # Map errors to appropriate exit codes using shared utility
        if result.error:
            exit_code = classify_error(result.error)
            sys.exit(exit_code)
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


@cli.command()
@click.option(
    '-d', '--dataset',
    type=click.Choice(['lite', 'verified', 'full']),
    required=True,
    help='Dataset to get information about'
)
def info(dataset: str) -> None:
    """Get information about a SWE-bench dataset."""
    try:
        from .datasets import DatasetManager
    except ImportError:
        click.echo("Error: Dataset features require additional dependencies", err=True)
        click.echo("Install with: pip install 'swebench-runner[datasets]'", err=True)
        sys.exit(exit_codes.GENERAL_ERROR)

    try:
        manager = DatasetManager(get_cache_dir())
        dataset_info = manager.get_dataset_info(dataset)

        click.echo(f"\n📊 SWE-bench {dataset} dataset:")
        click.echo(f"   Total instances: {dataset_info['total_instances']:,}")
        click.echo(f"   Download size: {dataset_info['download_size_mb']:.1f} MB")
        click.echo(f"   On-disk size: {dataset_info['dataset_size_mb']:.1f} MB")
        if dataset_info['description']:
            click.echo(f"   Description: {dataset_info['description'][:100]}...")
    except Exception as e:
        if "401" in str(e):
            click.echo("❌ Authentication required for this dataset", err=True)
            click.echo(
                "   Set HF_TOKEN environment variable with your HuggingFace token",
                err=True
            )
            click.echo(
                "   Get a token at: https://huggingface.co/settings/tokens", err=True
            )
        elif "Connection" in str(e) or "Network" in str(e):
            click.echo("❌ Network error accessing dataset", err=True)
            click.echo("   Check your internet connection", err=True)
        else:
            click.echo(f"❌ Failed to get dataset info: {e}", err=True)
        sys.exit(exit_codes.NETWORK_ERROR)


if __name__ == "__main__":
    cli()
