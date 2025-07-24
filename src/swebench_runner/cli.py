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
# === Primary Options (most common) ===
@click.option(
    "-d", "--dataset",
    type=click.Choice(['lite', 'verified', 'full']),
    help="📊 SWE-bench dataset: 'lite' (300 instances), 'verified' (500), 'full' (2294)"
)
@click.option(
    "--patches",
    type=click.Path(exists=True, path_type=Path),
    help="📄 Path to JSONL file containing patches (alternative to --dataset)",
)
@click.option(
    "--patches-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="📁 Directory with .patch files named {instance_id}.patch",
)
# === Filtering Options ===
@click.option(
    "--instances",
    help="🎯 Specific instance IDs (comma-separated): "
         "'django__django-123,requests__requests-456'"
)
@click.option(
    "--count",
    type=int,
    help="🔢 Number of random instances to run (e.g., --count 10)"
)
@click.option(
    "--sample",
    help="📈 Random percentage sample: '10%', '25%', or 'random-seed=42'"
)
@click.option(
    "--subset",
    help="🔍 Filter by pattern: 'django__*' (glob) or "
         "use with --regex for regex patterns"
)
@click.option(
    "--regex",
    is_flag=True,
    help="⚙️ Treat --subset as regex pattern instead of glob (e.g., 'django__.*-[0-9]+')"
)
# === Advanced Options ===
@click.option(
    "--rerun-failed",
    type=click.Path(exists=True, path_type=Path),
    help="🔄 Rerun only failed instances from previous run directory"
)
@click.option(
    "--offline",
    is_flag=True,
    help="📶 Use cached datasets only (fail if not available locally)",
)
@click.option(
    "--max-patch-size",
    default=5,
    type=int,
    help="📏 Maximum patch size in MB (default: 5)",
)
# === System Options ===
@click.option(
    "--no-input",
    is_flag=True,
    help="🤖 CI mode: fail on prompts instead of waiting for user input",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="📋 Output results as JSON to stdout (for scripting)",
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
    offline: bool,
) -> NoReturn:
    r"""Run SWE-bench evaluation.

    \b
    🚀 Quick Examples:
      swebench run -d lite --count 5        # Test with 5 random instances
      swebench run -d lite --sample 10%     # Use 10% of lite dataset
      swebench run -d verified --subset django__*  # Only Django instances
      swebench run --patches my_patches.jsonl      # Use custom patches

    \b
    📊 Datasets:
      lite     - 300 instances (1.2MB) - Great for testing
      verified - 500 instances (2MB)   - Human-verified fixes
      full     - 2294 instances (8MB)  - Complete benchmark

    \b
    🔍 Filtering:
      --instances: Specific IDs (django__django-123,requests__requests-456)
      --count: Random sample size (--count 10)
      --sample: Percentage (10%, 25%) or with seed (random-seed=42)
      --subset: Pattern matching (django__* or use --regex for advanced)

    \b
    ⚡ Performance:
      --offline: Use cached data only (faster, no network)
      --count 5: Quick test with just 5 instances

    Run 'swebench info -d lite' to see available instances!
    """
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
            "Error: Cannot provide multiple sources "
            "(--patches, --patches-dir, --dataset)",
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

            # Check memory requirements first
            can_proceed, memory_msg = manager.check_memory_requirements(dataset, count)
            if memory_msg and not json_output:
                click.echo(memory_msg)
                if not can_proceed and not no_input:
                    if not click.confirm("⚠️  Continue anyway?"):
                        click.echo("❌ Operation cancelled by user")
                        sys.exit(exit_codes.GENERAL_ERROR)

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
                random_seed=random_seed,
                offline=offline
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
                    f"✅ Loaded {len(dataset_instances)} instances from "
                    f"{dataset} dataset"
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
            from .datasets import get_helpful_error_message

            context = {'dataset': dataset}
            helpful_msg = get_helpful_error_message(e, context)
            click.echo(helpful_msg, err=True)

            # Classify exit code based on error type
            if "Authentication" in str(e) or "401" in str(e):
                sys.exit(exit_codes.NETWORK_ERROR)
            elif "Network" in str(e) or "Connection" in str(e):
                sys.exit(exit_codes.NETWORK_ERROR)
            else:
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
    r"""Interactive setup wizard for first-time users.

    \b
    🎆 What this does:
      • Checks if Docker is running
      • Sets up cache directories
      • Tests basic functionality
      • Provides setup guidance

    \b
    📝 Prerequisites:
      • Docker installed and running
      • Internet connection (for first run)
      • ~1GB free disk space

    Run this before your first evaluation!
    """
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
    help='📊 Dataset to inspect: lite, verified, or full'
)
def info(dataset: str) -> None:
    r"""Get detailed information about SWE-bench datasets.

    \b
    📊 Shows:
      • Total number of instances
      • Download and disk space requirements
      • Dataset description
      • Cache status (if already downloaded)

    \b
    🚀 Examples:
      swebench info -d lite      # Info about lite dataset
      swebench info -d full      # Check full dataset size

    Use this before running large datasets to check space requirements!
    """
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
        from .datasets import get_helpful_error_message

        context = {'dataset': dataset}
        helpful_msg = get_helpful_error_message(e, context)
        click.echo(helpful_msg, err=True)

        # Classify exit code based on error type
        if "Authentication" in str(e) or "401" in str(e):
            sys.exit(exit_codes.NETWORK_ERROR)
        elif "Network" in str(e) or "Connection" in str(e):
            sys.exit(exit_codes.NETWORK_ERROR)
        else:
            sys.exit(exit_codes.GENERAL_ERROR)


if __name__ == "__main__":
    cli()
