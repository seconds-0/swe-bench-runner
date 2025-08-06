"""Cache management utilities for SWE-bench Runner."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional


def get_cache_dir() -> Path:
    """Get the cache directory path, creating it if it doesn't exist."""
    # Check environment variable first
    cache_dir_env = os.environ.get("SWEBENCH_CACHE_DIR")
    if cache_dir_env:
        cache_dir = Path(cache_dir_env)
    else:
        cache_dir = Path.home() / ".swebench"

    # Create cache directory structure
    cache_dir.mkdir(exist_ok=True)
    (cache_dir / "datasets").mkdir(exist_ok=True)
    (cache_dir / "logs").mkdir(exist_ok=True)
    (cache_dir / "results").mkdir(exist_ok=True)
    (cache_dir / "temp").mkdir(exist_ok=True)

    return cache_dir


def is_first_run() -> bool:
    """Check if this is the first time running SWE-bench Runner."""
    cache_dir = get_cache_dir()
    config_file = cache_dir / "config.toml"
    return not config_file.exists()


def mark_first_run_complete() -> None:
    """Create a marker file to indicate first run is complete."""
    cache_dir = get_cache_dir()
    config_file = cache_dir / "config.toml"

    # Create basic config file
    config_content = """# SWE-bench Runner Configuration
# This file is created automatically on first run

[cache]
directory = "{cache_dir}"
created_at = "{timestamp}"

[runner]
version = "0.1.0"
""".format(
        cache_dir=str(cache_dir),
        timestamp=__import__("datetime").datetime.now().isoformat()
    )

    config_file.write_text(config_content)


def get_cache_usage() -> dict[str, int]:
    """Get cache directory usage statistics."""
    cache_dir = get_cache_dir()
    stats = {}

    for subdir in ["datasets", "logs", "results"]:
        subdir_path = cache_dir / subdir
        if subdir_path.exists():
            stats[subdir] = sum(
                f.stat().st_size for f in subdir_path.rglob("*") if f.is_file()
            )
        else:
            stats[subdir] = 0

    return stats


def clean_cache(
    clean_datasets: bool = False,
    clean_logs: bool = False,
    clean_results: bool = False,
    dry_run: bool = False
) -> dict[str, int]:
    """Clean cache directories based on options.

    Args:
        clean_datasets: Remove downloaded datasets
        clean_logs: Remove log files
        clean_results: Remove result files
        dry_run: Show what would be removed without actually removing

    Returns:
        Dictionary with bytes that were (or would be) removed
    """
    cache_dir = get_cache_dir()
    removed_bytes = {"datasets": 0, "logs": 0, "results": 0}

    # Helper function to calculate directory size
    def get_dir_size(path: Path) -> int:
        if not path.exists():
            return 0
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())

    # Clean datasets
    if clean_datasets:
        datasets_dir = cache_dir / "datasets"
        if datasets_dir.exists():
            removed_bytes["datasets"] = get_dir_size(datasets_dir)
            if not dry_run:
                shutil.rmtree(datasets_dir)
                datasets_dir.mkdir(exist_ok=True)

    # Clean logs
    if clean_logs:
        logs_dir = cache_dir / "logs"
        if logs_dir.exists():
            removed_bytes["logs"] = get_dir_size(logs_dir)
            if not dry_run:
                shutil.rmtree(logs_dir)
                logs_dir.mkdir(exist_ok=True)

    # Clean results
    if clean_results:
        results_dir = cache_dir / "results"
        if results_dir.exists():
            removed_bytes["results"] = get_dir_size(results_dir)
            if not dry_run:
                shutil.rmtree(results_dir)
                results_dir.mkdir(exist_ok=True)

    return removed_bytes


def get_results_dir() -> Path:
    """Get the results directory for storing evaluation outputs."""
    cache_dir = get_cache_dir()
    return cache_dir / "results"


def get_logs_dir() -> Path:
    """Get the logs directory for storing evaluation logs."""
    cache_dir = get_cache_dir()
    return cache_dir / "logs"


def get_datasets_dir() -> Path:
    """Get the datasets directory for storing downloaded datasets."""
    cache_dir = get_cache_dir()
    return cache_dir / "datasets"


def auto_detect_patches_file(current_dir: Optional[Path] = None) -> Optional[Path]:
    """Auto-detect patches file in current directory using smart defaults."""
    if current_dir is None:
        current_dir = Path.cwd()

    # Common patch file names in order of preference
    candidates = [
        "predictions.jsonl",
        "patches.jsonl",
        "model_patches.jsonl",
        "patches.json",
        "predictions.json"
    ]

    for candidate in candidates:
        file_path = current_dir / candidate
        if file_path.exists() and file_path.is_file() and file_path.stat().st_size > 0:
            return file_path

    return None
