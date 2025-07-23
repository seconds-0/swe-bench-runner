"""Output formatting and display utilities for SWE-bench runner."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import EvaluationResult


def detect_patches_file() -> Optional[Path]:
    """Auto-detect patches file in priority order.

    Looks for predictions.jsonl first, then patches.jsonl.
    Returns the first file found with non-zero size.

    Returns:
        Path to patches file if found, None otherwise
    """
    for name in ["predictions.jsonl", "patches.jsonl"]:
        path = Path(name)
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def display_result(result: EvaluationResult, output_dir: Optional[Path] = None) -> None:
    """Display evaluation result with proper formatting.

    Args:
        result: The evaluation result to display
        output_dir: Optional directory to save results summary
    """
    # Display pass/fail with appropriate emoji
    if result.passed:
        print(f"\n✅ {result.instance_id}: PASSED")
    else:
        print(f"\n❌ {result.instance_id}: FAILED")
        if result.error:
            # Format error message with proper indentation
            error_lines = result.error.strip().split('\n')
            print(f"   Error: {error_lines[0]}")
            for line in error_lines[1:]:
                print(f"          {line}")

    # Save result summary if output directory specified
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "summary.json"

        result_data = {
            "instance_id": result.instance_id,
            "passed": result.passed,
            "error": result.error,
            "timestamp": datetime.now().isoformat(),
            "swe_bench_runner_version": "0.1.0"  # TODO: Get from package version
        }

        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2)

        print(f"\n📁 Results saved to: {output_dir}")

        # Also create a simple pass/fail indicator file for easy checking
        status_file = output_dir / ("PASSED" if result.passed else "FAILED")
        status_file.touch()
