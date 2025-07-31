"""Output formatting and display utilities for SWE-bench runner."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from . import __version__
from .models import EvaluationResult

if TYPE_CHECKING:
    from .evaluation_tracker import EvaluationStats


def detect_patches_file() -> Path | None:
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


def display_result(result: EvaluationResult, output_dir: Path | None = None) -> None:
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
            "swe_bench_runner_version": __version__
        }

        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2)

        print(f"\n📁 Results saved to: {output_dir}")

        # Also create a simple pass/fail indicator file for easy checking
        status_file = output_dir / ("PASSED" if result.passed else "FAILED")
        status_file.touch()


def display_evaluation_summary(stats: EvaluationStats) -> None:
    """Display a formatted summary of evaluation results.

    Args:
        stats: Evaluation statistics to display
    """
    print("\n" + "=" * 60)
    print("🎯 EVALUATION SUMMARY")
    print("=" * 60)

    # Test set information
    if stats.test_set_type and stats.test_set_size:
        print(f"📊 Test Set: SWE-bench {stats.test_set_type} "
              f"({stats.test_set_size} instances)")

    # Statistics
    print("\n📈 Statistics:")
    print(f"   • Selected: {stats.selected_count}")
    print(f"   • Evaluated: {stats.evaluated_count}")
    print(f"   • Failed: {stats.failed_count}")
    print(f"   • Passed: {stats.passed_count}")

    # SWE-bench score
    print(f"\n🏆 SWE-bench Score: {stats.swe_bench_score:.1f}%")
    print(f"   ({stats.passed_count}/{stats.selected_count} instances resolved)")

    # Timing
    if stats.start_time and stats.end_time:
        print("\n⏱️  Timing:")
        print(f"   Start: {stats.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   End:   {stats.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Duration: {stats.duration_str}")

    print("=" * 60)
