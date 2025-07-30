"""Tracks statistics and progress for SWE-bench evaluations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class EvaluationStats:
    """Statistics for a SWE-bench evaluation run."""

    # Test set information
    test_set_type: str | None = None  # 'lite', 'verified', 'full'
    test_set_size: int | None = None

    # Counts
    selected_count: int = 0
    evaluated_count: int = 0
    missing_count: int = 0
    failed_count: int = 0
    passed_count: int = 0

    # Timing
    start_time: datetime | None = None
    end_time: datetime | None = None

    # Results tracking
    results: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def swe_bench_score(self) -> float:
        """Calculate SWE-bench score as percentage of passed/selected."""
        if self.selected_count == 0:
            return 0.0
        return (self.passed_count / self.selected_count) * 100

    @property
    def duration(self) -> float | None:
        """Calculate duration in seconds."""
        if not self.start_time or not self.end_time:
            return None
        return (self.end_time - self.start_time).total_seconds()

    @property
    def duration_str(self) -> str:
        """Format duration as human-readable string."""
        if not self.duration:
            return "N/A"

        seconds = int(self.duration)
        if seconds < 60:
            return f"{seconds}s"

        minutes = seconds // 60
        seconds = seconds % 60
        if minutes < 60:
            return f"{minutes}m {seconds}s"

        hours = minutes // 60
        minutes = minutes % 60
        return f"{hours}h {minutes}m {seconds}s"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "test_set_type": self.test_set_type,
            "test_set_size": self.test_set_size,
            "selected_count": self.selected_count,
            "evaluated_count": self.evaluated_count,
            "missing_count": self.missing_count,
            "failed_count": self.failed_count,
            "passed_count": self.passed_count,
            "swe_bench_score": round(self.swe_bench_score, 2),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "duration_str": self.duration_str,
            "results": self.results
        }

    def save(self, path: Path) -> None:
        """Save statistics to JSON file."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.to_dict(), indent=2))
        except (OSError, TypeError) as e:
            # OSError for file I/O issues, TypeError for JSON serialization
            print(f"Warning: Failed to save statistics to {path}: {e}")


class EvaluationTracker:
    """Tracks progress and statistics for SWE-bench evaluations."""

    def __init__(self) -> None:
        """Initialize tracker with empty statistics."""
        self.stats = EvaluationStats()

    def set_test_set_info(self, test_set_type: str, test_set_size: int) -> None:
        """Set information about the test set being evaluated."""
        self.stats.test_set_type = test_set_type
        self.stats.test_set_size = test_set_size

    def start_evaluation(self, selected_count: int) -> None:
        """Mark the start of evaluation with number of selected instances."""
        self.stats.selected_count = selected_count
        self.stats.start_time = datetime.now()

    def record_result(
        self, instance_id: str, passed: bool, details: dict[str, Any] | None = None
    ) -> None:
        """Record the result of evaluating an instance."""
        self.stats.evaluated_count += 1

        if passed:
            self.stats.passed_count += 1
        else:
            self.stats.failed_count += 1

        # Store result details
        self.stats.results[instance_id] = {
            "passed": passed,
            "details": details or {}
        }

    def record_missing(self, instance_id: str) -> None:
        """Record that a selected instance had no patch."""
        self.stats.missing_count += 1
        self.stats.results[instance_id] = {
            "passed": False,
            "details": {"status": "missing_patch"}
        }

    def finish_evaluation(self) -> None:
        """Mark the end of evaluation."""
        self.stats.end_time = datetime.now()

    def get_progress(self) -> tuple[int, int, float]:
        """Get current progress as (current, total, percentage)."""
        current = self.stats.evaluated_count + self.stats.missing_count
        total = self.stats.selected_count
        percentage = (current / total * 100) if total > 0 else 0
        return current, total, percentage
