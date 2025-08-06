"""Batch processing for SWE-bench instances with checkpointing and progress tracking."""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from swebench_runner.generation.patch_generator import GenerationResult, PatchGenerator
from swebench_runner.providers.exceptions import (
    ProviderError,
    ProviderRateLimitError,
    ProviderTokenLimitError,
)

logger = logging.getLogger(__name__)


@dataclass
class CheckpointData:
    """Checkpoint data for resume capability."""

    timestamp: datetime
    completed: set[str]
    failed: dict  # instance_id -> error_reason
    stats: dict
    total_instances: int
    start_time: datetime

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "completed": list(self.completed),
            "failed": self.failed,
            "stats": self.stats,
            "total_instances": self.total_instances,
            "start_time": self.start_time.isoformat()
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'CheckpointData':
        """Create from dictionary loaded from JSON."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            completed=set(data["completed"]),
            failed=data["failed"],
            stats=data["stats"],
            total_instances=data["total_instances"],
            start_time=datetime.fromisoformat(data["start_time"])
        )


@dataclass
class BatchStats:
    """Statistics for batch processing."""

    total_instances: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    total_cost: float = 0.0
    total_time: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    success_rate: float = 0.0

    def update_success_rate(self) -> None:
        """Update success rate calculation."""
        if self.completed + self.failed > 0:
            self.success_rate = self.completed / (self.completed + self.failed)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_instances": self.total_instances,
            "completed": self.completed,
            "failed": self.failed,
            "skipped": self.skipped,
            "total_cost": self.total_cost,
            "total_time": self.total_time,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "success_rate": self.success_rate
        }


@dataclass
class BatchResult:
    """Result of batch processing."""

    successful: list = field(default_factory=list)
    failed: list = field(default_factory=list)  # instance + error info
    skipped: list = field(default_factory=list)  # instance_ids
    stats: BatchStats = field(default_factory=BatchStats)
    checkpoint_saved: bool = False


class ProgressTracker:
    """Progress tracking for batch processing."""

    def __init__(self, total: int, show_bar: bool = True):
        self.total = total
        self.completed = 0
        self.failed = 0
        self.skipped = 0
        self.start_time = time.time()
        self.show_bar = show_bar

    def update(self, success: bool = True, skipped: bool = False) -> None:
        """Update progress."""
        if skipped:
            self.skipped += 1
        elif success:
            self.completed += 1
        else:
            self.failed += 1

    def get_eta(self) -> Optional[float]:
        """Get estimated time to completion."""
        if self.completed == 0:
            return None

        elapsed = time.time() - self.start_time
        rate = self.completed / elapsed
        remaining = self.total - (self.completed + self.failed + self.skipped)

        if rate <= 0:
            return None

        return remaining / rate

    def format_progress(self) -> str:
        """Format progress string."""
        processed = self.completed + self.failed + self.skipped
        percent = (processed / self.total * 100) if self.total > 0 else 0

        eta_str = ""
        eta = self.get_eta()
        if eta is not None:
            eta_mins = int(eta / 60)
            eta_secs = int(eta % 60)
            eta_str = f", ETA: {eta_mins:02d}:{eta_secs:02d}"

        return (f"Progress: {processed}/{self.total} ({percent:.1f}%) - "
                f"✓{self.completed} ✗{self.failed} ⏭{self.skipped}{eta_str}")


class BatchProcessor:
    """Processes batches of SWE-bench instances with checkpointing and recovery."""

    def __init__(
        self,
        generator: PatchGenerator,
        checkpoint_dir: Optional[Path] = None,
        max_concurrent: int = 5,
        retry_failed: bool = True,
        max_retries: int = 2,
        progress_bar: bool = True,
        save_intermediate: bool = True,
        checkpoint_interval: int = 10  # Save checkpoint every N completions
    ):
        """Initialize the batch processor.

        Args:
            generator: PatchGenerator instance for generating patches
            checkpoint_dir: Directory to save checkpoints (None disables checkpointing)
            max_concurrent: Maximum number of concurrent processing tasks
            retry_failed: Whether to retry failed instances on resume
            max_retries: Maximum number of retry attempts for failed instances
            progress_bar: Whether to show progress updates
            save_intermediate: Whether to save intermediate checkpoints
            checkpoint_interval: How often to save checkpoints (number of completions)
        """
        self.generator = generator
        self.checkpoint_dir = checkpoint_dir
        self.max_concurrent = max_concurrent
        self.retry_failed = retry_failed
        self.max_retries = max_retries
        self.progress_bar = progress_bar
        self.save_intermediate = save_intermediate
        self.checkpoint_interval = checkpoint_interval

        # Ensure checkpoint directory exists
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Internal state
        self._completion_count = 0
        self._last_checkpoint_save = 0

    async def process_batch(
        self,
        instances: list,
        resume_from_checkpoint: bool = True,
        save_final_checkpoint: bool = True
    ) -> BatchResult:
        """Process batch of instances with checkpointing and recovery.

        Args:
            instances: List of SWE-bench instance data
            resume_from_checkpoint: Whether to resume from existing checkpoint
            save_final_checkpoint: Whether to save final checkpoint

        Returns:
            BatchResult with processed results and statistics
        """
        logger.info(f"Starting batch processing for {len(instances)} instances")

        # Initialize statistics
        stats = BatchStats(
            total_instances=len(instances),
            start_time=datetime.now()
        )

        # Load checkpoint if resuming
        checkpoint_data = None
        completed_ids: set[str] = set()
        failed_attempts: dict = {}

        if resume_from_checkpoint and self.checkpoint_dir:
            checkpoint_data = self.load_checkpoint()
            if checkpoint_data:
                completed_ids = checkpoint_data.completed.copy()
                # Track retry attempts for failed instances
                for instance_id in checkpoint_data.failed.keys():
                    failed_attempts[instance_id] = (
                        failed_attempts.get(instance_id, 0) + 1
                    )

                # Restore stats from checkpoint
                stats.completed = len(completed_ids)
                stats.failed = len(checkpoint_data.failed)
                stats.total_cost = checkpoint_data.stats.get("total_cost", 0.0)

                logger.info(f"Resuming from checkpoint: {stats.completed} completed, "
                          f"{stats.failed} failed, {stats.total_cost:.4f} cost")

        # Filter instances based on checkpoint
        instances_to_process = []
        skipped_ids = []

        for instance in instances:
            instance_id = instance.get("instance_id", "unknown")

            # Skip already completed instances
            if instance_id in completed_ids:
                skipped_ids.append(instance_id)
                continue

            # Skip failed instances if not retrying or max retries exceeded
            if instance_id in failed_attempts:
                if (not self.retry_failed or
                    failed_attempts[instance_id] >= self.max_retries):
                    skipped_ids.append(instance_id)
                    continue

            instances_to_process.append(instance)

        stats.skipped = len(skipped_ids)
        logger.info(f"Processing {len(instances_to_process)} instances "
                   f"({stats.skipped} skipped from checkpoint)")

        # Initialize progress tracking
        progress = self.create_progress_tracker(len(instances_to_process))

        # Process instances with controlled concurrency
        semaphore = asyncio.Semaphore(self.max_concurrent)
        results = []
        failed_results = []

        # Create tasks for concurrent processing
        async def process_and_track(instance: dict) -> Optional[dict]:
            """Process instance and update progress."""
            instance_id = instance.get("instance_id", "unknown")

            try:
                result_data = await self._process_single_instance(
                    instance, semaphore, failed_attempts.get(instance_id, 0) + 1
                )

                # Update progress and stats
                if result_data["success"]:
                    progress.update(success=True)
                    stats.completed += 1
                    stats.total_cost += result_data["cost"]
                    results.append(result_data["result"])
                else:
                    progress.update(success=False)
                    stats.failed += 1
                    failed_results.append({
                        "instance": instance,
                        "error": result_data["error"],
                        "attempts": result_data["attempts"]
                    })

                # Save checkpoint periodically
                self._completion_count += 1
                if (self.save_intermediate and self.checkpoint_dir and
                    self._completion_count - self._last_checkpoint_save >=
                    self.checkpoint_interval):
                    await self._save_intermediate_checkpoint(
                        instances, results, failed_results, skipped_ids, stats
                    )
                    self._last_checkpoint_save = self._completion_count

                # Log progress
                if self.progress_bar:
                    logger.info(progress.format_progress())

                return result_data

            except Exception as e:
                logger.exception(f"Unexpected error processing {instance_id}: {e}")
                progress.update(success=False)
                stats.failed += 1
                failed_results.append({
                    "instance": instance,
                    "error": f"Unexpected error: {str(e)}",
                    "attempts": 1
                })
                return None

        # Execute all processing tasks
        if instances_to_process:
            tasks = [process_and_track(instance) for instance in instances_to_process]
            await asyncio.gather(*tasks, return_exceptions=True)

        # Finalize statistics
        stats.end_time = datetime.now()
        if stats.end_time and stats.start_time:
            stats.total_time = (stats.end_time - stats.start_time).total_seconds()
        stats.update_success_rate()

        # Create final result
        result = BatchResult(
            successful=results,
            failed=failed_results,
            skipped=skipped_ids,
            stats=stats
        )

        # Save final checkpoint if requested
        if save_final_checkpoint and self.checkpoint_dir:
            try:
                self._save_final_checkpoint(result)
                result.checkpoint_saved = True
            except Exception as e:
                logger.warning(f"Failed to save final checkpoint: {e}")

        logger.info(f"Batch processing complete: {stats.completed} successful, "
                   f"{stats.failed} failed, {stats.skipped} skipped, "
                   f"${stats.total_cost:.4f} total cost")

        return result

    async def _process_single_instance(
        self,
        instance: dict,
        semaphore: asyncio.Semaphore,
        attempt: int = 1
    ) -> dict:
        """Process a single instance with error handling and retry logic."""
        async with semaphore:

            try:
                result = await self.generator.generate_patch(instance)

                return {
                    "success": result.success,
                    "result": result,
                    "cost": result.cost,
                    "error": result.error,
                    "attempts": attempt
                }

            except Exception as e:
                error_msg = await self._handle_instance_error(instance, e, attempt)
                return {
                    "success": False,
                    "result": None,
                    "cost": 0.0,
                    "error": error_msg,
                    "attempts": attempt
                }

    async def _handle_instance_error(
        self,
        instance: dict,
        error: Exception,
        attempt: int
    ) -> str:
        """Handle error from instance processing."""
        instance_id = instance.get("instance_id", "unknown")
        error_type = self._classify_error(error)

        logger.error(
            f"Error processing {instance_id} (attempt {attempt}): "
            f"{error_type} - {error}"
        )

        # Handle rate limiting with backoff
        if isinstance(error, ProviderRateLimitError):
            backoff_time = min(60 * attempt, 300)  # Max 5 minutes
            logger.info(
                f"Rate limited, waiting {backoff_time} seconds before continuing..."
            )
            await asyncio.sleep(backoff_time)

        return f"{error_type}: {str(error)}"

    def _should_retry(self, error: Exception, attempt: int) -> bool:
        """Determine if error should trigger retry."""
        if attempt >= self.max_retries:
            return False

        # Retry on transient errors
        if isinstance(error, ProviderRateLimitError | ProviderTokenLimitError):
            return True

        # Don't retry on permanent errors
        if isinstance(error, ProviderError):
            return False

        # Retry on unexpected errors (once)
        return attempt == 1

    def _classify_error(self, error: Exception) -> str:
        """Classify error for reporting."""
        if isinstance(error, ProviderRateLimitError):
            return "rate_limit"
        elif isinstance(error, ProviderTokenLimitError):
            return "token_limit"
        elif isinstance(error, ProviderError):
            return "provider_error"
        else:
            return "unexpected_error"

    def load_checkpoint(self) -> Optional[CheckpointData]:
        """Load checkpoint data if exists."""
        if not self.checkpoint_dir:
            return None

        checkpoint_path = self.get_checkpoint_path()
        if not checkpoint_path.exists():
            return None

        try:
            with open(checkpoint_path) as f:
                data = json.load(f)
            return CheckpointData.from_dict(data)
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return None

    def save_checkpoint(self, checkpoint_data: CheckpointData) -> bool:
        """Save checkpoint data to disk."""
        if not self.checkpoint_dir:
            return False

        checkpoint_path = self.get_checkpoint_path()

        try:
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data.to_dict(), f, indent=2)
            logger.debug(f"Checkpoint saved to {checkpoint_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False

    def _create_checkpoint_data(
        self,
        completed: set[str],
        failed: dict,
        stats: BatchStats,
        total_instances: int,
        start_time: datetime
    ) -> CheckpointData:
        """Create checkpoint data structure."""
        return CheckpointData(
            timestamp=datetime.now(),
            completed=completed,
            failed=failed,
            stats=stats.to_dict(),
            total_instances=total_instances,
            start_time=start_time
        )

    async def _save_intermediate_checkpoint(
        self,
        instances: list,
        results: list,
        failed_results: list,
        skipped_ids: list,
        stats: BatchStats
    ) -> None:
        """Save intermediate checkpoint during processing."""
        if not self.checkpoint_dir:
            return

        completed_ids = {r.instance_id for r in results}
        failed_dict = {
            f["instance"]["instance_id"]: f["error"]
            for f in failed_results
        }

        checkpoint_data = self._create_checkpoint_data(
            completed_ids,
            failed_dict,
            stats,
            len(instances),
            stats.start_time or datetime.now()
        )

        self.save_checkpoint(checkpoint_data)

    def _save_final_checkpoint(self, result: BatchResult) -> None:
        """Save final checkpoint with complete results."""
        if not self.checkpoint_dir:
            return

        completed_ids = {r.instance_id for r in result.successful}
        failed_dict = {
            f["instance"]["instance_id"]: f["error"]
            for f in result.failed
        }

        checkpoint_data = self._create_checkpoint_data(
            completed_ids,
            failed_dict,
            result.stats,
            result.stats.total_instances,
            result.stats.start_time or datetime.now()
        )

        self.save_checkpoint(checkpoint_data)

    def clear_checkpoint(self) -> None:
        """Remove existing checkpoint file."""
        if not self.checkpoint_dir:
            return

        checkpoint_path = self.get_checkpoint_path()
        if checkpoint_path.exists():
            try:
                checkpoint_path.unlink()
                logger.info("Checkpoint cleared")
            except Exception as e:
                logger.warning(f"Failed to clear checkpoint: {e}")

    def create_progress_tracker(self, total: int) -> ProgressTracker:
        """Create progress tracking interface."""
        return ProgressTracker(total, self.progress_bar)

    def generate_report(self, result: BatchResult) -> str:
        """Generate detailed summary report."""
        stats = result.stats

        # Calculate additional metrics
        avg_cost_per_instance = stats.total_cost / max(stats.completed, 1)
        avg_time_per_instance = (
            stats.total_time / max(stats.completed + stats.failed, 1)
        )

        report_lines = [
            "=== Batch Processing Report ===",
            "",
            "Summary:",
            f"  Total instances: {stats.total_instances}",
            f"  Successful: {stats.completed} ({stats.success_rate:.1%})",
            f"  Failed: {stats.failed}",
            f"  Skipped: {stats.skipped}",
            "",
            "Performance:",
            f"  Total time: {stats.total_time:.1f} seconds",
            f"  Average time per instance: {avg_time_per_instance:.1f} seconds",
            f"  Total cost: ${stats.total_cost:.4f}",
            f"  Average cost per instance: ${avg_cost_per_instance:.4f}",
            "",
            self.format_stats_table(stats)
        ]

        if result.failed:
            report_lines.extend([
                "",
                "Failed Instances:",
                "  " + "\n  ".join([
                    f"{f['instance']['instance_id']}: {f['error'][:100]}"
                    for f in result.failed[:10]  # Show first 10
                ])
            ])

            if len(result.failed) > 10:
                report_lines.append(f"  ... and {len(result.failed) - 10} more")

        return "\n".join(report_lines)

    def format_stats_table(self, stats: BatchStats) -> str:
        """Format statistics as table."""
        return f"""Statistics Table:
┌─────────────────────┬─────────────┐
│ Metric              │ Value       │
├─────────────────────┼─────────────┤
│ Success Rate        │ {stats.success_rate:.1%}        │
│ Completed           │ {stats.completed:11d} │
│ Failed              │ {stats.failed:11d} │
│ Skipped             │ {stats.skipped:11d} │
│ Total Cost          │ ${stats.total_cost:10.4f} │
│ Total Time          │ {stats.total_time:8.1f}s   │
└─────────────────────┴─────────────┘"""

    def save_results(self, result: BatchResult, output_file: Path) -> None:
        """Save results to file."""
        try:
            # Convert results to serializable format
            data = {
                "successful": [
                    {
                        "instance_id": r.instance_id,
                        "model": r.model,
                        "attempts": r.attempts,
                        "cost": r.cost,
                        "success": r.success,
                        "truncated": r.truncated,
                        "patch": r.patch,
                        "error": r.error,
                        "metadata": r.metadata
                    }
                    for r in result.successful
                ],
                "failed": result.failed,
                "skipped": result.skipped,
                "stats": result.stats.to_dict(),
                "checkpoint_saved": result.checkpoint_saved
            }

            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

    def load_results(self, result_file: Path) -> BatchResult:
        """Load previously saved results."""
        with open(result_file) as f:
            data = json.load(f)

        # Reconstruct GenerationResult objects
        successful = []
        for r_data in data["successful"]:
            result = GenerationResult(
                patch=r_data["patch"],
                instance_id=r_data["instance_id"],
                model=r_data["model"],
                attempts=r_data["attempts"],
                truncated=r_data["truncated"],
                cost=r_data["cost"],
                success=r_data["success"],
                error=r_data["error"],
                metadata=r_data["metadata"]
            )
            successful.append(result)

        # Reconstruct stats
        stats_data = data["stats"]
        stats = BatchStats(
            total_instances=stats_data["total_instances"],
            completed=stats_data["completed"],
            failed=stats_data["failed"],
            skipped=stats_data["skipped"],
            total_cost=stats_data["total_cost"],
            total_time=stats_data["total_time"],
            success_rate=stats_data["success_rate"]
        )

        if stats_data.get("start_time"):
            stats.start_time = datetime.fromisoformat(stats_data["start_time"])
        if stats_data.get("end_time"):
            stats.end_time = datetime.fromisoformat(stats_data["end_time"])

        return BatchResult(
            successful=successful,
            failed=data["failed"],
            skipped=data["skipped"],
            stats=stats,
            checkpoint_saved=data.get("checkpoint_saved", False)
        )

    def estimate_batch_time(self, num_instances: int) -> float:
        """Estimate total processing time based on concurrency."""
        # Very rough estimate: 30 seconds per instance average
        # Adjusted for concurrency
        time_per_instance = 30.0
        return (num_instances * time_per_instance) / self.max_concurrent

    def estimate_batch_cost(self, instances: list) -> float:
        """Estimate total cost for batch."""
        # Very rough estimate: $0.01 per instance average
        # This would need to be based on actual model pricing
        return len(instances) * 0.01

    def get_checkpoint_path(self) -> Path:
        """Get path to checkpoint file."""
        if not self.checkpoint_dir:
            raise ValueError("No checkpoint directory configured")
        return self.checkpoint_dir / "batch_checkpoint.json"
