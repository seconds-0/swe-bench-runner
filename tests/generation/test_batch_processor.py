"""Tests for BatchProcessor."""

import asyncio
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from swebench_runner.generation.batch_processor import (
    BatchProcessor,
    BatchResult,
    BatchStats,
    CheckpointData,
    ProgressTracker,
)
from swebench_runner.generation.patch_generator import GenerationResult, PatchGenerator
from swebench_runner.providers.exceptions import (
    ProviderRateLimitError,
    ProviderTokenLimitError,
)


@pytest.fixture
def sample_instances():
    """Sample SWE-bench instances for testing."""
    return [
        {
            "instance_id": "test-1",
            "repo": "test/repo",
            "problem_statement": "Fix bug 1"
        },
        {
            "instance_id": "test-2",
            "repo": "test/repo",
            "problem_statement": "Fix bug 2"
        },
        {
            "instance_id": "test-3",
            "repo": "test/repo",
            "problem_statement": "Fix bug 3"
        }
    ]


@pytest.fixture
def mock_generator():
    """Mock PatchGenerator."""
    generator = MagicMock(spec=PatchGenerator)
    generator.generate_patch = AsyncMock()
    return generator


@pytest.fixture
def temp_checkpoint_dir():
    """Temporary directory for checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestProgressTracker:
    """Test ProgressTracker functionality."""

    def test_init(self):
        """Test ProgressTracker initialization."""
        tracker = ProgressTracker(100, show_bar=True)
        assert tracker.total == 100
        assert tracker.completed == 0
        assert tracker.failed == 0
        assert tracker.skipped == 0
        assert tracker.show_bar is True

    def test_update_success(self):
        """Test updating progress with success."""
        tracker = ProgressTracker(10)
        tracker.update(success=True)

        assert tracker.completed == 1
        assert tracker.failed == 0
        assert tracker.skipped == 0

    def test_update_failure(self):
        """Test updating progress with failure."""
        tracker = ProgressTracker(10)
        tracker.update(success=False)

        assert tracker.completed == 0
        assert tracker.failed == 1
        assert tracker.skipped == 0

    def test_update_skipped(self):
        """Test updating progress with skipped."""
        tracker = ProgressTracker(10)
        tracker.update(skipped=True)

        assert tracker.completed == 0
        assert tracker.failed == 0
        assert tracker.skipped == 1

    def test_get_eta_no_progress(self):
        """Test ETA calculation with no progress."""
        tracker = ProgressTracker(10)
        assert tracker.get_eta() is None

    def test_get_eta_with_progress(self):
        """Test ETA calculation with progress."""
        tracker = ProgressTracker(10)
        tracker.completed = 5
        tracker.start_time = tracker.start_time - 10  # 10 seconds ago

        eta = tracker.get_eta()
        assert eta is not None
        assert eta > 0

    def test_format_progress(self):
        """Test progress formatting."""
        tracker = ProgressTracker(10)
        tracker.completed = 3
        tracker.failed = 1
        tracker.skipped = 1

        progress_str = tracker.format_progress()
        assert "5/10 (50.0%)" in progress_str
        assert "✓3" in progress_str
        assert "✗1" in progress_str
        assert "⏭1" in progress_str


class TestCheckpointData:
    """Test CheckpointData serialization."""

    def test_to_dict(self):
        """Test converting to dictionary."""
        now = datetime.now()
        data = CheckpointData(
            timestamp=now,
            completed={"test-1", "test-2"},
            failed={"test-3": "error"},
            stats={"total_cost": 1.5},
            total_instances=3,
            start_time=now
        )

        result = data.to_dict()
        assert result["timestamp"] == now.isoformat()
        assert set(result["completed"]) == {"test-1", "test-2"}
        assert result["failed"] == {"test-3": "error"}
        assert result["stats"] == {"total_cost": 1.5}

    def test_from_dict(self):
        """Test creating from dictionary."""
        now = datetime.now()
        data_dict = {
            "timestamp": now.isoformat(),
            "completed": ["test-1", "test-2"],
            "failed": {"test-3": "error"},
            "stats": {"total_cost": 1.5},
            "total_instances": 3,
            "start_time": now.isoformat()
        }

        data = CheckpointData.from_dict(data_dict)
        assert data.timestamp == now
        assert data.completed == {"test-1", "test-2"}
        assert data.failed == {"test-3": "error"}
        assert data.stats == {"total_cost": 1.5}


class TestBatchStats:
    """Test BatchStats functionality."""

    def test_init(self):
        """Test BatchStats initialization."""
        stats = BatchStats()
        assert stats.total_instances == 0
        assert stats.completed == 0
        assert stats.failed == 0
        assert stats.skipped == 0
        assert stats.success_rate == 0.0

    def test_update_success_rate(self):
        """Test success rate calculation."""
        stats = BatchStats()
        stats.completed = 8
        stats.failed = 2
        stats.update_success_rate()

        assert stats.success_rate == 0.8

    def test_update_success_rate_no_attempts(self):
        """Test success rate with no attempts."""
        stats = BatchStats()
        stats.update_success_rate()

        assert stats.success_rate == 0.0

    def test_to_dict(self):
        """Test converting to dictionary."""
        now = datetime.now()
        stats = BatchStats(
            total_instances=10,
            completed=8,
            failed=2,
            total_cost=5.0,
            start_time=now
        )

        result = stats.to_dict()
        assert result["total_instances"] == 10
        assert result["completed"] == 8
        assert result["failed"] == 2
        assert result["total_cost"] == 5.0
        assert result["start_time"] == now.isoformat()


class TestBatchProcessor:
    """Test BatchProcessor functionality."""

    def test_init_default(self, mock_generator):
        """Test BatchProcessor initialization with defaults."""
        processor = BatchProcessor(mock_generator)

        assert processor.generator == mock_generator
        assert processor.checkpoint_dir is None
        assert processor.max_concurrent == 5
        assert processor.retry_failed is True
        assert processor.max_retries == 2
        assert processor.progress_bar is True

    def test_init_with_checkpoint_dir(self, mock_generator, temp_checkpoint_dir):
        """Test initialization with checkpoint directory."""
        processor = BatchProcessor(
            mock_generator,
            checkpoint_dir=temp_checkpoint_dir,
            max_concurrent=3
        )

        assert processor.checkpoint_dir == temp_checkpoint_dir
        assert processor.max_concurrent == 3
        assert temp_checkpoint_dir.exists()

    def test_create_progress_tracker(self, mock_generator):
        """Test creating progress tracker."""
        processor = BatchProcessor(mock_generator)
        tracker = processor.create_progress_tracker(10)

        assert isinstance(tracker, ProgressTracker)
        assert tracker.total == 10

    def test_classify_error(self, mock_generator):
        """Test error classification."""
        processor = BatchProcessor(mock_generator)

        assert processor._classify_error(
            ProviderRateLimitError("rate limit")
        ) == "rate_limit"
        assert processor._classify_error(
            ProviderTokenLimitError("token limit")
        ) == "token_limit"
        assert processor._classify_error(
            ValueError("other error")
        ) == "unexpected_error"

    def test_should_retry(self, mock_generator):
        """Test retry logic."""
        processor = BatchProcessor(mock_generator, max_retries=2)

        # Should retry rate limit errors
        assert processor._should_retry(ProviderRateLimitError("rate limit"), 1) is True
        assert processor._should_retry(ProviderRateLimitError("rate limit"), 2) is False

        # Should retry token limit errors
        assert processor._should_retry(
            ProviderTokenLimitError("token limit"), 1
        ) is True

        # Should retry unexpected errors once
        assert processor._should_retry(ValueError("error"), 1) is True
        assert processor._should_retry(ValueError("error"), 2) is False

    def test_get_checkpoint_path(self, mock_generator, temp_checkpoint_dir):
        """Test getting checkpoint path."""
        processor = BatchProcessor(mock_generator, checkpoint_dir=temp_checkpoint_dir)
        path = processor.get_checkpoint_path()

        assert path == temp_checkpoint_dir / "batch_checkpoint.json"

    def test_get_checkpoint_path_no_dir(self, mock_generator):
        """Test getting checkpoint path without directory."""
        processor = BatchProcessor(mock_generator)

        with pytest.raises(ValueError, match="No checkpoint directory configured"):
            processor.get_checkpoint_path()

    @pytest.mark.asyncio
    async def test_process_batch_simple(self, mock_generator, sample_instances):
        """Test simple batch processing without checkpoints."""
        # Mock successful generation
        mock_generator.generate_patch.return_value = GenerationResult(
            patch="test patch",
            instance_id="test-1",
            model="test-model",
            attempts=1,
            truncated=False,
            cost=0.1,
            success=True
        )

        processor = BatchProcessor(mock_generator, max_concurrent=1, progress_bar=False)
        result = await processor.process_batch(sample_instances)

        assert len(result.successful) == 3
        assert len(result.failed) == 0
        assert len(result.skipped) == 0
        assert result.stats.completed == 3
        assert result.stats.success_rate == 1.0
        # 3 * 0.1, allow for floating point precision
        assert abs(result.stats.total_cost - 0.3) < 0.001

    @pytest.mark.asyncio
    async def test_process_batch_with_failures(self, mock_generator, sample_instances):
        """Test batch processing with some failures."""
        # Mock mixed results
        def mock_generate(instance):
            instance_id = instance.get("instance_id")
            if instance_id == "test-2":
                return GenerationResult(
                    patch=None,
                    instance_id=instance_id,
                    model="test-model",
                    attempts=1,
                    truncated=False,
                    cost=0.05,
                    success=False,
                    error="Generation failed"
                )
            else:
                return GenerationResult(
                    patch="test patch",
                    instance_id=instance_id,
                    model="test-model",
                    attempts=1,
                    truncated=False,
                    cost=0.1,
                    success=True
                )

        mock_generator.generate_patch.side_effect = mock_generate

        processor = BatchProcessor(mock_generator, max_concurrent=1, progress_bar=False)
        result = await processor.process_batch(sample_instances)

        assert len(result.successful) == 2
        assert len(result.failed) == 1
        assert result.stats.completed == 2
        assert result.stats.failed == 1
        assert result.stats.success_rate == 2/3

    @pytest.mark.asyncio
    async def test_process_single_instance(self, mock_generator):
        """Test processing a single instance."""
        instance = {"instance_id": "test-1", "problem_statement": "Test problem"}
        semaphore = asyncio.Semaphore(1)

        mock_generator.generate_patch.return_value = GenerationResult(
            patch="test patch",
            instance_id="test-1",
            model="test-model",
            attempts=1,
            truncated=False,
            cost=0.1,
            success=True
        )

        processor = BatchProcessor(mock_generator)
        result = await processor._process_single_instance(instance, semaphore)

        assert result["success"] is True
        assert result["cost"] == 0.1
        assert result["attempts"] == 1

    @pytest.mark.asyncio
    async def test_handle_instance_error(self, mock_generator):
        """Test error handling for instance processing."""
        instance = {"instance_id": "test-1"}
        error = ProviderRateLimitError("Rate limited")

        processor = BatchProcessor(mock_generator)

        with patch('asyncio.sleep') as mock_sleep:
            error_msg = await processor._handle_instance_error(instance, error, 1)

        assert "rate_limit" in error_msg
        mock_sleep.assert_called_once_with(60)  # First attempt backoff

    def test_save_and_load_checkpoint(self, mock_generator, temp_checkpoint_dir):
        """Test checkpoint saving and loading."""
        processor = BatchProcessor(mock_generator, checkpoint_dir=temp_checkpoint_dir)

        # Create test checkpoint data
        now = datetime.now()
        checkpoint_data = CheckpointData(
            timestamp=now,
            completed={"test-1", "test-2"},
            failed={"test-3": "error"},
            stats={"total_cost": 1.5},
            total_instances=3,
            start_time=now
        )

        # Save checkpoint
        success = processor.save_checkpoint(checkpoint_data)
        assert success is True

        # Load checkpoint
        loaded = processor.load_checkpoint()
        assert loaded is not None
        assert loaded.completed == {"test-1", "test-2"}
        assert loaded.failed == {"test-3": "error"}
        assert loaded.stats == {"total_cost": 1.5}

    def test_clear_checkpoint(self, mock_generator, temp_checkpoint_dir):
        """Test clearing checkpoint."""
        processor = BatchProcessor(mock_generator, checkpoint_dir=temp_checkpoint_dir)

        # Create a checkpoint file
        checkpoint_path = processor.get_checkpoint_path()
        checkpoint_path.write_text("{}")
        assert checkpoint_path.exists()

        # Clear checkpoint
        processor.clear_checkpoint()
        assert not checkpoint_path.exists()

    @pytest.mark.asyncio
    async def test_process_batch_with_resume(
        self, mock_generator, sample_instances, temp_checkpoint_dir
    ):
        """Test batch processing with checkpoint resume."""
        processor = BatchProcessor(
            mock_generator,
            checkpoint_dir=temp_checkpoint_dir,
            max_concurrent=1,
            progress_bar=False
        )

        # Create existing checkpoint
        now = datetime.now()
        checkpoint_data = CheckpointData(
            timestamp=now,
            completed={"test-1"},  # test-1 already completed
            failed={},
            stats={"total_cost": 0.1},
            total_instances=3,
            start_time=now
        )
        processor.save_checkpoint(checkpoint_data)

        # Mock generation for remaining instances
        mock_generator.generate_patch.return_value = GenerationResult(
            patch="test patch",
            instance_id="test-2",
            model="test-model",
            attempts=1,
            truncated=False,
            cost=0.1,
            success=True
        )

        result = await processor.process_batch(
            sample_instances, resume_from_checkpoint=True
        )

        # Should only process test-2 and test-3, skip test-1
        assert len(result.skipped) == 1
        assert "test-1" in result.skipped
        assert len(result.successful) == 2  # test-2 and test-3
        # 0.1 from checkpoint + 0.2 from processing
        assert abs(result.stats.total_cost - 0.3) < 0.001

    def test_generate_report(self, mock_generator):
        """Test report generation."""
        processor = BatchProcessor(mock_generator)

        # Create test result
        stats = BatchStats(
            total_instances=10,
            completed=8,
            failed=1,
            skipped=1,
            total_cost=2.5,
            total_time=60.0,
            success_rate=0.8
        )

        result = BatchResult(stats=stats)
        report = processor.generate_report(result)

        assert "Batch Processing Report" in report
        assert "Total instances: 10" in report
        assert "Successful: 8 (80.0%)" in report
        assert "Failed: 1" in report
        assert "Total cost: $2.5000" in report

    def test_format_stats_table(self, mock_generator):
        """Test statistics table formatting."""
        processor = BatchProcessor(mock_generator)

        stats = BatchStats(
            completed=8,
            failed=2,
            skipped=0,
            total_cost=1.25,
            total_time=120.0,
            success_rate=0.8
        )

        table = processor.format_stats_table(stats)

        assert "Statistics Table:" in table
        assert "80.0%" in table
        assert "1.2500" in table  # Just check the number part
        assert "120.0s" in table

    def test_save_and_load_results(self, mock_generator, temp_checkpoint_dir):
        """Test saving and loading results."""
        processor = BatchProcessor(mock_generator)
        output_file = temp_checkpoint_dir / "results.json"

        # Create test result
        generation_result = GenerationResult(
            patch="test patch",
            instance_id="test-1",
            model="test-model",
            attempts=1,
            truncated=False,
            cost=0.1,
            success=True,
            metadata={"test": "value"}
        )

        stats = BatchStats(total_instances=1, completed=1, total_cost=0.1)
        result = BatchResult(successful=[generation_result], stats=stats)

        # Save results
        processor.save_results(result, output_file)
        assert output_file.exists()

        # Load results
        loaded_result = processor.load_results(output_file)
        assert len(loaded_result.successful) == 1
        assert loaded_result.successful[0].instance_id == "test-1"
        assert loaded_result.successful[0].patch == "test patch"
        assert loaded_result.stats.total_instances == 1

    def test_estimate_batch_time(self, mock_generator):
        """Test batch time estimation."""
        processor = BatchProcessor(mock_generator, max_concurrent=5)

        # 10 instances with 5 concurrent should take roughly 2 * 30 seconds
        estimated_time = processor.estimate_batch_time(10)
        assert estimated_time == 60.0  # (10 * 30) / 5

    def test_estimate_batch_cost(self, mock_generator):
        """Test batch cost estimation."""
        processor = BatchProcessor(mock_generator)

        # 10 instances at $0.01 each
        estimated_cost = processor.estimate_batch_cost([{}] * 10)
        assert estimated_cost == 0.1  # 10 * 0.01


# Integration tests
class TestBatchProcessorIntegration:
    """Integration tests for BatchProcessor."""

    @pytest.mark.asyncio
    async def test_full_workflow_with_checkpoints(
        self, mock_generator, temp_checkpoint_dir
    ):
        """Test complete workflow with checkpointing."""
        instances = [
            {"instance_id": f"test-{i}", "problem_statement": f"Problem {i}"}
            for i in range(5)
        ]

        # Mock generator to succeed for first 3, fail for 4th, then succeed for 5th
        def mock_generate(instance):
            instance_id = instance.get("instance_id")
            if instance_id == "test-3":  # 4th instance (0-indexed)
                return GenerationResult(
                    patch=None,
                    instance_id=instance_id,
                    model="test-model",
                    attempts=1,
                    truncated=False,
                    cost=0.05,
                    success=False,
                    error="Mock failure"
                )
            else:
                return GenerationResult(
                    patch=f"patch for {instance_id}",
                    instance_id=instance_id,
                    model="test-model",
                    attempts=1,
                    truncated=False,
                    cost=0.1,
                    success=True
                )

        mock_generator.generate_patch.side_effect = mock_generate

        processor = BatchProcessor(
            mock_generator,
            checkpoint_dir=temp_checkpoint_dir,
            max_concurrent=2,
            checkpoint_interval=2,
            progress_bar=False,
            save_intermediate=True
        )

        # Process batch
        result = await processor.process_batch(instances)

        # Verify results
        assert len(result.successful) == 4  # test-0,1,2,4
        assert len(result.failed) == 1      # test-3
        assert result.stats.completed == 4
        assert result.stats.failed == 1
        # Only successful instances count toward cost: 4 * 0.1
        assert abs(result.stats.total_cost - 0.4) < 0.001
        assert result.checkpoint_saved is True

        # Verify checkpoint was saved
        checkpoint = processor.load_checkpoint()
        assert checkpoint is not None
        assert len(checkpoint.completed) == 4
        assert len(checkpoint.failed) == 1

        # Generate and verify report
        report = processor.generate_report(result)
        assert "4 (80.0%)" in report  # Success rate
        assert "Failed Instances:" in report
