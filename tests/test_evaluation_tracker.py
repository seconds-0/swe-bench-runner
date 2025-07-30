"""Tests for evaluation tracker functionality."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.swebench_runner.evaluation_tracker import EvaluationStats, EvaluationTracker


class TestEvaluationStats:
    """Test EvaluationStats dataclass functionality."""

    def test_default_initialization(self):
        """Test that EvaluationStats initializes with correct defaults."""
        stats = EvaluationStats()
        
        assert stats.test_set_type is None
        assert stats.test_set_size is None
        assert stats.selected_count == 0
        assert stats.evaluated_count == 0
        assert stats.missing_count == 0
        assert stats.failed_count == 0
        assert stats.passed_count == 0
        assert stats.start_time is None
        assert stats.end_time is None
        assert stats.results == {}
    
    def test_swe_bench_score_calculation(self):
        """Test SWE-bench score calculation."""
        stats = EvaluationStats()
        
        # Test zero division
        assert stats.swe_bench_score == 0.0
        
        # Test normal calculation
        stats.selected_count = 10
        stats.passed_count = 7
        assert stats.swe_bench_score == 70.0
        
        # Test all passed
        stats.selected_count = 5
        stats.passed_count = 5
        assert stats.swe_bench_score == 100.0
        
        # Test none passed
        stats.selected_count = 3
        stats.passed_count = 0
        assert stats.swe_bench_score == 0.0
    
    def test_duration_calculation(self):
        """Test duration calculation and formatting."""
        stats = EvaluationStats()
        
        # Test with no times set
        assert stats.duration is None
        assert stats.duration_str == "N/A"
        
        # Test with only start time
        stats.start_time = datetime.now()
        assert stats.duration is None
        assert stats.duration_str == "N/A"
        
        # Test with both times - seconds only
        stats.start_time = datetime.now()
        stats.end_time = stats.start_time + timedelta(seconds=45)
        assert abs(stats.duration - 45.0) < 0.1
        assert stats.duration_str == "45s"
        
        # Test with minutes
        stats.end_time = stats.start_time + timedelta(minutes=2, seconds=30)
        assert stats.duration_str == "2m 30s"
        
        # Test with hours
        stats.end_time = stats.start_time + timedelta(hours=1, minutes=15, seconds=45)
        assert stats.duration_str == "1h 15m 45s"
    
    def test_to_dict_serialization(self):
        """Test conversion to dictionary for JSON serialization."""
        stats = EvaluationStats()
        stats.test_set_type = "lite"
        stats.test_set_size = 300
        stats.selected_count = 10
        stats.evaluated_count = 8
        stats.passed_count = 6
        stats.failed_count = 2
        stats.missing_count = 2
        stats.start_time = datetime(2023, 1, 1, 10, 0, 0)
        stats.end_time = datetime(2023, 1, 1, 10, 5, 0)
        stats.results = {
            "test-1": {"passed": True, "error": None},
            "test-2": {"passed": False, "error": "Failed"}
        }
        
        result = stats.to_dict()
        
        assert result["test_set_type"] == "lite"
        assert result["test_set_size"] == 300
        assert result["selected_count"] == 10
        assert result["evaluated_count"] == 8
        assert result["passed_count"] == 6
        assert result["failed_count"] == 2
        assert result["missing_count"] == 2
        assert result["swe_bench_score"] == 60.0
        assert result["duration"] == 300.0
        assert result["duration_str"] == "5m 0s"
        assert result["results"] == stats.results
        
        # Ensure it's JSON serializable
        json_str = json.dumps(result)
        assert json_str is not None
    
    def test_save_to_file(self, tmp_path):
        """Test saving statistics to JSON file."""
        stats = EvaluationStats()
        stats.test_set_type = "custom"
        stats.selected_count = 5
        stats.passed_count = 3
        
        # Test successful save
        output_file = tmp_path / "test_stats.json"
        stats.save(output_file)
        
        assert output_file.exists()
        
        # Verify content
        with open(output_file) as f:
            loaded = json.load(f)
        
        assert loaded["test_set_type"] == "custom"
        assert loaded["selected_count"] == 5
        assert loaded["passed_count"] == 3
        assert loaded["swe_bench_score"] == 60.0
    
    def test_save_with_error_handling(self, tmp_path, monkeypatch):
        """Test save method error handling."""
        stats = EvaluationStats()
        
        # Test with non-existent parent directory
        deep_path = tmp_path / "non" / "existent" / "path" / "stats.json"
        stats.save(deep_path)  # Should create directories
        assert deep_path.exists()
        
        # Test with permission error (mock)
        def mock_write_text(*args, **kwargs):
            raise OSError("Permission denied")
        
        monkeypatch.setattr(Path, "write_text", mock_write_text)
        
        # Should print warning but not raise
        stats.save(tmp_path / "test.json")  # This will print a warning


class TestEvaluationTracker:
    """Test EvaluationTracker functionality."""
    
    def test_initialization(self):
        """Test tracker initialization."""
        tracker = EvaluationTracker()
        
        assert tracker.stats is not None
        assert isinstance(tracker.stats, EvaluationStats)
        assert tracker.stats.selected_count == 0
    
    def test_set_test_set_info(self):
        """Test setting test set information."""
        tracker = EvaluationTracker()
        
        tracker.set_test_set_info("verified", 500)
        
        assert tracker.stats.test_set_type == "verified"
        assert tracker.stats.test_set_size == 500
    
    def test_start_evaluation(self):
        """Test starting evaluation tracking."""
        tracker = EvaluationTracker()
        
        tracker.start_evaluation(25)
        
        assert tracker.stats.selected_count == 25
        assert tracker.stats.start_time is not None
        assert isinstance(tracker.stats.start_time, datetime)
    
    def test_record_result(self):
        """Test recording evaluation results."""
        tracker = EvaluationTracker()
        tracker.start_evaluation(3)
        
        # Record passed result
        tracker.record_result("test-1", True)
        assert tracker.stats.evaluated_count == 1
        assert tracker.stats.passed_count == 1
        assert tracker.stats.failed_count == 0
        assert "test-1" in tracker.stats.results
        assert tracker.stats.results["test-1"]["passed"] is True
        
        # Record failed result
        tracker.record_result("test-2", False, {"error": "Test failed"})
        assert tracker.stats.evaluated_count == 2
        assert tracker.stats.passed_count == 1
        assert tracker.stats.failed_count == 1
        assert tracker.stats.results["test-2"]["passed"] is False
        assert tracker.stats.results["test-2"]["details"]["error"] == "Test failed"
        
        # Record another passed
        tracker.record_result("test-3", True)
        assert tracker.stats.evaluated_count == 3
        assert tracker.stats.passed_count == 2
        assert tracker.stats.failed_count == 1
    
    def test_record_missing(self):
        """Test recording missing instances."""
        tracker = EvaluationTracker()
        
        tracker.record_missing("missing-1")
        
        assert tracker.stats.missing_count == 1
        assert "missing-1" in tracker.stats.results
        assert tracker.stats.results["missing-1"]["passed"] is False
        assert tracker.stats.results["missing-1"]["details"]["status"] == "missing_patch"
    
    def test_finish_evaluation(self):
        """Test finishing evaluation."""
        tracker = EvaluationTracker()
        tracker.start_evaluation(5)
        
        # Ensure start time is set
        assert tracker.stats.start_time is not None
        
        tracker.finish_evaluation()
        
        assert tracker.stats.end_time is not None
        assert tracker.stats.end_time >= tracker.stats.start_time
    
    def test_get_progress(self):
        """Test progress reporting."""
        tracker = EvaluationTracker()
        
        # Test with no evaluation started
        progress = tracker.get_progress()
        assert progress == (0, 0, 0.0)
        
        # Test with evaluation in progress
        tracker.start_evaluation(10)
        progress = tracker.get_progress()
        assert progress == (0, 10, 0.0)
        
        # Record some results
        tracker.record_result("test-1", True)
        progress = tracker.get_progress()
        assert progress == (1, 10, 10.0)
        
        tracker.record_result("test-2", False)
        tracker.record_result("test-3", True)
        progress = tracker.get_progress()
        assert progress == (3, 10, 30.0)
        
        # Test completion
        for i in range(4, 11):
            tracker.record_result(f"test-{i}", True)
        progress = tracker.get_progress()
        assert progress == (10, 10, 100.0)
    
    def test_full_evaluation_workflow(self):
        """Test a complete evaluation workflow."""
        tracker = EvaluationTracker()
        
        # Setup
        tracker.set_test_set_info("lite", 300)
        tracker.start_evaluation(5)
        
        # Simulate evaluation
        tracker.record_result("django__django-12345", True)
        tracker.record_result("flask__flask-2345", False, {"error": "Import error"})
        tracker.record_result("requests__requests-3456", True)
        tracker.record_missing("pandas__pandas-4567")
        tracker.record_result("numpy__numpy-5678", True)
        
        # Finish
        tracker.finish_evaluation()
        
        # Verify final state
        assert tracker.stats.test_set_type == "lite"
        assert tracker.stats.test_set_size == 300
        assert tracker.stats.selected_count == 5
        assert tracker.stats.evaluated_count == 4  # missing doesn't count as evaluated
        assert tracker.stats.passed_count == 3
        assert tracker.stats.failed_count == 1
        assert tracker.stats.missing_count == 1
        assert tracker.stats.swe_bench_score == 60.0  # 3/5 = 60%
        
        # Check progress (includes missing instances)
        current, total, percentage = tracker.get_progress()
        assert current == 5  # 4 evaluated + 1 missing
        assert total == 5
        assert percentage == 100.0  # 5/5 = 100% progress


class TestIntegration:
    """Integration tests for evaluation tracking."""
    
    def test_batch_evaluation_simulation(self, tmp_path):
        """Simulate a batch evaluation with various outcomes."""
        tracker = EvaluationTracker()
        tracker.set_test_set_info("custom", 10)
        tracker.start_evaluation(10)
        
        # Simulate various evaluation outcomes
        test_cases = [
            ("test-1", True, None),
            ("test-2", False, {"error": "Syntax error"}),
            ("test-3", True, None),
            ("test-4", True, None),
            ("test-5", False, {"error": "Import error"}),
            ("test-6", True, None),
            ("test-7", False, {"error": "Test failed"}),
            ("test-8", True, None),
            ("test-9", True, None),
            ("test-10", True, None),
        ]
        
        for instance_id, passed, details in test_cases:
            tracker.record_result(instance_id, passed, details)
        
        tracker.finish_evaluation()
        
        # Save results
        stats_file = tmp_path / "batch_stats.json"
        tracker.stats.save(stats_file)
        
        # Verify saved data
        with open(stats_file) as f:
            saved_data = json.load(f)
        
        assert saved_data["selected_count"] == 10
        assert saved_data["evaluated_count"] == 10
        assert saved_data["passed_count"] == 7
        assert saved_data["failed_count"] == 3
        assert saved_data["swe_bench_score"] == 70.0
        assert len(saved_data["results"]) == 10