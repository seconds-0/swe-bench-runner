"""Tests for evaluation summary display functionality."""

from __future__ import annotations

from datetime import datetime, timedelta
from io import StringIO
from unittest.mock import patch

import pytest

from src.swebench_runner.evaluation_tracker import EvaluationStats
from src.swebench_runner.output import display_evaluation_summary


class TestDisplayEvaluationSummary:
    """Test the display_evaluation_summary function."""
    
    def test_display_basic_summary(self):
        """Test displaying a basic evaluation summary."""
        stats = EvaluationStats()
        stats.test_set_type = "lite"
        stats.test_set_size = 300
        stats.selected_count = 10
        stats.evaluated_count = 10
        stats.failed_count = 3
        stats.passed_count = 7
        
        # Capture output
        with patch('sys.stdout', new=StringIO()) as fake_out:
            display_evaluation_summary(stats)
            output = fake_out.getvalue()
        
        # Verify output contains expected elements
        assert "=" * 60 in output
        assert "🎯 EVALUATION SUMMARY" in output
        assert "📊 Test Set: SWE-bench lite" in output
        assert "📈 Statistics:" in output
        assert "   • Selected: 10" in output
        assert "   • Evaluated: 10" in output
        assert "   • Failed: 3" in output
        assert "   • Passed: 7" in output
        assert "🏆 SWE-bench Score: 70.0%" in output
        assert "   (7/10 instances resolved)" in output
    
    def test_display_with_timing(self):
        """Test displaying summary with timing information."""
        stats = EvaluationStats()
        stats.test_set_type = "verified"
        stats.selected_count = 5
        stats.evaluated_count = 5
        stats.passed_count = 3
        stats.failed_count = 2
        stats.start_time = datetime(2023, 1, 1, 10, 0, 0)
        stats.end_time = datetime(2023, 1, 1, 10, 15, 30)
        
        with patch('sys.stdout', new=StringIO()) as fake_out:
            display_evaluation_summary(stats)
            output = fake_out.getvalue()
        
        assert "⏱️  Timing:" in output
        assert "   Start: 2023-01-01 10:00:00" in output
        assert "   End:   2023-01-01 10:15:30" in output
        assert "   Duration: 15m 30s" in output
    
    def test_display_without_timing(self):
        """Test displaying summary without timing information."""
        stats = EvaluationStats()
        stats.test_set_type = "custom"
        stats.selected_count = 1
        stats.evaluated_count = 1
        stats.passed_count = 1
        
        with patch('sys.stdout', new=StringIO()) as fake_out:
            display_evaluation_summary(stats)
            output = fake_out.getvalue()
        
        # Should not contain timing section
        assert "⏱️  Timing:" not in output
        assert "   Start:" not in output
        assert "   Duration:" not in output
    
    def test_display_zero_selected(self):
        """Test displaying summary with zero selected instances."""
        stats = EvaluationStats()
        stats.test_set_type = "empty"
        stats.selected_count = 0
        
        with patch('sys.stdout', new=StringIO()) as fake_out:
            display_evaluation_summary(stats)
            output = fake_out.getvalue()
        
        assert "   • Selected: 0" in output
        assert "🏆 SWE-bench Score: 0.0%" in output
        assert "   (0/0 instances resolved)" in output
    
    def test_display_all_failed(self):
        """Test displaying summary where all instances failed."""
        stats = EvaluationStats()
        stats.test_set_type = "full"
        stats.test_set_size = 2294
        stats.selected_count = 20
        stats.evaluated_count = 20
        stats.failed_count = 20
        stats.passed_count = 0
        
        with patch('sys.stdout', new=StringIO()) as fake_out:
            display_evaluation_summary(stats)
            output = fake_out.getvalue()
        
        assert "📊 Test Set: SWE-bench full" in output
        assert "   • Failed: 20" in output
        assert "   • Passed: 0" in output
        assert "🏆 SWE-bench Score: 0.0%" in output
    
    def test_display_all_passed(self):
        """Test displaying summary where all instances passed."""
        stats = EvaluationStats()
        stats.test_set_type = "perfect"
        stats.selected_count = 15
        stats.evaluated_count = 15
        stats.failed_count = 0
        stats.passed_count = 15
        
        with patch('sys.stdout', new=StringIO()) as fake_out:
            display_evaluation_summary(stats)
            output = fake_out.getvalue()
        
        assert "   • Failed: 0" in output
        assert "   • Passed: 15" in output
        assert "🏆 SWE-bench Score: 100.0%" in output
        assert "   (15/15 instances resolved)" in output
    
    def test_display_with_missing(self):
        """Test displaying summary with missing instances."""
        stats = EvaluationStats()
        stats.test_set_type = "incomplete"
        stats.selected_count = 10
        stats.evaluated_count = 7
        stats.missing_count = 3
        stats.passed_count = 5
        stats.failed_count = 2
        
        with patch('sys.stdout', new=StringIO()) as fake_out:
            display_evaluation_summary(stats)
            output = fake_out.getvalue()
        
        # Note: Current implementation doesn't show missing count
        # but the score calculation includes them
        assert "   • Selected: 10" in output
        assert "   • Evaluated: 7" in output
        assert "🏆 SWE-bench Score: 50.0%" in output  # 5/10 = 50%
    
    def test_display_formatting(self):
        """Test the overall formatting of the summary."""
        stats = EvaluationStats()
        stats.test_set_type = "format-test"
        stats.selected_count = 100
        stats.evaluated_count = 100
        stats.passed_count = 66
        stats.failed_count = 34
        
        with patch('sys.stdout', new=StringIO()) as fake_out:
            display_evaluation_summary(stats)
            output = fake_out.getvalue()
        
        lines = output.strip().split('\n')
        
        # Check structure
        assert lines[0] == "=" * 60
        assert lines[1] == "🎯 EVALUATION SUMMARY"
        assert lines[2] == "=" * 60
        assert lines[-1] == "=" * 60
        
        # Check score precision
        assert "66.0%" in output  # Should show one decimal place