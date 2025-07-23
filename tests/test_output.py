"""Tests for output formatting and display utilities."""

import json
import os
from pathlib import Path

from swebench_runner.models import EvaluationResult
from swebench_runner.output import detect_patches_file, display_result


class TestDetectPatchesFile:
    """Test patches file auto-detection."""

    def test_detects_predictions_jsonl(self, tmp_path):
        """Test detection of predictions.jsonl file."""
        # Create predictions.jsonl with content
        predictions_file = tmp_path / "predictions.jsonl"
        predictions_file.write_text('{"instance_id": "test", "patch": "diff"}')

        # Change to tmp directory for the test
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = detect_patches_file()
            assert result == Path("predictions.jsonl")
        finally:
            os.chdir(original_cwd)

    def test_detects_patches_jsonl(self, tmp_path):
        """Test detection of patches.jsonl file."""
        # Create patches.jsonl with content
        patches_file = tmp_path / "patches.jsonl"
        patches_file.write_text('{"instance_id": "test", "patch": "diff"}')

        # Change to tmp directory for the test
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = detect_patches_file()
            assert result == Path("patches.jsonl")
        finally:
            os.chdir(original_cwd)

    def test_prefers_predictions_over_patches(self, tmp_path):
        """Test that predictions.jsonl is preferred over patches.jsonl."""
        # Create both files
        predictions_file = tmp_path / "predictions.jsonl"
        predictions_file.write_text('{"instance_id": "test1", "patch": "diff1"}')
        patches_file = tmp_path / "patches.jsonl"
        patches_file.write_text('{"instance_id": "test2", "patch": "diff2"}')

        # Change to tmp directory for the test
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = detect_patches_file()
            assert result == Path("predictions.jsonl")
        finally:
            os.chdir(original_cwd)

    def test_ignores_empty_files(self, tmp_path):
        """Test that empty files are ignored."""
        # Create empty predictions.jsonl
        predictions_file = tmp_path / "predictions.jsonl"
        predictions_file.touch()

        # Create patches.jsonl with content
        patches_file = tmp_path / "patches.jsonl"
        patches_file.write_text('{"instance_id": "test", "patch": "diff"}')

        # Change to tmp directory for the test
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = detect_patches_file()
            assert result == Path("patches.jsonl")
        finally:
            os.chdir(original_cwd)

    def test_returns_none_if_no_files(self, tmp_path):
        """Test returns None if no patch files found."""
        # Change to tmp directory for the test
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = detect_patches_file()
            assert result is None
        finally:
            os.chdir(original_cwd)

    def test_returns_none_if_only_empty_files(self, tmp_path):
        """Test returns None if only empty files exist."""
        # Create empty files
        (tmp_path / "predictions.jsonl").touch()
        (tmp_path / "patches.jsonl").touch()

        # Change to tmp directory for the test
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = detect_patches_file()
            assert result is None
        finally:
            os.chdir(original_cwd)


class TestDisplayResult:
    """Test result display functionality."""

    def test_display_passed_result(self, capsys):
        """Test displaying a passed result."""
        result = EvaluationResult(
            instance_id="django__django-12345",
            passed=True,
            error=None
        )

        display_result(result)

        captured = capsys.readouterr()
        assert "✅ django__django-12345: PASSED" in captured.out
        assert "Error:" not in captured.out

    def test_display_failed_result(self, capsys):
        """Test displaying a failed result."""
        result = EvaluationResult(
            instance_id="astropy__astropy-12907",
            passed=False,
            error="Test failed: assertion error in test_foo"
        )

        display_result(result)

        captured = capsys.readouterr()
        assert "❌ astropy__astropy-12907: FAILED" in captured.out
        assert "Error: Test failed: assertion error in test_foo" in captured.out

    def test_display_multiline_error(self, capsys):
        """Test displaying a multi-line error message."""
        result = EvaluationResult(
            instance_id="test__test-123",
            passed=False,
            error="First line of error\nSecond line of error\nThird line"
        )

        display_result(result)

        captured = capsys.readouterr()
        assert "Error: First line of error" in captured.out
        assert "          Second line of error" in captured.out
        assert "          Third line" in captured.out

    def test_save_result_summary(self, tmp_path, capsys):
        """Test saving result summary to output directory."""
        result = EvaluationResult(
            instance_id="test__test-123",
            passed=True,
            error=None
        )

        output_dir = tmp_path / "test_output"
        display_result(result, output_dir)

        # Check directory was created
        assert output_dir.exists()

        # Check summary.json was created
        summary_path = output_dir / "summary.json"
        assert summary_path.exists()

        # Verify content
        with open(summary_path) as f:
            data = json.load(f)
            assert data["instance_id"] == "test__test-123"
            assert data["passed"] is True
            assert data["error"] is None
            assert "timestamp" in data
            assert "swe_bench_runner_version" in data

        # Check status file
        assert (output_dir / "PASSED").exists()
        assert not (output_dir / "FAILED").exists()

        # Check output message
        captured = capsys.readouterr()
        assert f"📁 Results saved to: {output_dir}" in captured.out

    def test_save_failed_result(self, tmp_path):
        """Test saving a failed result creates FAILED status file."""
        result = EvaluationResult(
            instance_id="test__test-456",
            passed=False,
            error="Some error"
        )

        output_dir = tmp_path / "test_output"
        display_result(result, output_dir)

        # Check status file
        assert (output_dir / "FAILED").exists()
        assert not (output_dir / "PASSED").exists()

    def test_no_save_without_output_dir(self, tmp_path, capsys):
        """Test that results are not saved if output_dir is None."""
        result = EvaluationResult(
            instance_id="test__test-789",
            passed=True,
            error=None
        )

        display_result(result, None)

        captured = capsys.readouterr()
        assert "✅ test__test-789: PASSED" in captured.out
        assert "Results saved to:" not in captured.out

        # Verify no files were created in current directory
        assert not Path("summary.json").exists()
        assert not Path("PASSED").exists()
