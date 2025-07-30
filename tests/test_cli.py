"""Tests for the CLI interface."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from swebench_runner.cli import cli
from swebench_runner.models import EvaluationResult


class TestCLI:
    """Test suite for CLI functionality."""

    def test_version_flag(self) -> None:
        """Test --version flag shows correct version."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])

        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_help_flag(self) -> None:
        """Test --help flag shows usage information."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "SWE-bench evaluation runner" in result.output
        assert "Commands:" in result.output
        assert "run" in result.output

    def test_run_command_help(self) -> None:
        """Test run command --help shows specific help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--help"])

        assert result.exit_code == 0
        assert "Run SWE-bench evaluation" in result.output
        assert "--patches" in result.output
        assert "Path to JSONL file containing patches" in result.output

    def test_run_missing_patches_argument(self) -> None:
        """Test run command fails when --patches is missing."""
        runner = CliRunner()
        result = runner.invoke(cli, ["run"])

        assert result.exit_code != 0
        assert (
            "Error: Must provide --patches, --patches-dir, or --dataset"
            in result.output
        )

    def test_run_with_nonexistent_file(self) -> None:
        """Test run command fails when patches file doesn't exist."""
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--patches", "nonexistent.jsonl"])

        assert result.exit_code != 0
        assert "does not exist" in result.output.lower()

    def test_run_with_empty_file(self) -> None:
        """Test run command fails when patches file is empty."""
        runner = CliRunner()

        # Use the empty fixture file
        fixtures_dir = Path(__file__).parent / "fixtures"
        empty_file = fixtures_dir / "empty.jsonl"

        result = runner.invoke(cli, ["run", "--patches", str(empty_file)])

        assert result.exit_code == 1
        assert "Error: Patches file is empty" in result.output

    def test_run_with_directory_instead_of_file(self) -> None:
        """Test run command fails when patches path is a directory."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as temp_dir:
            result = runner.invoke(cli, ["run", "--patches", temp_dir])

            assert result.exit_code == 1
            assert "is not a file" in result.output

    @patch("swebench_runner.cli.run_evaluation")
    def test_run_with_valid_file(self, mock_run_evaluation) -> None:
        """Test run command succeeds with valid patches file."""
        mock_run_evaluation.return_value = EvaluationResult(
            instance_id="test__repo-123",
            passed=True,
            error=None
        )

        runner = CliRunner()

        # Use the sample fixture file
        fixtures_dir = Path(__file__).parent / "fixtures"
        sample_file = fixtures_dir / "sample.jsonl"

        result = runner.invoke(cli, ["run", "--patches", str(sample_file)])

        assert result.exit_code == 0
        assert "✅ test__repo-123: PASSED" in result.output

    @patch("swebench_runner.cli.run_evaluation")
    def test_run_with_multiline_file(self, mock_run_evaluation) -> None:
        """Test run command works with single-patch JSONL file.
        
        Note: Multi-patch files now trigger batch evaluation with summary display.
        This test verifies backward compatibility with single-patch files.
        """
        mock_run_evaluation.return_value = EvaluationResult(
            instance_id="test__repo-456",
            passed=True,
            error=None
        )

        runner = CliRunner()

        # Use the single-patch fixture file to avoid batch mode
        fixtures_dir = Path(__file__).parent / "fixtures"
        single_file = fixtures_dir / "sample.jsonl"

        result = runner.invoke(cli, ["run", "--patches", str(single_file)])

        assert result.exit_code == 0
        assert "✅ test__repo-456: PASSED" in result.output

    @patch("swebench_runner.docker_run.run_batch_evaluation")
    def test_run_with_batch_file(self, mock_run_batch_evaluation) -> None:
        """Test run command works with multi-patch JSONL file triggering batch mode."""
        # For batch evaluation, return a list of results
        mock_run_batch_evaluation.return_value = [
            EvaluationResult(
                instance_id="django__django-12345",
                passed=True,
                error=None
            ),
            EvaluationResult(
                instance_id="flask__flask-2001",
                passed=True,
                error=None
            )
        ]

        runner = CliRunner()

        # Use the multi-patch fixture file
        fixtures_dir = Path(__file__).parent / "fixtures"
        multi_file = fixtures_dir / "multi_patch.jsonl"

        result = runner.invoke(cli, ["run", "--patches", str(multi_file)])

        assert result.exit_code == 0
        # Check that batch evaluation was called
        mock_run_batch_evaluation.assert_called_once()
        # Check that summary is displayed (our new feature)
        assert "EVALUATION SUMMARY" in result.output
        assert "Selected: 2" in result.output
        assert "Succeeded: 2" in result.output

    @patch("swebench_runner.cli.run_evaluation")
    def test_run_with_relative_path(self, mock_run_evaluation) -> None:
        """Test run command works with relative path."""
        mock_run_evaluation.return_value = EvaluationResult(
            instance_id="test",
            passed=True,
            error=None
        )

        runner = CliRunner()

        # Change to fixtures directory and use relative path
        with runner.isolated_filesystem():
            # Create a sample file in the isolated filesystem
            sample_content = (
                '{"instance_id": "test", '
                '"patch": "diff --git a/test.py b/test.py\\n+test"}'
            )
            with open("test.jsonl", "w") as f:
                f.write(sample_content)

            result = runner.invoke(cli, ["run", "--patches", "test.jsonl"])

            assert result.exit_code == 0
            assert "✅ test: PASSED" in result.output

    @patch("swebench_runner.cli.run_evaluation")
    def test_run_with_absolute_path(self, mock_run_evaluation) -> None:
        """Test run command works with absolute path."""
        mock_run_evaluation.return_value = EvaluationResult(
            instance_id="test__repo-123",
            passed=True,
            error=None
        )

        runner = CliRunner()

        fixtures_dir = Path(__file__).parent / "fixtures"
        sample_file = fixtures_dir / "sample.jsonl"

        result = runner.invoke(cli, ["run", "--patches", str(sample_file.absolute())])

        assert result.exit_code == 0
        assert "✅ test__repo-123: PASSED" in result.output

    def test_cli_module_execution(self) -> None:
        """Test that CLI can be executed as a module."""
        # This test ensures that the __main__.py works correctly
        result = os.system("cd /tmp && python3 -m swebench_runner --version")
        # A successful execution should return 0
        assert result == 0

    def test_cli_entry_point_format(self) -> None:
        """Test that CLI entry point is correctly formatted."""
        # This is more of a validation test for the package structure
        from swebench_runner.cli import cli as cli_function

        # Should be a Click command
        assert callable(cli_function)
        assert hasattr(cli_function, "commands")
        assert "run" in cli_function.commands


class TestErrorHandling:
    """Test suite for error handling scenarios."""

    def test_permission_denied_file(self) -> None:
        """Test behavior when file exists but is not readable."""
        runner = CliRunner()

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
            tmp.write('{"instance_id": "test", "patch": "diff"}')
            tmp.flush()

            # Remove read permissions
            os.chmod(tmp.name, 0o000)

            try:
                result = runner.invoke(cli, ["run", "--patches", tmp.name])
                # Click should handle the permission error gracefully
                assert result.exit_code != 0
            finally:
                # Restore permissions for cleanup
                os.chmod(tmp.name, 0o644)
                os.unlink(tmp.name)

    @patch("swebench_runner.cli.run_evaluation")
    def test_special_characters_in_filename(self, mock_run_evaluation) -> None:
        """Test run command with special characters in filename."""
        mock_run_evaluation.return_value = EvaluationResult(
            instance_id="test",
            passed=True,
            error=None
        )

        runner = CliRunner()

        with runner.isolated_filesystem():
            # Create a file with special characters
            special_filename = "test-file_with.special@chars.jsonl"
            sample_content = (
                '{"instance_id": "test", '
                '"patch": "diff --git a/test.py b/test.py\\n+test"}'
            )
            with open(special_filename, "w") as f:
                f.write(sample_content)

            result = runner.invoke(cli, ["run", "--patches", special_filename])

            assert result.exit_code == 0
            assert "✅ test: PASSED" in result.output
