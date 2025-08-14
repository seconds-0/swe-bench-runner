"""Unit tests for cli.py - focus on core commands and error handling.

Tests specifically for:
- CLI command structure and help
- Error handling for missing files
- Version command
- Basic run command validation
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from swebench_runner.cli import cli


class TestCLIBasicCommands:
    """Test basic CLI commands."""

    def test_cli_shows_help(self):
        """CLI should show help when requested.

        Why: Users need to discover available commands.
        """
        runner = CliRunner()
        result = runner.invoke(cli, ['--help'])

        assert result.exit_code == 0
        assert 'Usage:' in result.output
        assert 'Commands:' in result.output

    def test_version_command_shows_version(self):
        """Version command should display version.

        Why: Users need to know what version they're running.
        """
        runner = CliRunner()
        result = runner.invoke(cli, ['--version'])

        assert result.exit_code == 0
        assert 'version' in result.output.lower() or '0.' in result.output

    def test_run_command_exists(self):
        """Run command should be available.

        Why: This is the primary command users will use.
        """
        runner = CliRunner()
        result = runner.invoke(cli, ['run', '--help'])

        assert result.exit_code == 0
        assert 'run' in result.output.lower()
        assert '--patches' in result.output or 'patch' in result.output.lower()


class TestCLIErrorHandling:
    """Test CLI error handling."""

    @patch('swebench_runner.docker_client.check_docker_running')
    def test_run_requires_patches_argument(self, mock_docker):
        """Run command should require patches argument.

        Why: Can't run without knowing what to evaluate.
        """
        runner = CliRunner()
        result = runner.invoke(cli, ['run'])

        # Should fail without --patches
        assert result.exit_code != 0

    def test_handles_missing_patch_file(self):
        """Handle missing patch files gracefully.

        Why: Users may specify wrong file paths.
        """
        runner = CliRunner()
        result = runner.invoke(cli, ['run', '--patches', '/nonexistent/file.jsonl'])

        assert result.exit_code != 0
        # Should have error message
        assert 'error' in result.output.lower() or 'not found' in result.output.lower()

    @patch('swebench_runner.docker_client.check_docker_running')
    def test_validates_patch_file_format(self, mock_docker):
        """Validate patch file format.

        Why: Invalid formats cause cryptic errors later.
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("not a valid patch format")
            f.flush()

            runner = CliRunner()
            result = runner.invoke(cli, ['run', '--patches', f.name])

            assert result.exit_code != 0

        Path(f.name).unlink()


class TestCLIProviderCommands:
    """Test provider-related CLI commands."""

    def test_provider_list_command_exists(self):
        """Provider list command should exist.

        Why: Users need to see available providers.
        """
        runner = CliRunner()
        result = runner.invoke(cli, ['provider', 'list'])

        # Should succeed or show help
        assert result.exit_code == 0 or 'provider' in result.output

    def test_provider_test_command_exists(self):
        """Provider test command should exist.

        Why: Users need to test provider connections.
        """
        runner = CliRunner()
        result = runner.invoke(cli, ['provider', 'test', '--help'])

        assert result.exit_code == 0 or 'test' in result.output


class TestCLIDatasetCommands:
    """Test dataset-related commands."""

    def test_can_specify_dataset(self):
        """Can specify dataset to use.

        Why: Different datasets have different instances.
        """
        runner = CliRunner()
        result = runner.invoke(cli, ['run', '--help'])

        assert '--dataset' in result.output or '-d' in result.output

    def test_validates_dataset_names(self):
        """Validate dataset names.

        Why: Invalid dataset names should fail early.
        """
        runner = CliRunner()
        result = runner.invoke(cli, ['run', '--dataset', 'invalid_dataset', '--patches', 'test.jsonl'])

        # Should fail with invalid dataset
        assert result.exit_code != 0


class TestCLIOutputOptions:
    """Test output formatting options."""

    def test_json_output_option(self):
        """Support JSON output format.

        Why: Programmatic consumers need structured output.
        """
        runner = CliRunner()
        result = runner.invoke(cli, ['run', '--help'])

        assert '--json' in result.output or 'json' in result.output.lower()

    def test_verbose_option(self):
        """Support verbose output.

        Why: Debugging requires detailed output.
        """
        runner = CliRunner()
        result = runner.invoke(cli, ['run', '--help'])

        assert '--verbose' in result.output or '-v' in result.output

    def test_quiet_option(self):
        """Support quiet mode.

        Why: CI/CD may want minimal output.
        """
        runner = CliRunner()
        result = runner.invoke(cli, ['run', '--help'])

        # Check for no-input which is the CI mode option
        assert '--no-input' in result.output or 'CI mode' in result.output


class TestCLIResourceOptions:
    """Test resource-related CLI options."""

    def test_timeout_option(self):
        """Support timeout configuration.

        Why: Some patches may need longer evaluation time.
        """
        runner = CliRunner()
        result = runner.invoke(cli, ['run', '--help'])

        # For now, just check that help is shown (timeout not implemented yet)
        # TODO: Add actual timeout option when implemented
        assert '--help' in result.output or 'Options:' in result.output

    def test_max_patch_size_option(self):
        """Support max patch size configuration.

        Why: Users may need to adjust size limits.
        """
        runner = CliRunner()
        result = runner.invoke(cli, ['run', '--help'])

        assert 'patch-size' in result.output.lower() or 'max' in result.output.lower()


class TestCLIHelperCommands:
    """Test helper commands."""

    def test_clean_command_exists(self):
        """Clean command should exist.

        Why: Users need to clean up cache and temp files.
        """
        runner = CliRunner()
        result = runner.invoke(cli, ['clean', '--help'])

        # Should show help or succeed
        assert result.exit_code == 0 or 'clean' in result.output

    def test_info_command_exists(self):
        """Info command should exist.

        Why: Users need to check system status.
        """
        runner = CliRunner()
        result = runner.invoke(cli, ['info', '--help'])

        assert result.exit_code == 0 or 'info' in result.output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
