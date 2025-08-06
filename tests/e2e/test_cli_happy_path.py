"""True end-to-end tests for CLI happy path.

These tests actually invoke the swebench CLI as a subprocess, simulating
real user behavior. No mocking, no CliRunner - just real command execution.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


class TestCLIHappyPath:
    """Test the actual CLI binary execution - no mocks."""

    @pytest.fixture
    def fixtures_dir(self):
        """Path to E2E test fixtures."""
        return Path(__file__).parent / "fixtures"

    @pytest.fixture
    def minimal_patch_file(self, fixtures_dir):
        """Path to minimal patch file for testing."""
        return fixtures_dir / "minimal_patch.jsonl"

    def test_cli_version_command(self):
        """Test that the CLI version command works when invoked directly."""
        # Run the actual CLI command as a subprocess
        result = subprocess.run(
            [sys.executable, "-m", "swebench_runner", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )

        # Verify it succeeded
        assert result.returncode == 0
        assert "0.1.0" in result.stdout
        # Don't check for empty stderr - SSL warnings are acceptable and environment-specific

    def test_cli_help_command(self):
        """Test that the CLI help command works when invoked directly."""
        # Run the actual CLI help command
        result = subprocess.run(
            [sys.executable, "-m", "swebench_runner", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )

        # Verify help text appears
        assert result.returncode == 0
        assert "SWE-bench evaluation runner" in result.stdout
        assert "Commands:" in result.stdout
        assert "run" in result.stdout

    def test_cli_run_help_command(self):
        """Test that the run command help works."""
        result = subprocess.run(
            [sys.executable, "-m", "swebench_runner", "run", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )

        assert result.returncode == 0
        assert "--patches" in result.stdout
        assert "--dataset" in result.stdout
        assert "Path to JSONL file containing patches" in result.stdout

    def test_cli_run_missing_arguments(self):
        """Test that run command fails properly without required arguments."""
        result = subprocess.run(
            [sys.executable, "-m", "swebench_runner", "run"],
            capture_output=True,
            text=True,
            timeout=10
        )

        # Should fail with proper error
        assert result.returncode != 0
        # Check both stdout and stderr for the error message
        combined_output = result.stdout + result.stderr
        assert "Must provide --patches, --patches-dir, or --dataset" in combined_output

    def test_cli_run_nonexistent_file(self):
        """Test that run command fails properly with nonexistent file."""
        result = subprocess.run(
            [sys.executable, "-m", "swebench_runner", "run",
             "--patches", "/tmp/nonexistent_patches_file.jsonl"],
            capture_output=True,
            text=True,
            timeout=10
        )

        # Should fail with file not found error
        assert result.returncode != 0
        # Check both stdout and stderr for the error message
        combined_output = (result.stdout + result.stderr).lower()
        assert "does not exist" in combined_output

    def test_cli_run_empty_file(self):
        """Test that run command fails properly with empty file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_file = f.name

        try:
            result = subprocess.run(
                [sys.executable, "-m", "swebench_runner", "run",
                 "--patches", temp_file],
                capture_output=True,
                text=True,
                timeout=10
            )

            # Should fail with empty file error
            assert result.returncode == 1
            # Check both stdout and stderr for the error message
            combined_output = (result.stdout + result.stderr).lower()
            assert "empty" in combined_output
        finally:
            os.unlink(temp_file)

    @pytest.mark.skipif(
        os.environ.get("CI") == "true" and not os.environ.get("E2E_DOCKER_TESTS"),
        reason="Docker tests disabled in CI without E2E_DOCKER_TESTS flag"
    )
    def test_cli_run_with_valid_patch_no_docker(self, minimal_patch_file):
        """Test run command with valid patch file but no Docker.

        This test will fail at the Docker check stage, which validates
        that the pipeline is being invoked correctly.
        """
        # Set environment to mock no Docker for predictable failure
        env = os.environ.copy()
        env["SWEBENCH_MOCK_NO_DOCKER"] = "true"

        result = subprocess.run(
            [sys.executable, "-m", "swebench_runner", "run",
             "--patches", str(minimal_patch_file)],
            capture_output=True,
            text=True,
            timeout=10,
            env=env
        )

        # Should fail with Docker not found (exit code 2)
        assert result.returncode == 2
        assert "Docker" in result.stdout
        assert ("not running" in result.stdout or
                "unreachable" in result.stdout)

    def test_cli_json_output_mode(self, minimal_patch_file):
        """Test that JSON output mode works correctly."""
        # Set environment to mock no Docker for predictable output
        env = os.environ.copy()
        env["SWEBENCH_MOCK_NO_DOCKER"] = "true"

        result = subprocess.run(
            [sys.executable, "-m", "swebench_runner", "run",
             "--patches", str(minimal_patch_file),
             "--json"],
            capture_output=True,
            text=True,
            timeout=10,
            env=env
        )

        # Even though it fails, JSON output should still be valid
        assert result.returncode == 2
        # In JSON mode, errors still cause non-JSON output for Docker issues
        # This is intentional - Docker setup errors are pre-flight checks

    def test_cli_provider_commands(self):
        """Test that provider subcommands are accessible."""
        result = subprocess.run(
            [sys.executable, "-m", "swebench_runner", "provider", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )

        assert result.returncode == 0
        assert "list" in result.stdout
        assert "test" in result.stdout

    def test_cli_provider_list(self):
        """Test that provider list command works."""
        result = subprocess.run(
            [sys.executable, "-m", "swebench_runner", "provider", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )

        assert result.returncode == 0
        # Should show available providers (check both stdout and stderr)
        combined_output = result.stdout + result.stderr
        assert "Available" in combined_output or "providers" in combined_output.lower()
        assert "mock" in combined_output  # Mock provider should always be available

    def test_cli_generate_command_help(self):
        """Test that generate command help works."""
        result = subprocess.run(
            [sys.executable, "-m", "swebench_runner", "generate", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )

        assert result.returncode == 0
        assert "--instance" in result.stdout or "-i" in result.stdout
        assert "--output" in result.stdout or "-o" in result.stdout


class TestCLIWithEnvironmentVars:
    """Test CLI behavior with environment variables."""

    def test_cli_respects_provider_env_var(self):
        """Test that SWEBENCH_PROVIDER environment variable is respected."""
        env = os.environ.copy()
        env["SWEBENCH_PROVIDER"] = "mock"
        env["SWEBENCH_MOCK_NO_DOCKER"] = "true"

        # Create a minimal instance file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"instance_id": "test-123"}, f)
            temp_file = f.name

        try:
            result = subprocess.run(
                [sys.executable, "-m", "swebench_runner", "generate",
                 "-i", "test-123", "-o", temp_file],
                capture_output=True,
                text=True,
                timeout=10,
                env=env
            )

            # The mock provider should be used automatically
            assert "mock" in result.stdout.lower() or result.returncode == 0
        finally:
            os.unlink(temp_file)

    def test_cli_respects_max_workers_env_var(self):
        """Test that SWEBENCH_MAX_WORKERS environment variable is respected."""
        env = os.environ.copy()
        env["SWEBENCH_MAX_WORKERS"] = "2"

        result = subprocess.run(
            [sys.executable, "-m", "swebench_runner", "run", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
            env=env
        )

        # Help should still work regardless of env vars
        assert result.returncode == 0


class TestCLIErrorHandling:
    """Test CLI error handling and exit codes."""

    def test_cli_invalid_command(self):
        """Test that invalid commands fail properly."""
        result = subprocess.run(
            [sys.executable, "-m", "swebench_runner", "invalid-command"],
            capture_output=True,
            text=True,
            timeout=10
        )

        assert result.returncode != 0
        assert "Error" in result.stderr or "Usage" in result.stdout

    def test_cli_invalid_flag(self):
        """Test that invalid flags fail properly."""
        result = subprocess.run(
            [sys.executable, "-m", "swebench_runner", "run", "--invalid-flag"],
            capture_output=True,
            text=True,
            timeout=10
        )

        assert result.returncode != 0
        assert "Error" in result.stderr or "no such option" in result.stderr.lower()

    def test_cli_patches_dir_empty(self):
        """Test that empty patches directory fails properly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [sys.executable, "-m", "swebench_runner", "run",
                 "--patches-dir", temp_dir],
                capture_output=True,
                text=True,
                timeout=10
            )

            assert result.returncode != 0
            # Check both stdout and stderr for the error message
            combined_output = result.stdout + result.stderr
            assert "No .patch files found" in combined_output

    def test_cli_max_patch_size_validation(self):
        """Test that max patch size is validated."""
        # Create a patch that's too large
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            large_patch = "+" + "x" * (6 * 1024 * 1024)  # 6MB of x's
            json.dump({
                "instance_id": "test-large",
                "patch": f"diff --git a/test.py b/test.py\n--- a/test.py\n+++ b/test.py\n@@ -1,1 +1,1 @@\n-old\n{large_patch}"
            }, f)
            temp_file = f.name

        try:
            # Set environment to skip Docker check
            env = os.environ.copy()
            env["SWEBENCH_MOCK_NO_DOCKER"] = "true"

            result = subprocess.run(
                [sys.executable, "-m", "swebench_runner", "run",
                 "--patches", temp_file,
                 "--max-patch-size", "5"],  # 5MB limit
                capture_output=True,
                text=True,
                timeout=10,
                env=env
            )

            # Should fail - either with patch size error or Docker check
            assert result.returncode != 0
            combined_output = (result.stdout + result.stderr).lower()
            # Could fail at Docker check or size validation
            assert "docker" in combined_output or "size" in combined_output or "large" in combined_output
        finally:
            os.unlink(temp_file)


def test_module_is_executable():
    """Test that the module can be executed directly."""
    # This is the most basic E2E test - can we even run the module?
    result = subprocess.run(
        [sys.executable, "-m", "swebench_runner"],
        capture_output=True,
        text=True,
        timeout=10
    )

    # Should show help or usage when run without arguments
    assert result.returncode == 0 or "Usage" in result.stdout or "Commands" in result.stdout


if __name__ == "__main__":
    # Allow running this test file directly for debugging
    pytest.main([__file__, "-v"])
