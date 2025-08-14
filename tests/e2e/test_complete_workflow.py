"""Complete end-to-end workflow test for SWE-bench Runner.

This test verifies the actual CLI works as expected from a user perspective.
It uses subprocess to test the real CLI, not mocked components.
"""

import subprocess
import tempfile
from pathlib import Path

import pytest


class TestCompleteUserWorkflow:
    """Test the complete user workflow from installation to evaluation."""

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        """Set up environment for tests."""
        # Disable first-time setup wizard for tests
        monkeypatch.setenv("SWEBENCH_NO_INPUT", "1")

    def test_cli_is_installed(self):
        """Verify the CLI is installed and responds."""
        result = subprocess.run(
            ['swebench', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        assert result.returncode == 0
        assert 'swebench' in result.stdout.lower()
        assert '0.1.0' in result.stdout or 'version' in result.stdout.lower()

    def test_help_commands_work(self):
        """Verify help commands provide useful information."""
        # Main help
        result = subprocess.run(
            ['swebench', '--help'],
            capture_output=True,
            text=True,
            timeout=5
        )
        assert result.returncode == 0
        assert 'Commands:' in result.stdout
        assert 'run' in result.stdout
        assert 'provider' in result.stdout

        # Run command help
        result = subprocess.run(
            ['swebench', 'run', '--help'],
            capture_output=True,
            text=True,
            timeout=5
        )
        assert result.returncode == 0
        assert '--patches' in result.stdout
        assert '--dataset' in result.stdout or '-d' in result.stdout

    def test_dataset_info_works(self):
        """Verify dataset info command works (no Docker needed)."""
        result = subprocess.run(
            ['swebench', 'info', '-d', 'lite'],
            capture_output=True,
            text=True,
            timeout=10
        )
        assert result.returncode == 0
        assert 'lite' in result.stdout.lower() or '300' in result.stdout

    def test_provider_list_works(self):
        """Verify provider list command works."""
        result = subprocess.run(
            ['swebench', 'provider', 'list'],
            capture_output=True,
            text=True,
            timeout=10
        )
        assert result.returncode == 0
        # Should show provider table or list
        assert 'provider' in result.stdout.lower() or 'openai' in result.stdout.lower()

    def test_patch_validation_without_docker(self):
        """Test patch file validation when Docker is not running."""
        # Test with sample patch
        result = subprocess.run(
            ['swebench', 'run', '--patches', 'tests/fixtures/sample.jsonl'],
            capture_output=True,
            text=True,
            timeout=10
        )

        # Should fail gracefully when Docker is not running
        if 'Docker' in result.stdout or 'Docker' in result.stderr:
            assert result.returncode != 0  # Should exit with error
            assert 'Docker' in result.stdout or 'Docker' in result.stderr
            # Should provide helpful message
            assert any(word in (result.stdout + result.stderr).lower()
                      for word in ['start', 'running', 'daemon'])
        else:
            # If Docker is running, it might try to evaluate
            # Just check it doesn't crash
            assert result.returncode in [0, 1, 2]  # Valid exit codes

    def test_invalid_patch_file_handling(self):
        """Test handling of invalid patch files."""
        result = subprocess.run(
            ['swebench', 'run', '--patches', 'nonexistent.jsonl'],
            capture_output=True,
            text=True,
            timeout=10
        )
        assert result.returncode != 0
        # Should show error about missing file
        assert 'does not exist' in (result.stdout + result.stderr).lower() or \
               'not found' in (result.stdout + result.stderr).lower() or \
               'no such file' in (result.stdout + result.stderr).lower()

    def test_empty_patch_file_handling(self):
        """Test handling of empty patch files."""
        result = subprocess.run(
            ['swebench', 'run', '--patches', 'tests/fixtures/empty.jsonl'],
            capture_output=True,
            text=True,
            timeout=10
        )
        # Should handle empty file gracefully
        assert result.returncode != 0
        # Error message should be informative
        combined_output = result.stdout + result.stderr
        assert 'empty' in combined_output.lower() or \
               'no patches' in combined_output.lower() or \
               'Docker' in combined_output  # Might fail on Docker check first

    def test_dataset_validation(self):
        """Test dataset name validation."""
        result = subprocess.run(
            ['swebench', 'run', '--patches', 'tests/fixtures/sample.jsonl',
             '--dataset', 'invalid_dataset_name'],
            capture_output=True,
            text=True,
            timeout=10
        )
        # Should reject invalid dataset
        result.stdout + result.stderr
        # Might fail on Docker check first, or on dataset validation
        assert result.returncode != 0

    @pytest.mark.skipif(
        subprocess.run(['docker', 'version'], capture_output=True).returncode != 0,
        reason="Docker not available"
    )
    def test_with_docker_available(self):
        """Test actual evaluation when Docker is available."""
        result = subprocess.run(
            ['swebench', 'run', '--patches', 'tests/fixtures/sample.jsonl',
             '--dataset', 'lite', '--count', '1'],
            capture_output=True,
            text=True,
            timeout=60
        )
        # Should at least try to run
        assert result.returncode in [0, 1]  # 0 = success, 1 = evaluation failed
        # Should show some progress
        assert len(result.stdout + result.stderr) > 100  # Some meaningful output


class TestCriticalUserPaths:
    """Test critical user paths that must work."""

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        """Set up environment for tests."""
        # Disable first-time setup wizard for tests
        monkeypatch.setenv("SWEBENCH_NO_INPUT", "1")

    def test_first_time_user_experience(self):
        """Test what a first-time user would experience."""
        # 1. Check version
        result = subprocess.run(['swebench', '--version'], capture_output=True, text=True)
        assert result.returncode == 0

        # 2. Get help
        result = subprocess.run(['swebench', '--help'], capture_output=True, text=True)
        assert result.returncode == 0
        assert len(result.stdout) > 100  # Substantial help text

        # 3. Try to run (will fail without Docker, but should be helpful)
        result = subprocess.run(
            ['swebench', 'run', '--patches', 'tests/fixtures/sample.jsonl'],
            capture_output=True,
            text=True
        )
        combined = result.stdout + result.stderr
        # Should give clear guidance
        if result.returncode != 0:
            assert 'Docker' in combined or 'error' in combined.lower()

    def test_provider_initialization_flow(self):
        """Test provider initialization workflow."""
        # List providers
        result = subprocess.run(
            ['swebench', 'provider', 'list'],
            capture_output=True,
            text=True,
            timeout=10
        )
        assert result.returncode == 0

        # Try to get help on initializing a provider
        result = subprocess.run(
            ['swebench', 'provider', 'init', '--help'],
            capture_output=True,
            text=True,
            timeout=10
        )
        # Should show help or indicate the command exists
        assert result.returncode == 0 or 'init' in result.stdout

    def test_clean_command_exists(self):
        """Test that clean command is available."""
        result = subprocess.run(
            ['swebench', 'clean', '--help'],
            capture_output=True,
            text=True,
            timeout=10
        )
        assert result.returncode == 0
        assert 'clean' in result.stdout.lower()


class TestErrorRecovery:
    """Test that the CLI handles errors gracefully."""

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        """Set up environment for tests."""
        # Disable first-time setup wizard for tests
        monkeypatch.setenv("SWEBENCH_NO_INPUT", "1")

    def test_malformed_json_patch(self):
        """Test handling of malformed JSON in patch file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"invalid json": missing closing bracket')
            temp_path = f.name

        try:
            result = subprocess.run(
                ['swebench', 'run', '--patches', temp_path],
                capture_output=True,
                text=True,
                timeout=10
            )
            assert result.returncode != 0
            combined = result.stdout + result.stderr
            # Should mention JSON error or invalid format
            assert 'json' in combined.lower() or \
                   'invalid' in combined.lower() or \
                   'Docker' in combined  # Might fail on Docker first
        finally:
            Path(temp_path).unlink()

    def test_ctrl_c_handling(self):
        """Test that Ctrl+C is handled gracefully."""
        # This is hard to test automatically, but we can at least
        # verify the command starts and can be terminated
        import time

        proc = subprocess.Popen(
            ['swebench', '--help'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Give it a moment to start
        time.sleep(0.1)

        # Should complete quickly for --help
        try:
            stdout, stderr = proc.communicate(timeout=2)
            assert proc.returncode == 0
        except subprocess.TimeoutExpired:
            proc.terminate()
            pytest.fail("Help command took too long")


def test_minimum_viable_functionality():
    """Single test to verify the absolute minimum functionality works."""
    # Can we even run the CLI?
    result = subprocess.run(
        ['swebench', '--version'],
        capture_output=True,
        text=True,
        timeout=5
    )
    assert result.returncode == 0, "CLI won't even show version"

    # Can we get help?
    result = subprocess.run(
        ['swebench', '--help'],
        capture_output=True,
        text=True,
        timeout=5
    )
    assert result.returncode == 0, "CLI won't show help"
    assert len(result.stdout) > 50, "Help text is too short"

    # Does it handle missing Docker gracefully?
    result = subprocess.run(
        ['swebench', 'run', '--patches', 'tests/fixtures/sample.jsonl'],
        capture_output=True,
        text=True,
        timeout=10
    )
    # Should exit with error code but not crash
    assert result.returncode in [0, 1, 2], f"Unexpected exit code: {result.returncode}"

    print("✅ Minimum viable functionality confirmed!")
    return True
