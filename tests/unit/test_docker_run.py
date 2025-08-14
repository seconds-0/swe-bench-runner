"""Unit tests for docker_run.py - focus on resource checking and error handling.

Tests specifically for:
- Resource checking with mocked psutil
- Patch loading with various formats
- Error classification and exit codes
- Platform detection
- SWE-bench harness integration
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from swebench_runner import exit_codes
from swebench_runner.docker_run import (
    _python_preflight_ok,
    _resolve_python_interpreter,
    check_resources,
    check_swebench_installed,
    create_predictions_file,
    detect_platform,
    install_swebench,
    load_all_patches,
    load_first_patch,
    parse_harness_results,
    run_swebench_harness,
)
from swebench_runner.models import Patch


class TestDockerRunResourceChecking:
    """Test resource checking to prevent failures."""

    @patch('swebench_runner.docker_run.shutil.disk_usage')
    @patch('swebench_runner.docker_run.psutil')
    def test_checks_minimum_memory_requirements(self, mock_psutil, mock_disk):
        """Check minimum memory before running.

        Why: Insufficient memory causes Docker containers to crash with OOM.
        """
        # Mock 4GB available (below default 8GB requirement)
        mock_psutil.virtual_memory.return_value = Mock(available=4 * 1024**3)
        mock_disk.return_value = Mock(free=100 * 1024**3)

        with pytest.raises(SystemExit) as exc_info:
            check_resources()

        # Should exit with resource error
        assert exc_info.value.code == exit_codes.RESOURCE_ERROR

    @patch('swebench_runner.docker_run.shutil.disk_usage')
    @patch('swebench_runner.docker_run.psutil')
    def test_checks_minimum_disk_space(self, mock_psutil, mock_disk):
        """Check disk space before downloading images.

        Why: Docker images are large (14GB+) and need sufficient space.
        """
        mock_psutil.virtual_memory.return_value = Mock(available=16 * 1024**3)
        # Mock 20GB free (below default 50GB requirement)
        mock_disk.return_value = Mock(free=20 * 1024**3)

        with pytest.raises(SystemExit) as exc_info:
            check_resources()

        assert exc_info.value.code == exit_codes.RESOURCE_ERROR

    @patch('swebench_runner.docker_run.shutil.disk_usage')
    @patch('swebench_runner.docker_run.psutil')
    @patch.dict('os.environ', {'CI': 'true'})
    def test_ci_mode_has_lower_requirements(self, mock_psutil, mock_disk):
        """CI mode should have lower resource requirements.

        Why: CI environments have limited resources but controlled workloads.
        """
        # Mock 4GB RAM and 20GB disk (OK for CI)
        mock_psutil.virtual_memory.return_value = Mock(available=4 * 1024**3)
        mock_disk.return_value = Mock(free=20 * 1024**3)

        # Should not raise in CI mode
        check_resources()  # No exception = pass

    @patch('swebench_runner.docker_run.shutil.disk_usage')
    @patch.dict('os.environ', {'SWEBENCH_SKIP_RESOURCE_CHECK': 'true'})
    def test_can_skip_resource_checks(self, mock_disk):
        """Allow skipping resource checks via environment.

        Why: Some environments need to bypass checks.
        """
        # Don't even set up psutil mock - should be skipped
        check_resources()  # Should not raise


class TestDockerRunPatchLoading:
    """Test patch loading from various sources."""

    def test_loads_patch_from_jsonl_file(self):
        """Load patches from JSONL format.

        Why: JSONL is the standard format for SWE-bench patches.
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            json.dump({"instance_id": "test-123", "patch": "diff content"}, f)
            f.write('\n')
            f.flush()

            patch = load_first_patch(f.name)

            assert patch.instance_id == "test-123"
            assert patch.patch == "diff content"

        Path(f.name).unlink()

    def test_loads_patch_from_directory(self):
        """Load patches from directory of .patch files.

        Why: Users may have patches as separate files.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            patch_file = Path(tmpdir) / "test-123.patch"
            patch_file.write_text("diff content")

            patch = load_first_patch(tmpdir)

            assert patch.instance_id == "test-123"
            assert patch.patch == "diff content"

    def test_validates_patch_size_limits(self):
        """Enforce patch size limits to prevent memory issues.

        Why: Huge patches can exhaust memory and fail in Docker.
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            # Create 10MB patch (over default 5MB limit)
            huge_patch = "x" * (10 * 1024 * 1024)
            json.dump({"instance_id": "test", "patch": huge_patch}, f)
            f.write('\n')
            f.flush()

            with pytest.raises(SystemExit) as exc_info:
                load_first_patch(f.name, max_size_mb=5)

            assert exc_info.value.code == exit_codes.GENERAL_ERROR

        Path(f.name).unlink()

    def test_handles_invalid_json_gracefully(self):
        """Handle malformed JSON without crashing.

        Why: User files may be corrupted or manually edited.
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"invalid": json syntax}')
            f.flush()

            with pytest.raises(SystemExit) as exc_info:
                load_first_patch(f.name)

            assert exc_info.value.code == exit_codes.GENERAL_ERROR

        Path(f.name).unlink()

    def test_handles_missing_patch_file(self):
        """Handle missing files with clear error.

        Why: Users may specify wrong paths.
        """
        with pytest.raises(SystemExit) as exc_info:
            load_first_patch("/nonexistent/file.jsonl")

        assert exc_info.value.code == exit_codes.GENERAL_ERROR

    def test_handles_empty_patch_file(self):
        """Handle empty patch files.

        Why: Files may be created but not populated.
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.flush()  # Empty file

            with pytest.raises(SystemExit) as exc_info:
                load_first_patch(f.name)

            assert exc_info.value.code == exit_codes.GENERAL_ERROR

        Path(f.name).unlink()


class TestDockerRunPlatformDetection:
    """Test platform detection for Docker images."""

    @patch('platform.machine')
    def test_detects_x86_64_architecture(self, mock_machine):
        """Detect x86_64 for Intel/AMD systems.

        Why: Different architectures need different Docker images.
        """
        mock_machine.return_value = "x86_64"

        arch = detect_platform()

        assert arch == "x86_64"

    @patch('platform.machine')
    def test_detects_arm64_architecture(self, mock_machine):
        """Detect ARM64 for Apple Silicon and ARM servers.

        Why: ARM systems need ARM-compatible images.
        """
        mock_machine.return_value = "arm64"

        arch = detect_platform()

        assert arch == "arm64"

    @patch('platform.machine')
    def test_defaults_to_x86_for_unknown(self, mock_machine):
        """Default to x86_64 for unknown architectures.

        Why: Better to try x86 with emulation than fail completely.
        """
        mock_machine.return_value = "riscv64"  # Unsupported

        arch = detect_platform()

        assert arch == "x86_64"  # Should default


class TestDockerRunSWEBenchIntegration:
    """Test SWE-bench harness integration."""

    @patch('subprocess.run')
    def test_checks_swebench_installation(self, mock_run):
        """Check if SWE-bench harness is installed.

        Why: Need to install if missing to avoid runtime failures.
        """
        mock_run.return_value = Mock(returncode=0)

        installed = check_swebench_installed()

        assert installed is True
        # Should run with Python interpreter
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert sys.executable in call_args
        assert "swebench.harness.run_evaluation" in call_args

    @patch('subprocess.run')
    def test_installs_swebench_when_missing(self, mock_run):
        """Install SWE-bench when not found.

        Why: Automatic installation improves user experience.
        """
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

        install_swebench()

        # Should use pip to install
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert sys.executable in call_args
        assert "pip" in call_args
        assert "swebench" in call_args

    @patch('subprocess.run')
    def test_handles_network_errors_during_install(self, mock_run):
        """Detect network errors during installation.

        Why: Network issues need clear error messages.
        """
        mock_run.return_value = Mock(
            returncode=1,
            stderr="Connection timeout"
        )

        with pytest.raises(SystemExit) as exc_info:
            install_swebench()

        assert exc_info.value.code == exit_codes.NETWORK_ERROR

    def test_creates_valid_predictions_file(self):
        """Create predictions file for harness.

        Why: Harness requires specific JSON format.
        """
        patch = Patch(instance_id="test-123", patch="diff content")

        with tempfile.TemporaryDirectory() as tmpdir:
            pred_file = create_predictions_file(patch, Path(tmpdir))

            assert pred_file.exists()

            # Verify content
            with open(pred_file) as f:
                data = json.loads(f.readline())
                assert data["instance_id"] == "test-123"
                assert data["model_patch"] == "diff content"
                assert "model_name_or_path" in data

    @patch('subprocess.run')
    def test_runs_harness_with_correct_arguments(self, mock_run):
        """Run harness with proper arguments.

        Why: Incorrect arguments cause harness to fail silently.
        """
        # Disable progress tracking for tests
        os.environ["SWEBENCH_DISABLE_PROGRESS"] = "1"
        
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

        patch = Patch(instance_id="test-123", patch="diff")
        with tempfile.TemporaryDirectory() as tmpdir:
            pred_file = Path(tmpdir) / "pred.jsonl"
            pred_file.write_text('{"instance_id": "test-123"}')

            run_swebench_harness(pred_file, Path(tmpdir), patch)

            # Check arguments
            assert mock_run.called, "subprocess.run should have been called"
            call_args = mock_run.call_args[0][0]
            assert "--dataset_name" in call_args
            assert "SWE-bench_Lite" in call_args
            assert "--max_workers" in call_args
            assert "1" in call_args  # Single worker for determinism
        
        # Clean up env var
        os.environ.pop("SWEBENCH_DISABLE_PROGRESS", None)

    def test_parses_successful_harness_results(self):
        """Parse successful evaluation results.

        Why: Need to extract pass/fail status accurately.
        """
        patch = Patch(instance_id="test-123", patch="diff")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock results
            results_dir = Path(tmpdir) / "evaluation_results"
            results_dir.mkdir()

            results_file = results_dir / "results.json"
            results_file.write_text(json.dumps({
                "test-123": {
                    "resolved": True,
                    "error": None
                }
            }))

            result = parse_harness_results(Path(tmpdir), patch)

            assert result.instance_id == "test-123"
            assert result.passed is True
            assert result.error is None

    def test_parses_failed_harness_results(self):
        """Parse failed evaluation results.

        Why: Need to report failures with error details.
        """
        patch = Patch(instance_id="test-123", patch="diff")

        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir) / "evaluation_results"
            results_dir.mkdir()

            results_file = results_dir / "results.json"
            results_file.write_text(json.dumps({
                "test-123": {
                    "resolved": False,
                    "error": "Tests failed: 5 failures"
                }
            }))

            result = parse_harness_results(Path(tmpdir), patch)

            assert result.instance_id == "test-123"
            assert result.passed is False
            assert "Tests failed" in result.error


class TestPythonInterpreterResolution:
    def test_python_preflight_rejects_old_python(self, monkeypatch):
        def fake_run(args, capture_output=True, text=True, timeout=10):  # noqa: ARG001
            class R:
                def __init__(self, code):
                    self.returncode = code
            # First call checks version -> fail
            if isinstance(args, list) and "sys.version_info>=(3,10)" in args[-1]:
                return R(1)
            return R(0)

        monkeypatch.setattr("swebench_runner.docker_run.subprocess.run", fake_run)
        assert _python_preflight_ok("/usr/bin/python3.9") is False

    def test_python_preflight_accepts_modern_python(self, monkeypatch):
        def fake_run(args, capture_output=True, text=True, timeout=10):  # noqa: ARG001
            class R:
                def __init__(self):
                    self.returncode = 0
            return R()

        monkeypatch.setattr("swebench_runner.docker_run.subprocess.run", fake_run)
        assert _python_preflight_ok("/usr/bin/python3.11") is True

    def test_resolve_python_honors_override(self, monkeypatch):
        monkeypatch.setenv("SWEBENCH_PYTHON", "/custom/python")
        monkeypatch.setattr("swebench_runner.docker_run._python_preflight_ok", lambda p: p == "/custom/python")
        assert _resolve_python_interpreter() == "/custom/python"


class TestDockerRunErrorClassification:
    """Test error classification for proper exit codes."""

    def test_classifies_docker_errors_correctly(self):
        """Classify Docker errors with correct exit code.

        Why: Different errors need different handling strategies.
        """
        # Test via docker_client module since that's where classification happens
        from docker.errors import APIError

        from swebench_runner.docker_client import check_docker_running

        with patch('swebench_runner.docker_client.get_docker_client') as mock_get:
            mock_client = Mock()
            mock_client.ping.side_effect = APIError("Connection refused")
            mock_get.return_value = mock_client

            with pytest.raises(SystemExit) as exc_info:
                check_docker_running()

            assert exc_info.value.code == exit_codes.DOCKER_NOT_FOUND

    @patch('subprocess.run')
    def test_classifies_network_errors_correctly(self, mock_run):
        """Classify network errors during operations.

        Why: Network errors may be transient and worth retrying.
        """
        mock_run.return_value = Mock(
            returncode=1,
            stderr="Failed to pull: connection timeout"
        )

        # Network errors in docker_run context
        with pytest.raises(SystemExit) as exc_info:
            install_swebench()  # This checks for network errors

        assert exc_info.value.code == exit_codes.NETWORK_ERROR


class TestDockerRunBatchOperations:
    """Test batch patch processing."""

    def test_loads_all_patches_from_file(self):
        """Load all patches from JSONL file.

        Why: Batch processing needs to handle multiple patches.
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"instance_id": "test-1", "patch": "diff1"}\n')
            f.write('{"instance_id": "test-2", "patch": "diff2"}\n')
            f.write('{"instance_id": "test-3", "patch": "diff3"}\n')
            f.flush()

            patches = load_all_patches(f.name)

            assert len(patches) == 3
            assert patches[0].instance_id == "test-1"
            assert patches[2].patch == "diff3"

        Path(f.name).unlink()

    def test_skips_invalid_patches_in_batch(self):
        """Skip invalid patches without failing entire batch.

        Why: One bad patch shouldn't stop processing others.
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"instance_id": "test-1", "patch": "good"}\n')
            f.write('{"invalid": "json"}\n')  # Missing required fields
            f.write('{"instance_id": "test-3", "patch": "good"}\n')
            f.flush()

            patches = load_all_patches(f.name)

            # Should skip invalid, load valid
            assert len(patches) == 2
            assert patches[0].instance_id == "test-1"
            assert patches[1].instance_id == "test-3"

        Path(f.name).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
