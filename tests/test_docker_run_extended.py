"""Extended tests for docker_run module to improve coverage."""

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from docker.errors import APIError

from swebench_runner import docker_run, exit_codes
from swebench_runner.models import Patch


class TestDockerErrorHandling:
    """Test Docker error handling scenarios."""

    @patch("docker.from_env")
    def test_check_docker_network_error(self, mock_from_env):
        """Test network error during Docker check."""
        mock_client = Mock()
        mock_from_env.return_value = mock_client
        mock_client.ping.side_effect = APIError("Network timeout occurred")

        with pytest.raises(SystemExit) as exc_info:
            docker_run.check_docker_running()

        assert exc_info.value.code == exit_codes.NETWORK_ERROR

    @patch("docker.from_env")
    def test_check_docker_dns_error(self, mock_from_env):
        """Test DNS resolution error."""
        mock_client = Mock()
        mock_from_env.return_value = mock_client
        mock_client.ping.side_effect = Exception("Could not resolve DNS")

        with pytest.raises(SystemExit) as exc_info:
            docker_run.check_docker_running()

        assert exc_info.value.code == exit_codes.NETWORK_ERROR

    @patch("docker.from_env")
    def test_check_docker_generic_api_error(self, mock_from_env):
        """Test generic Docker API error."""
        mock_client = Mock()
        mock_from_env.return_value = mock_client
        mock_client.ping.side_effect = APIError("Unknown Docker error")

        with pytest.raises(SystemExit) as exc_info:
            docker_run.check_docker_running()

        assert exc_info.value.code == exit_codes.DOCKER_NOT_FOUND


class TestPatchLoading:
    """Test patch loading from various sources."""

    def test_load_patch_from_directory(self, tmp_path):
        """Test loading patch from directory."""
        patch_dir = tmp_path / "patches"
        patch_dir.mkdir()

        patch_file = patch_dir / "test-001.patch"
        patch_content = """diff --git a/test.py b/test.py
--- a/test.py
+++ b/test.py
@@ -1 +1 @@
-old
+new"""
        patch_file.write_text(patch_content)

        patch = docker_run.load_first_patch(str(patch_dir))

        assert patch.instance_id == "test-001"
        assert patch.patch == patch_content

    def test_load_patch_empty_directory(self, tmp_path):
        """Test loading from empty directory."""
        patch_dir = tmp_path / "empty_patches"
        patch_dir.mkdir()

        with pytest.raises(SystemExit) as exc_info:
            docker_run.load_first_patch(str(patch_dir))

        assert exc_info.value.code == exit_codes.GENERAL_ERROR

    def test_load_patch_invalid_json(self, tmp_path):
        """Test loading invalid JSON patch file."""
        patch_file = tmp_path / "invalid.jsonl"
        patch_file.write_text("not valid json")

        with pytest.raises(SystemExit) as exc_info:
            docker_run.load_first_patch(str(patch_file))

        assert exc_info.value.code == exit_codes.GENERAL_ERROR

    def test_load_patch_missing_required_fields(self, tmp_path):
        """Test loading patch with missing fields."""
        patch_file = tmp_path / "incomplete.jsonl"
        patch_file.write_text('{"instance_id": "test"}')  # Missing patch field

        with pytest.raises(SystemExit) as exc_info:
            docker_run.load_first_patch(str(patch_file))

        assert exc_info.value.code == exit_codes.GENERAL_ERROR

    def test_load_patch_unicode_error(self, tmp_path):
        """Test loading patch with invalid encoding."""
        patch_file = tmp_path / "bad_encoding.jsonl"
        # Write invalid UTF-8 bytes
        patch_file.write_bytes(b'\xff\xfe Invalid UTF-8')

        with pytest.raises(SystemExit) as exc_info:
            docker_run.load_first_patch(str(patch_file))

        assert exc_info.value.code == exit_codes.GENERAL_ERROR


class TestResourceChecking:
    """Test resource checking with various configurations."""

    @patch.dict(os.environ, {"CI": "true", "SWEBENCH_CI_MIN_MEMORY_GB": "2"})
    @patch("psutil.virtual_memory")
    def test_ci_mode_custom_memory_requirement(self, mock_memory):
        """Test CI mode with custom memory requirement."""
        mock_memory.return_value = Mock(available=1.5 * 1024**3)  # 1.5GB

        # Should only warn in CI mode
        docker_run.check_resources()  # Should not raise

    @patch.dict(os.environ, {"SWEBENCH_MIN_DISK_GB": "100", "CI": "false"})
    @patch("shutil.disk_usage")
    def test_custom_disk_requirement(self, mock_disk):
        """Test custom disk requirement."""
        mock_disk.return_value = Mock(free=50 * 1024**3)  # 50GB

        with pytest.raises(SystemExit) as exc_info:
            docker_run.check_resources()

        assert exc_info.value.code == exit_codes.RESOURCE_ERROR

    @patch("psutil.virtual_memory", side_effect=ImportError)
    @patch("shutil.disk_usage")
    def test_psutil_not_available(self, mock_disk, mock_memory):
        """Test when psutil is not available."""
        mock_disk.return_value = Mock(free=100 * 1024**3)

        # Should not raise even without psutil
        docker_run.check_resources()

    @patch("shutil.disk_usage", side_effect=Exception("Disk error"))
    def test_disk_check_failure(self, mock_disk):
        """Test when disk check fails."""
        # Should not raise even with disk check failure
        docker_run.check_resources()


class TestSWEBenchInstallation:
    """Test SWE-bench harness installation."""

    @patch("subprocess.run")
    def test_install_swebench_generic_error(self, mock_run):
        """Test generic error during installation."""
        mock_run.return_value = Mock(
            returncode=1,
            stderr="Generic installation error"
        )

        with pytest.raises(SystemExit) as exc_info:
            docker_run.install_swebench()

        assert exc_info.value.code == exit_codes.GENERAL_ERROR

    @patch("subprocess.run", side_effect=Exception("Unexpected error"))
    def test_install_swebench_exception(self, mock_run):
        """Test unexpected exception during installation."""
        with pytest.raises(SystemExit) as exc_info:
            docker_run.install_swebench()

        assert exc_info.value.code == exit_codes.GENERAL_ERROR


class TestPlatformDetection:
    """Test platform detection for Docker images."""

    @patch("platform.machine")
    def test_detect_arm64_platform(self, mock_machine):
        """Test ARM64 platform detection."""
        mock_machine.return_value = "aarch64"
        assert docker_run.detect_platform() == "arm64"

    @patch("platform.machine")
    def test_detect_unknown_platform(self, mock_machine):
        """Test unknown platform handling."""
        mock_machine.return_value = "unknown_arch"
        assert docker_run.detect_platform() == "x86_64"  # Default


class TestHarnessExecution:
    """Test SWE-bench harness execution."""

    def test_create_predictions_file(self, tmp_path):
        """Test creating predictions file."""
        patch = Patch(instance_id="test-123", patch="test patch")

        pred_file = docker_run.create_predictions_file(patch, tmp_path)

        assert pred_file.exists()
        with pred_file.open() as f:
            data = json.load(f)
            assert data["instance_id"] == "test-123"
            assert data["model"] == "swebench-runner-mvp"
            assert data["prediction"] == "test patch"

    @patch("subprocess.run")
    def test_run_harness_timeout(self, mock_run):
        """Test harness timeout handling."""
        mock_run.side_effect = subprocess.TimeoutExpired("cmd", 4200)

        patch = Patch(instance_id="test-123", patch="test")
        pred_file = Path("/tmp/predictions.jsonl")

        with pytest.raises(subprocess.TimeoutExpired):
            docker_run.run_swebench_harness(pred_file, Path("/tmp"), patch)

    def test_parse_results_no_directory(self, tmp_path):
        """Test parsing results when directory doesn't exist."""
        patch = Patch(instance_id="test-123", patch="test")

        result = docker_run.parse_harness_results(tmp_path, patch)

        assert not result.passed
        assert "No evaluation results directory" in result.error

    def test_parse_results_no_files(self, tmp_path):
        """Test parsing results with empty results directory."""
        patch = Patch(instance_id="test-123", patch="test")
        results_dir = tmp_path / "evaluation_results"
        results_dir.mkdir()

        result = docker_run.parse_harness_results(tmp_path, patch)

        assert not result.passed
        assert "No result files found" in result.error

    def test_parse_results_instance_not_found(self, tmp_path):
        """Test parsing results when instance not in results."""
        patch = Patch(instance_id="test-123", patch="test")
        results_dir = tmp_path / "evaluation_results"
        results_dir.mkdir()

        result_file = results_dir / "results.json"
        result_file.write_text('{"other-instance": {"resolved": true}}')

        result = docker_run.parse_harness_results(tmp_path, patch)

        assert not result.passed
        assert "not found in results" in result.error

    def test_parse_results_json_error(self, tmp_path):
        """Test parsing results with invalid JSON."""
        patch = Patch(instance_id="test-123", patch="test")
        results_dir = tmp_path / "evaluation_results"
        results_dir.mkdir()

        result_file = results_dir / "results.json"
        result_file.write_text('invalid json')

        result = docker_run.parse_harness_results(tmp_path, patch)

        assert not result.passed
        assert "Failed to parse results" in result.error


class TestRunEvaluationIntegration:
    """Test full evaluation flow integration."""

    @patch("swebench_runner.docker_run.check_docker_running")
    @patch("swebench_runner.docker_run.check_resources")
    @patch("swebench_runner.docker_run.check_swebench_installed")
    @patch("subprocess.run")
    def test_evaluation_network_error_in_harness(self, mock_run, mock_installed,
                                                mock_resources, mock_docker, tmp_path):
        """Test network error during harness execution."""
        mock_installed.return_value = True

        # Create test patch
        patch_file = tmp_path / "test.jsonl"
        patch_file.write_text(
            '{"instance_id": "test", "patch": "diff --git a/test.py b/test.py\\n+fix"}'
        )

        # Mock harness failure with network error
        mock_run.return_value = Mock(
            returncode=1,
            stderr="Failed to pull image from registry"
        )

        result = docker_run.run_evaluation(str(patch_file))

        assert not result.passed
        assert "Network error" in result.error

    @patch("swebench_runner.docker_run.check_docker_running")
    @patch("swebench_runner.docker_run.check_resources")
    @patch("swebench_runner.docker_run.check_swebench_installed")
    @patch("subprocess.run")
    def test_evaluation_generic_exception(self, mock_run, mock_installed,
                                        mock_resources, mock_docker, tmp_path):
        """Test generic exception during evaluation."""
        mock_installed.return_value = True

        # Create test patch
        patch_file = tmp_path / "test.jsonl"
        patch_file.write_text(
            '{"instance_id": "test", "patch": "diff --git a/test.py b/test.py\\n+fix"}'
        )

        # Mock unexpected exception
        mock_run.side_effect = Exception("Unexpected error")

        result = docker_run.run_evaluation(str(patch_file))

        assert not result.passed
        assert "Evaluation failed" in result.error
