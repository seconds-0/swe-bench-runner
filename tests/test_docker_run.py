"""Tests for docker_run module."""

import json
import subprocess
from unittest.mock import Mock, patch

import pytest
from docker.errors import APIError

from swebench_runner.docker_run import (
    check_docker_running,
    check_resources,
    check_swebench_installed,
    create_predictions_file,
    detect_platform,
    install_swebench,
    load_first_patch,
    parse_harness_results,
    run_evaluation,
    run_swebench_harness,
)
from swebench_runner.models import EvaluationResult, Patch


class TestCheckDockerRunning:
    """Test Docker daemon availability checking."""

    @patch("swebench_runner.docker_run.docker.from_env")
    def test_docker_running_success(self, mock_from_env):
        """Test successful Docker connection."""
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_from_env.return_value = mock_client

        # Should not raise or exit
        check_docker_running()

        mock_from_env.assert_called_once()
        mock_client.ping.assert_called_once()

    @patch("swebench_runner.docker_run.docker.from_env")
    @patch("swebench_runner.docker_run.platform.system")
    def test_docker_not_running_macos(self, mock_system, mock_from_env):
        """Test Docker not running on macOS."""
        mock_system.return_value = "Darwin"
        mock_client = Mock()
        mock_client.ping.side_effect = APIError("Connection refused")
        mock_from_env.return_value = mock_client

        with pytest.raises(SystemExit) as exc_info:
            check_docker_running()

        assert exc_info.value.code == 2

    @patch("swebench_runner.docker_run.docker.from_env")
    @patch("swebench_runner.docker_run.platform.system")
    def test_docker_not_running_linux(self, mock_system, mock_from_env):
        """Test Docker not running on Linux."""
        mock_system.return_value = "Linux"
        mock_client = Mock()
        mock_client.ping.side_effect = APIError("Connection refused")
        mock_from_env.return_value = mock_client

        with pytest.raises(SystemExit) as exc_info:
            check_docker_running()

        assert exc_info.value.code == 2


class TestLoadFirstPatch:
    """Test patch loading functionality."""

    def test_load_valid_patch(self, tmp_path):
        """Test loading a valid patch."""
        patch_file = tmp_path / "test.jsonl"
        patch_content = {
            "instance_id": "test__repo-123",
            "patch": "diff --git a/file.py b/file.py\n+test"
        }
        patch_file.write_text(json.dumps(patch_content))

        patch = load_first_patch(str(patch_file))

        assert patch.instance_id == "test__repo-123"
        assert patch.patch == "diff --git a/file.py b/file.py\n+test"

    def test_load_multiple_patches_returns_first(self, tmp_path):
        """Test loading first patch from multiple patches."""
        patch_file = tmp_path / "test.jsonl"
        patch1 = {"instance_id": "first",
                  "patch": "diff --git a/file.py b/file.py\n+first"}
        patch2 = {"instance_id": "second",
                  "patch": "diff --git a/file.py b/file.py\n+second"}

        with patch_file.open("w") as f:
            json.dump(patch1, f)
            f.write("\n")
            json.dump(patch2, f)
            f.write("\n")

        patch = load_first_patch(str(patch_file))

        assert patch.instance_id == "first"

    def test_load_empty_file(self, tmp_path):
        """Test loading from empty file."""
        patch_file = tmp_path / "empty.jsonl"
        patch_file.write_text("")

        with pytest.raises(SystemExit) as exc_info:
            load_first_patch(str(patch_file))

        assert exc_info.value.code == 1

    def test_load_invalid_json(self, tmp_path):
        """Test loading invalid JSON."""
        patch_file = tmp_path / "invalid.jsonl"
        patch_file.write_text("invalid json")

        with pytest.raises(SystemExit) as exc_info:
            load_first_patch(str(patch_file))

        assert exc_info.value.code == 1

    def test_load_missing_keys(self, tmp_path):
        """Test loading JSON with missing required keys."""
        patch_file = tmp_path / "missing.jsonl"
        patch_content = {"instance_id": "test"}  # Missing patch key
        patch_file.write_text(json.dumps(patch_content))

        with pytest.raises(SystemExit) as exc_info:
            load_first_patch(str(patch_file))

        assert exc_info.value.code == 1

    def test_load_nonexistent_file(self):
        """Test loading non-existent file."""
        with pytest.raises(SystemExit) as exc_info:
            load_first_patch("nonexistent.jsonl")

        assert exc_info.value.code == 1


class TestCheckResources:
    """Test resource checking functionality."""

    def test_sufficient_resources(self):
        """Test with sufficient resources."""
        # Since psutil is optional and check_resources has try/except,
        # we'll just test that it doesn't raise an exception
        try:
            check_resources()
        except SystemExit:
            # If it exits, it should be due to insufficient resources
            pytest.fail("check_resources should not exit with sufficient resources")

    def test_insufficient_disk(self, monkeypatch):
        """Test with insufficient disk space."""
        # Override CI mode for this test
        monkeypatch.delenv("CI", raising=False)

        with patch("swebench_runner.docker_run.shutil.disk_usage") as mock_disk_usage:
            mock_disk_usage.return_value.free = 10 * 1024**3  # 10GB

            with pytest.raises(SystemExit) as exc_info:
                check_resources()

            assert exc_info.value.code == 4


class TestSWEBenchInstallation:
    """Test SWE-bench installation checking and installation."""

    @patch("swebench_runner.docker_run.subprocess.run")
    def test_swebench_installed(self, mock_run):
        """Test checking when SWE-bench is installed."""
        mock_run.return_value.returncode = 0

        assert check_swebench_installed() is True

    @patch("swebench_runner.docker_run.subprocess.run")
    def test_swebench_not_installed(self, mock_run):
        """Test checking when SWE-bench is not installed."""
        mock_run.return_value.returncode = 1

        assert check_swebench_installed() is False

    @patch("swebench_runner.docker_run.subprocess.run")
    def test_swebench_check_timeout(self, mock_run):
        """Test timeout during SWE-bench check."""
        mock_run.side_effect = subprocess.TimeoutExpired("cmd", 30)

        assert check_swebench_installed() is False

    @patch("swebench_runner.docker_run.subprocess.run")
    def test_install_swebench_success(self, mock_run):
        """Test successful SWE-bench installation."""
        mock_run.return_value.returncode = 0

        # Should not raise or exit
        install_swebench()

    @patch("swebench_runner.docker_run.subprocess.run")
    def test_install_swebench_failure(self, mock_run):
        """Test failed SWE-bench installation."""
        mock_run.return_value.returncode = 1
        mock_run.return_value.stderr = "Installation failed"

        with pytest.raises(SystemExit) as exc_info:
            install_swebench()

        assert exc_info.value.code == 1

    @patch("swebench_runner.docker_run.subprocess.run")
    def test_install_swebench_timeout(self, mock_run):
        """Test timeout during SWE-bench installation."""
        mock_run.side_effect = subprocess.TimeoutExpired("cmd", 300)

        with pytest.raises(SystemExit) as exc_info:
            install_swebench()

        assert exc_info.value.code == 1


class TestCreatePredictionsFile:
    """Test predictions file creation."""

    def test_create_predictions_file(self, tmp_path):
        """Test creating predictions file."""
        patch = Patch(
            instance_id="test__repo-123",
            patch="diff --git a/file.py b/file.py\n+test"
        )

        predictions_file = create_predictions_file(patch, tmp_path)

        assert predictions_file.exists()
        assert predictions_file.name == "predictions.jsonl"

        with predictions_file.open("r") as f:
            data = json.load(f)

        assert data["instance_id"] == "test__repo-123"
        assert data["model"] == "swebench-runner-mvp"
        assert data["prediction"] == "diff --git a/file.py b/file.py\n+test"


class TestDetectPlatform:
    """Test platform detection."""

    @patch("swebench_runner.docker_run.platform.machine")
    def test_detect_x86_64(self, mock_machine):
        """Test detecting x86_64 architecture."""
        mock_machine.return_value = "x86_64"

        assert detect_platform() == "x86_64"

    @patch("swebench_runner.docker_run.platform.machine")
    def test_detect_amd64(self, mock_machine):
        """Test detecting amd64 architecture."""
        mock_machine.return_value = "amd64"

        assert detect_platform() == "x86_64"

    @patch("swebench_runner.docker_run.platform.machine")
    def test_detect_arm64(self, mock_machine):
        """Test detecting arm64 architecture."""
        mock_machine.return_value = "arm64"

        assert detect_platform() == "arm64"

    @patch("swebench_runner.docker_run.platform.machine")
    def test_detect_unknown_architecture(self, mock_machine):
        """Test detecting unknown architecture."""
        mock_machine.return_value = "unknown"

        assert detect_platform() == "x86_64"  # Should default to x86_64


class TestRunSWEBenchHarness:
    """Test SWE-bench harness execution."""

    @patch("swebench_runner.docker_run.detect_platform")
    @patch("swebench_runner.docker_run.subprocess.run")
    def test_run_harness_x86_64(self, mock_run, mock_detect_platform, tmp_path):
        """Test running harness on x86_64."""
        mock_detect_platform.return_value = "x86_64"
        mock_run.return_value.returncode = 0

        patch = Patch("test__repo-123", "diff --git a/file.py b/file.py\n+test")
        predictions_file = tmp_path / "predictions.jsonl"
        predictions_file.write_text("{}")

        result = run_swebench_harness(predictions_file, tmp_path, patch)

        assert result.returncode == 0
        # Check that namespace was added for x86_64
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]  # First positional argument (command)
        assert "--namespace" in call_args
        assert "ghcr.io/epoch-research" in call_args

    @patch("swebench_runner.docker_run.detect_platform")
    @patch("swebench_runner.docker_run.subprocess.run")
    def test_run_harness_arm64(self, mock_run, mock_detect_platform, tmp_path):
        """Test running harness on ARM64."""
        mock_detect_platform.return_value = "arm64"
        mock_run.return_value.returncode = 0

        patch = Patch("test__repo-123", "diff --git a/file.py b/file.py\n+test")
        predictions_file = tmp_path / "predictions.jsonl"
        predictions_file.write_text("{}")

        result = run_swebench_harness(predictions_file, tmp_path, patch)

        assert result.returncode == 0
        # Check that namespace was NOT added for ARM64
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]  # First positional argument (command)
        assert "--namespace" not in call_args


class TestParseHarnessResults:
    """Test parsing harness results."""

    def test_parse_successful_result(self, tmp_path):
        """Test parsing successful evaluation result."""
        patch = Patch("test__repo-123", "diff --git a/file.py b/file.py\n+test")

        # Create results directory and file
        results_dir = tmp_path / "evaluation_results"
        results_dir.mkdir()

        result_file = results_dir / "results.json"
        result_data = {
            "test__repo-123": {
                "resolved": True,
                "error": None
            }
        }
        result_file.write_text(json.dumps(result_data))

        result = parse_harness_results(tmp_path, patch)

        assert result.instance_id == "test__repo-123"
        assert result.passed is True
        assert result.error is None

    def test_parse_failed_result(self, tmp_path):
        """Test parsing failed evaluation result."""
        patch = Patch("test__repo-123", "diff --git a/file.py b/file.py\n+test")

        # Create results directory and file
        results_dir = tmp_path / "evaluation_results"
        results_dir.mkdir()

        result_file = results_dir / "results.json"
        result_data = {
            "test__repo-123": {
                "resolved": False,
                "error": "Test failed"
            }
        }
        result_file.write_text(json.dumps(result_data))

        result = parse_harness_results(tmp_path, patch)

        assert result.instance_id == "test__repo-123"
        assert result.passed is False
        assert result.error == "Test failed"

    def test_parse_missing_results_dir(self, tmp_path):
        """Test parsing when results directory doesn't exist."""
        patch = Patch("test__repo-123", "diff --git a/file.py b/file.py\n+test")

        result = parse_harness_results(tmp_path, patch)

        assert result.instance_id == "test__repo-123"
        assert result.passed is False
        assert "No evaluation results directory found" in result.error

    def test_parse_no_result_files(self, tmp_path):
        """Test parsing when no result files exist."""
        patch = Patch("test__repo-123", "diff --git a/file.py b/file.py\n+test")

        # Create empty results directory
        results_dir = tmp_path / "evaluation_results"
        results_dir.mkdir()

        result = parse_harness_results(tmp_path, patch)

        assert result.instance_id == "test__repo-123"
        assert result.passed is False
        assert "No result files found" in result.error

    def test_parse_instance_not_in_results(self, tmp_path):
        """Test parsing when instance is not in results."""
        patch = Patch("test__repo-123", "diff --git a/file.py b/file.py\n+test")

        # Create results directory and file with different instance
        results_dir = tmp_path / "evaluation_results"
        results_dir.mkdir()

        result_file = results_dir / "results.json"
        result_data = {
            "other__repo-456": {
                "resolved": True,
                "error": None
            }
        }
        result_file.write_text(json.dumps(result_data))

        result = parse_harness_results(tmp_path, patch)

        assert result.instance_id == "test__repo-123"
        assert result.passed is False
        assert "not found in results" in result.error


class TestRunEvaluation:
    """Test full evaluation workflow."""

    @patch("swebench_runner.docker_run.check_docker_running")
    @patch("swebench_runner.docker_run.check_resources")
    @patch("swebench_runner.docker_run.check_swebench_installed")
    @patch("swebench_runner.docker_run.run_swebench_harness")
    @patch("swebench_runner.docker_run.parse_harness_results")
    def test_run_evaluation_success(self, mock_parse, mock_run_harness,
                                   mock_check_swebench, mock_check_resources,
                                   mock_check_docker, tmp_path):
        """Test successful evaluation run."""
        # Setup mocks
        mock_check_swebench.return_value = True
        mock_run_harness.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        mock_parse.return_value = EvaluationResult(
            instance_id="test__repo-123",
            passed=True,
            error=None
        )

        # Create test patch file
        patch_file = tmp_path / "test.jsonl"
        patch_content = {
            "instance_id": "test__repo-123",
            "patch": "diff --git a/file.py b/file.py\n+test"
        }
        patch_file.write_text(json.dumps(patch_content))

        result = run_evaluation(str(patch_file))

        assert result.instance_id == "test__repo-123"
        assert result.passed is True
        assert result.error is None

    @patch("swebench_runner.docker_run.check_docker_running")
    @patch("swebench_runner.docker_run.check_resources")
    def test_run_evaluation_large_patch(self, mock_check_resources,
                                       mock_check_docker, tmp_path):
        """Test evaluation with large patch."""
        # Create test patch file with patch larger than default 5MB limit
        patch_file = tmp_path / "large.jsonl"
        large_patch = "diff --git a/file.py b/file.py\n+" + "x" * (6 * 1024 * 1024)  # 6MB
        patch_content = {
            "instance_id": "test__repo-123",
            "patch": large_patch
        }
        patch_file.write_text(json.dumps(patch_content))

        result = run_evaluation(str(patch_file))

        assert result.instance_id == "test__repo-123"
        assert result.passed is False
        assert "exceeds" in result.error and "limit" in result.error

    @patch("swebench_runner.docker_run.check_docker_running")
    @patch("swebench_runner.docker_run.check_resources")
    @patch("swebench_runner.docker_run.check_swebench_installed")
    @patch("swebench_runner.docker_run.install_swebench")
    def test_run_evaluation_install_swebench(self, mock_install, mock_check_swebench,
                                            mock_check_resources, mock_check_docker,
                                            tmp_path):
        """Test evaluation with SWE-bench installation."""
        # SWE-bench not installed initially
        mock_check_swebench.return_value = False

        # Create test patch file
        patch_file = tmp_path / "test.jsonl"
        patch_content = {
            "instance_id": "test__repo-123",
            "patch": "diff --git a/file.py b/file.py\n+test"
        }
        patch_file.write_text(json.dumps(patch_content))

        with patch("swebench_runner.docker_run.run_swebench_harness") as mock_harness:
            mock_harness.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="", stderr=""
            )

            with patch("swebench_runner.docker_run.parse_harness_results") as mock_p:
                mock_p.return_value = EvaluationResult(
                    instance_id="test__repo-123",
                    passed=True,
                    error=None
                )

                result = run_evaluation(str(patch_file))

        # Should have installed SWE-bench
        mock_install.assert_called_once()
        assert result.passed is True
