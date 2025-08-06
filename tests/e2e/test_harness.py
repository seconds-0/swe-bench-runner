"""Test harness for consistent E2E test environments.

This module provides a TestHarness class that creates isolated, reproducible
test environments for E2E testing of the SWE-bench Runner CLI.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class SWEBenchTestHarness:
    """Provides consistent test environment for E2E tests.

    This harness ensures:
    - Isolated temporary directories
    - Controlled environment variables
    - Process lifecycle management
    - Proper cleanup on teardown
    """

    def __init__(self):
        """Initialize the test harness."""
        self.temp_dir: Optional[Path] = None
        self.original_env: Dict[str, str] = {}
        self.env_vars: Dict[str, str] = {}
        self.processes: List[subprocess.Popen] = []
        self.created_files: List[Path] = []
        self.created_dirs: List[Path] = []

    def setup(self, test_name: str = "test") -> Path:
        """Create isolated test environment.

        Args:
            test_name: Name of the test for directory naming

        Returns:
            Path to the temporary test directory
        """
        # Store original environment
        self.original_env = os.environ.copy()

        # Create temp directory
        self.temp_dir = Path(tempfile.mkdtemp(prefix=f"swebench_test_{test_name}_"))
        self.created_dirs.append(self.temp_dir)

        # Set up isolated environment variables
        self.env_vars = {
            "SWEBENCH_TEST_MODE": "true",
            "SWEBENCH_CACHE_DIR": str(self.temp_dir / "cache"),
            "SWEBENCH_MOCK_NO_DOCKER": "true",  # Default to no Docker for safety
            "PYTHONPATH": str(Path.cwd()),
            "NO_COLOR": "1",  # Disable color output for predictable assertions
        }

        # Create cache directory
        cache_dir = self.temp_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        return self.temp_dir

    def teardown(self):
        """Clean up all resources created during testing."""
        # Kill any running processes
        for proc in self.processes:
            if proc.poll() is None:  # Still running
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()

        # Clean up created files
        for file_path in self.created_files:
            if file_path.exists():
                try:
                    file_path.unlink()
                except Exception:
                    pass  # Best effort cleanup

        # Clean up created directories
        for dir_path in reversed(self.created_dirs):  # Reverse to delete children first
            if dir_path.exists():
                try:
                    shutil.rmtree(dir_path)
                except Exception:
                    pass  # Best effort cleanup

        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)

    def run_cli(
        self,
        args: List[str],
        env: Optional[Dict[str, str]] = None,
        timeout: int = 30,
        input_text: Optional[str] = None,
        check_returncode: bool = False
    ) -> Tuple[int, str, str]:
        """Run CLI command with proper isolation.

        Args:
            args: CLI arguments (e.g., ["run", "--patches", "file.jsonl"])
            env: Additional environment variables
            timeout: Command timeout in seconds
            input_text: Optional input to send to stdin
            check_returncode: Whether to raise on non-zero return code

        Returns:
            Tuple of (returncode, stdout, stderr)
        """
        # Build command
        cmd = [sys.executable, "-m", "swebench_runner"] + args

        # Merge environment variables
        run_env = self.env_vars.copy()
        if env:
            run_env.update(env)

        # Add to existing environment (don't replace completely)
        full_env = os.environ.copy()
        full_env.update(run_env)

        # Run command
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=full_env,
                input=input_text,
                cwd=str(self.temp_dir) if self.temp_dir else None
            )

            if check_returncode and result.returncode != 0:
                raise subprocess.CalledProcessError(
                    result.returncode, cmd, result.stdout, result.stderr
                )

            return result.returncode, result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            # Return timeout as special exit code
            return -1, "", f"Command timed out after {timeout} seconds"

    def create_patch_file(
        self,
        patches: List[Dict[str, str]],
        filename: str = "patches.jsonl"
    ) -> Path:
        """Create a JSONL patch file for testing.

        Args:
            patches: List of patch dictionaries with 'instance_id' and 'patch' keys
            filename: Name of the file to create

        Returns:
            Path to the created file
        """
        file_path = self.temp_dir / filename

        with open(file_path, 'w') as f:
            for patch in patches:
                f.write(json.dumps(patch) + '\n')

        self.created_files.append(file_path)
        return file_path

    def create_minimal_patch(self, instance_id: str = "test-instance") -> Path:
        """Create a minimal valid patch file for testing.

        Args:
            instance_id: Instance ID for the patch

        Returns:
            Path to the created patch file
        """
        patch_content = {
            "instance_id": instance_id,
            "patch": "diff --git a/test.py b/test.py\n"
                    "--- a/test.py\n"
                    "+++ b/test.py\n"
                    "@@ -1,1 +1,1 @@\n"
                    "-print('old')\n"
                    "+print('new')\n"
        }
        return self.create_patch_file([patch_content], "minimal.jsonl")

    def create_dataset_cache(self, dataset_name: str = "lite") -> Path:
        """Create a mock dataset cache for testing.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Path to the dataset cache directory
        """
        dataset_dir = Path(self.temp_dir) / "cache" / "datasets" / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Create a minimal dataset file
        dataset_file = dataset_dir / "data.json"
        with open(dataset_file, 'w') as f:
            json.dump([
                {
                    "instance_id": "django__django-11001",
                    "problem_statement": "Test problem",
                    "base_commit": "abc123"
                }
            ], f)

        self.created_files.append(dataset_file)
        return dataset_dir

    def assert_cli_success(
        self,
        args: List[str],
        expected_output: Optional[str] = None,
        env: Optional[Dict[str, str]] = None
    ):
        """Assert that a CLI command succeeds.

        Args:
            args: CLI arguments
            expected_output: Optional string that should appear in stdout
            env: Additional environment variables
        """
        returncode, stdout, stderr = self.run_cli(args, env=env)

        assert returncode == 0, (
            f"CLI command failed with code {returncode}\n"
            f"stdout: {stdout}\n"
            f"stderr: {stderr}"
        )

        if expected_output:
            assert expected_output in stdout, (
                f"Expected output '{expected_output}' not found in stdout:\n{stdout}"
            )

    def assert_cli_error(
        self,
        args: List[str],
        expected_code: int,
        expected_error: Optional[str] = None,
        env: Optional[Dict[str, str]] = None
    ):
        """Assert that a CLI command fails with expected error.

        Args:
            args: CLI arguments
            expected_code: Expected exit code
            expected_error: Optional string that should appear in output
            env: Additional environment variables
        """
        returncode, stdout, stderr = self.run_cli(args, env=env)

        assert returncode == expected_code, (
            f"Expected exit code {expected_code}, got {returncode}\n"
            f"stdout: {stdout}\n"
            f"stderr: {stderr}"
        )

        if expected_error:
            combined_output = stdout + stderr
            assert expected_error in combined_output, (
                f"Expected error '{expected_error}' not found in output:\n"
                f"stdout: {stdout}\n"
                f"stderr: {stderr}"
            )

    def wait_for_condition(
        self,
        condition_func,
        timeout: int = 10,
        interval: float = 0.5
    ) -> bool:
        """Wait for a condition to become true.

        Args:
            condition_func: Function that returns True when condition is met
            timeout: Maximum time to wait in seconds
            interval: Check interval in seconds

        Returns:
            True if condition was met, False if timeout
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if condition_func():
                return True
            time.sleep(interval)
        return False

    def create_mock_docker_response(self, success: bool = True) -> Dict[str, Any]:
        """Create a mock Docker API response for testing.

        Args:
            success: Whether the Docker check should succeed

        Returns:
            Mock response dictionary
        """
        if success:
            return {
                "status": "running",
                "version": "20.10.17",
                "containers": [],
                "images": ["ghcr.io/swebench/runner:latest"]
            }
        else:
            return {
                "status": "not_running",
                "error": "Cannot connect to Docker daemon"
            }

    def __enter__(self):
        """Context manager entry."""
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.teardown()
        return False  # Don't suppress exceptions
