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
from typing import Any

try:
    # When run as part of tests package
    # Import others only if needed to avoid unused import warnings
    from tests.e2e.test_doubles import (  # noqa: F401
        DockerClientDouble,
        FileSystemDouble,
        HuggingFaceDouble,
        NetworkDouble,
        ProviderDouble,
        TestDoubleFactory,
    )
except ImportError:
    # When run directly from tests/e2e directory
    from test_doubles import (
        DockerClientDouble,
        TestDoubleFactory,
    )


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
        self.temp_dir: Path | None = None
        self.original_env: dict[str, str] = {}
        self.env_vars: dict[str, str] = {}
        self.processes: list[subprocess.Popen] = []
        self.created_files: list[Path] = []
        self.created_dirs: list[Path] = []
        self.injected_doubles: dict[str, Any] = {}
        self._original_factories = {}

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

        return self.temp_dir

    def teardown(self):
        """Clean up all resources created during testing."""
        # Restore original factories
        self._restore_factories()

        # Reset injected abstractions to avoid leaking into other tests
        try:
            from swebench_runner.hf_abstraction import set_hf_client  # type: ignore
            set_hf_client(None)
        except Exception:
            pass
        try:
            from swebench_runner.fs_abstraction import set_filesystem  # type: ignore
            set_filesystem(None)
        except Exception:
            pass
        try:
            from swebench_runner.patch_validation import (
                set_patch_validator,  # type: ignore
            )
            set_patch_validator(None)
        except Exception:
            pass
        try:
            from swebench_runner.network_abstraction import (
                set_network_client,  # type: ignore
            )
            set_network_client(None)
        except Exception:
            pass
        try:
            from swebench_runner.instance_abstraction import (
                set_instance_client,  # type: ignore
            )
            set_instance_client(None)
        except Exception:
            pass
        try:
            import swebench_runner.docker_client as docker_client  # type: ignore
            docker_client.reset_docker_client_factory()
        except Exception:
            pass

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

    def inject_docker_double(self, scenario: str = "success") -> DockerClientDouble:
        """Inject a Docker client test double.

        Args:
            scenario: Test scenario for the double

        Returns:
            The injected Docker double
        """
        double = TestDoubleFactory.create_docker_double(scenario)

        # Import here to avoid circular dependency
        import swebench_runner.docker_client as docker_client

        # Store original factory
        if "docker" not in self._original_factories:
            self._original_factories["docker"] = docker_client._docker_client_factory

        # Inject the double
        docker_client.set_docker_client_factory(lambda: double)

        self.injected_doubles["docker"] = double
        return double

    def inject_patch_double(self, scenario: str = "success"):
        """Inject a patch validator test double.

        Args:
            scenario: Test scenario for the double

        Returns:
            The injected patch validator double
        """
        from tests.e2e.test_doubles import TestDoubleFactory
        double = TestDoubleFactory.create_patch_double(scenario)

        # Hook into the runner's validation seam
        try:
            from swebench_runner.patch_validation import set_patch_validator
            set_patch_validator(double)
        except Exception:
            pass

        self.injected_doubles["patch"] = double
        return double

    def inject_network_double(self, scenario: str = "success"):
        """Inject a network test double.

        Args:
            scenario: Test scenario for the double

        Returns:
            The injected network double
        """
        from tests.e2e.test_doubles import TestDoubleFactory
        double = TestDoubleFactory.create_network_double(scenario)

        # Hook into network abstraction
        try:
            from swebench_runner.network_abstraction import set_network_client
            set_network_client(double)
        except Exception:
            pass

        self.injected_doubles["network"] = double
        return double

    def inject_filesystem_double(self, scenario: str = "success"):
        """Inject a file system test double.

        Args:
            scenario: Test scenario for the double

        Returns:
            The injected filesystem double
        """
        from tests.e2e.test_doubles import TestDoubleFactory
        double = TestDoubleFactory.create_filesystem_double(scenario)

        # Hook into runner filesystem abstraction
        try:
            from swebench_runner.fs_abstraction import set_filesystem
            set_filesystem(double)
        except Exception:
            pass

        self.injected_doubles["filesystem"] = double
        return double

    def inject_huggingface_double(self, scenario: str = "success"):
        """Inject a HuggingFace test double.

        Args:
            scenario: Test scenario for the double

        Returns:
            The injected HuggingFace double
        """
        from tests.e2e.test_doubles import TestDoubleFactory
        double = TestDoubleFactory.create_huggingface_double(scenario)

        # Hook into HF abstraction so dataset fetches use the double
        try:
            from swebench_runner.hf_abstraction import set_hf_client
            set_hf_client(double)
        except Exception:
            pass

        self.injected_doubles["huggingface"] = double
        return double

    def inject_provider_double(self, provider: str = "openai", scenario: str = "success"):
        """Inject a provider test double.

        Args:
            provider: Provider name (openai, anthropic, ollama)
            scenario: Test scenario for the double

        Returns:
            The injected provider double
        """
        from tests.e2e.test_doubles import TestDoubleFactory
        double = TestDoubleFactory.create_provider_double(provider, scenario)

        # TODO: Hook up to actual provider operations once abstraction exists
        # For now, just store for test verification
        self.injected_doubles[provider] = double
        return double

    def inject_instance_double(self, scenario: str = "success"):
        """Inject an instance test double.

        Args:
            scenario: Test scenario for the double

        Returns:
            The injected instance double
        """
        from tests.e2e.test_doubles import TestDoubleFactory
        double = TestDoubleFactory.create_instance_double(scenario)

        # Hook into runner instance abstraction so validations are exercised
        try:
            from swebench_runner.instance_abstraction import (
                set_instance_client,  # type: ignore
            )
            set_instance_client(double)
        except Exception:
            pass

        self.injected_doubles["instance"] = double
        return double

    def inject_all_doubles(self, scenario: str = "success") -> dict[str, Any]:
        """Inject all test doubles with the same scenario.

        Args:
            scenario: Test scenario for all doubles

        Returns:
            Dictionary of all injected doubles
        """
        doubles = TestDoubleFactory.create_all_doubles(scenario)

        # Inject Docker
        if "docker" in doubles:
            self.inject_docker_double(scenario)

        # Store all doubles
        self.injected_doubles.update(doubles)
        return doubles

    def _restore_factories(self):
        """Restore original factories."""
        if "docker" in self._original_factories:
            import swebench_runner.docker_client as docker_client
            docker_client._docker_client_factory = self._original_factories["docker"]

        self._original_factories.clear()
        self.injected_doubles.clear()

    def run_cli(
        self,
        args: list[str],
        env: dict[str, str] | None = None,
        timeout: int = 30,
        input_text: str | None = None,
        check_returncode: bool = False
    ) -> tuple[int, str, str]:
        """Run CLI command with proper isolation using subprocess.

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

    def run_cli_direct(
        self,
        args: list[str],
        env: dict[str, str] | None = None,
        input_text: str | None = None
    ) -> tuple[int, str, str]:
        """Run CLI command directly in the same process (for test double injection).

        This method calls the CLI module directly instead of using subprocess,
        allowing injected test doubles to work correctly.

        Args:
            args: CLI arguments (e.g., ["run", "--patches", "file.jsonl"])
            env: Additional environment variables to set
            input_text: Optional input to simulate stdin (not fully supported)

        Returns:
            Tuple of (returncode, stdout, stderr)
        """
        import contextlib
        import io

        from swebench_runner import cli

        # Save original argv and environment
        original_argv = sys.argv
        original_env = os.environ.copy()

        # Set up new argv
        sys.argv = ["swebench_runner"] + args

        # Apply environment variables if provided
        if env:
            os.environ.update(env)

        # Capture stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        returncode = 0

        try:
            # Redirect stdout and stderr
            with contextlib.redirect_stdout(stdout_capture), \
                 contextlib.redirect_stderr(stderr_capture):

                # Add --no-input flag if not present to avoid prompts
                if "--no-input" not in args and "setup" not in args:
                    sys.argv.append("--no-input")

                # In direct mode, do not force Docker-not-running unless explicitly tested
                os.environ["SWEBENCH_MOCK_NO_DOCKER"] = "false"

                # Call CLI directly
                # Ensure a default Docker double is present if none injected
                try:
                    from swebench_runner.docker_client import (
                        is_custom_factory_set as _factory_set,
                    )
                    from swebench_runner.docker_client import (
                        set_docker_client_factory as _set_docker_factory,
                    )
                    if not _factory_set():
                        # Lazy import to avoid circular issues
                        from tests.e2e.test_doubles import (
                            DockerClientDouble,  # type: ignore
                        )
                        _set_docker_factory(lambda: DockerClientDouble("success"))
                except Exception:
                    pass

                cli.cli()

        except SystemExit as e:
            # Capture the exit code
            returncode = e.code if e.code is not None else 0
        except Exception as e:
            # Unexpected error
            stderr_capture.write(f"Unexpected error: {e}\n")
            returncode = 1
        finally:
            # Restore original argv and environment
            sys.argv = original_argv
            os.environ.clear()
            os.environ.update(original_env)

        return returncode, stdout_capture.getvalue(), stderr_capture.getvalue()

    def create_patch_file(
        self,
        patches: list[dict[str, str]],
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
        args: list[str],
        expected_output: str | None = None,
        env: dict[str, str] | None = None
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
        args: list[str],
        expected_code: int,
        expected_error: str | None = None,
        env: dict[str, str] | None = None
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

    def create_mock_docker_response(self, success: bool = True) -> dict[str, Any]:
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
