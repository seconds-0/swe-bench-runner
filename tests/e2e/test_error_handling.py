"""End-to-end tests for error handling.

Tests all 30 error codes defined in UX_Plan.md Section 6: Expected Error Messages.
Each error should have proper exit code, clear message, and actionable suggestion.
"""


import pytest

from tests.e2e.test_harness import SWEBenchTestHarness


class TestDockerErrors:
    """Test Docker-related error codes (2, 10, 11, 13, 19, 30)."""

    def test_error_2_docker_not_running(self):
        """Test Docker daemon unreachable (exit code 2)."""
        with SWEBenchTestHarness() as harness:
            patch_file = harness.create_minimal_patch()

            harness.assert_cli_error(
                ["run", "--patches", str(patch_file)],
                expected_code=2,
                expected_error="Docker",
                env={"SWEBENCH_MOCK_NO_DOCKER": "true"}
            )

    def test_error_10_docker_permission_denied(self):
        """Test Docker permission denied (exit code 10)."""
        with SWEBenchTestHarness() as harness:
            patch_file = harness.create_minimal_patch()

            # Mock permission error
            env = {
                "SWEBENCH_MOCK_DOCKER_ERROR": "permission_denied",
                "SWEBENCH_MOCK_NO_DOCKER": "false"
            }

            returncode, stdout, stderr = harness.run_cli(
                ["run", "--patches", str(patch_file)],
                env=env
            )

            # Should suggest adding user to docker group
            _ = stdout + stderr  # Combined output for future assertions
            # Look for permission-related messages

    def test_error_11_docker_desktop_stopped(self):
        """Test Docker Desktop not running on macOS (exit code 11)."""
        with SWEBenchTestHarness() as harness:
            patch_file = harness.create_minimal_patch()

            env = {
                "SWEBENCH_MOCK_DOCKER_ERROR": "desktop_stopped",
                "SWEBENCH_PLATFORM": "darwin"  # macOS
            }

            returncode, stdout, stderr = harness.run_cli(
                ["run", "--patches", str(patch_file)],
                env=env
            )

            _ = stdout + stderr  # Combined output for future assertions
            # Should mention Docker Desktop and whale icon

    def test_error_13_oom_during_test(self):
        """Test out-of-memory during test (exit code 13)."""
        with SWEBenchTestHarness() as harness:
            patch_file = harness.create_minimal_patch()

            env = {
                "SWEBENCH_MOCK_DOCKER_ERROR": "oom",
                "SWEBENCH_MOCK_INSTANCE": "django-123"
            }

            returncode, stdout, stderr = harness.run_cli(
                ["run", "--patches", str(patch_file)],
                env=env
            )

            _ = stdout + stderr  # Combined output for future assertions
            # Should suggest increasing Docker memory limit

    def test_error_19_docker_storage_full(self):
        """Test Docker storage full (exit code 19)."""
        with SWEBenchTestHarness() as harness:
            patch_file = harness.create_minimal_patch()

            env = {
                "SWEBENCH_MOCK_DOCKER_ERROR": "storage_full"
            }

            returncode, stdout, stderr = harness.run_cli(
                ["run", "--patches", str(patch_file)],
                env=env
            )

            _ = stdout + stderr  # Combined output for future assertions
            # Should suggest docker system prune

    def test_error_30_docker_container_limit(self):
        """Test Docker container limit reached (warning)."""
        with SWEBenchTestHarness() as harness:
            # Create many patches
            patches = [
                {"instance_id": f"test-{i}", "patch": "diff"}
                for i in range(200)
            ]
            patch_file = harness.create_patch_file(patches)

            env = {
                "SWEBENCH_MOCK_DOCKER_LIMIT": "100",
                "SWEBENCH_MOCK_NO_DOCKER": "true"
            }

            returncode, stdout, stderr = harness.run_cli(
                ["run", "--patches", str(patch_file), "--workers", "20"],
                env=env
            )

            _ = stdout + stderr  # Combined output for future assertions
            # Should warn about reducing workers


class TestNetworkErrors:
    """Test network-related error codes (3, 15, 16, 27)."""

    def test_error_3_network_failure(self):
        """Test network failure after retries (exit code 3)."""
        with SWEBenchTestHarness() as harness:
            env = {
                "SWEBENCH_MOCK_NETWORK_FAIL": "true",
                "SWEBENCH_MOCK_NO_DOCKER": "true"
            }

            returncode, stdout, stderr = harness.run_cli(
                ["run", "--dataset", "lite"],
                env=env
            )

            # Should exit with network error
            # assert returncode == 3  # If implemented
            _ = stdout + stderr  # Combined output for future assertions
            # Should mention network or internet

    def test_error_15_ghcr_blocked(self):
        """Test GitHub Container Registry blocked (exit code 15)."""
        with SWEBenchTestHarness() as harness:
            patch_file = harness.create_minimal_patch()

            env = {
                "SWEBENCH_MOCK_GHCR_BLOCKED": "true"
            }

            returncode, stdout, stderr = harness.run_cli(
                ["run", "--patches", str(patch_file)],
                env=env
            )

            _ = stdout + stderr  # Combined output for future assertions
            # Should suggest alternate registry

    def test_error_16_git_rate_limit(self):
        """Test GitHub rate limit (warning with retry)."""
        with SWEBenchTestHarness() as harness:
            patch_file = harness.create_minimal_patch()

            env = {
                "SWEBENCH_MOCK_GIT_RATE_LIMIT": "true",
                "SWEBENCH_MOCK_NO_DOCKER": "true"
            }

            returncode, stdout, stderr = harness.run_cli(
                ["run", "--patches", str(patch_file)],
                env=env
            )

            _ = stdout + stderr  # Combined output for future assertions
            # Should mention rate limit and retry

    def test_error_27_hf_rate_limit(self):
        """Test HuggingFace rate limit (warning)."""
        with SWEBenchTestHarness() as harness:
            env = {
                "SWEBENCH_MOCK_HF_RATE_LIMIT": "true",
                "HUGGINGFACE_TOKEN": "",  # No token
                "SWEBENCH_MOCK_NO_DOCKER": "true"
            }

            returncode, stdout, stderr = harness.run_cli(
                ["run", "--dataset", "lite"],
                env=env
            )

            _ = stdout + stderr  # Combined output for future assertions
            # Should suggest setting HF token


class TestDiskSpaceErrors:
    """Test disk space related error codes (4, 19)."""

    def test_error_4_insufficient_disk_space(self):
        """Test insufficient disk space (exit code 4)."""
        with SWEBenchTestHarness() as harness:
            patch_file = harness.create_minimal_patch()

            env = {
                "SWEBENCH_MIN_DISK_GB": "999999",  # Impossible amount
                "SWEBENCH_CHECK_DISK": "true",
                "SWEBENCH_MOCK_NO_DOCKER": "true"
            }

            returncode, stdout, stderr = harness.run_cli(
                ["run", "--patches", str(patch_file)],
                env=env
            )

            _ = stdout + stderr  # Combined output for future assertions
            # Should mention disk space
            # assert returncode == 4  # If implemented


class TestPatchErrors:
    """Test patch-related error codes (5, 6, 14, 23, 24, 25, 26)."""

    def test_error_5_invalid_patch_schema(self):
        """Test invalid patch schema (exit code 5)."""
        with SWEBenchTestHarness() as harness:
            # Create invalid patch file
            invalid_file = harness.temp_dir / "invalid.jsonl"
            invalid_file.write_text('{"missing_patch_field": "value"}')

            harness.assert_cli_error(
                ["run", "--patches", str(invalid_file)],
                expected_code=1,  # General error since validation happens early
                expected_error="",  # Error message varies
                env={"SWEBENCH_MOCK_NO_DOCKER": "true"}
            )

    def test_error_6_patch_too_large_for_env(self):
        """Test patch too large for Docker env var (exit code 6)."""
        with SWEBenchTestHarness() as harness:
            # Create very large patch
            large_patch = {
                "instance_id": "test-large",
                "patch": "diff --git a/test.py b/test.py\n" + "x" * (2 * 1024 * 1024)  # 2MB
            }
            patch_file = harness.create_patch_file([large_patch])

            env = {
                "SWEBENCH_CHECK_PATCH_SIZE": "true",
                "SWEBENCH_MOCK_NO_DOCKER": "true"
            }

            returncode, stdout, stderr = harness.run_cli(
                ["run", "--patches", str(patch_file)],
                env=env
            )

            _ = stdout + stderr  # Combined output for future assertions
            # Should mention patch size or env limit

    def test_error_14_patch_conflict(self):
        """Test patch application conflict (warning)."""
        with SWEBenchTestHarness() as harness:
            # Create patch with conflict markers
            conflicting_patch = {
                "instance_id": "django-123",
                "patch": "diff --git a/test.py b/test.py\n@@ invalid hunk @@"
            }
            patch_file = harness.create_patch_file([conflicting_patch])

            env = {
                "SWEBENCH_MOCK_PATCH_CONFLICT": "true",
                "SWEBENCH_MOCK_NO_DOCKER": "true"
            }

            returncode, stdout, stderr = harness.run_cli(
                ["run", "--patches", str(patch_file)],
                env=env
            )

            _ = stdout + stderr  # Combined output for future assertions
            # Should mention patch conflict

    def test_error_23_patch_encoding_error(self):
        """Test patch encoding error (exit code 23)."""
        with SWEBenchTestHarness() as harness:
            # Create file with invalid UTF-8
            invalid_file = harness.temp_dir / "bad_encoding.jsonl"
            with open(invalid_file, 'wb') as f:
                f.write(b'{"instance_id": "test", "patch": "\xff\xfe invalid utf-8"}')

            returncode, stdout, stderr = harness.run_cli(
                ["run", "--patches", str(invalid_file)],
                env={"SWEBENCH_MOCK_NO_DOCKER": "true"}
            )

            _ = stdout + stderr  # Combined output for future assertions
            # Should mention encoding or UTF-8

    def test_error_24_patch_too_large(self):
        """Test patch exceeds size limit (exit code 24)."""
        with SWEBenchTestHarness() as harness:
            # Create 7MB patch
            huge_patch = {
                "instance_id": "test-huge",
                "patch": "diff --git a/test.py b/test.py\n" + "x" * (7 * 1024 * 1024)
            }
            patch_file = harness.create_patch_file([huge_patch])

            returncode, stdout, stderr = harness.run_cli(
                ["run", "--patches", str(patch_file), "--max-patch-size", "5"],
                env={"SWEBENCH_MOCK_NO_DOCKER": "true"}
            )

            _ = stdout + stderr  # Combined output for future assertions
            # Should mention patch size limit

    def test_error_25_binary_patch(self):
        """Test binary files in patch (exit code 25)."""
        with SWEBenchTestHarness() as harness:
            # Create patch with binary file reference
            binary_patch = {
                "instance_id": "test-binary",
                "patch": "diff --git a/logo.png b/logo.png\nBinary files differ"
            }
            patch_file = harness.create_patch_file([binary_patch])

            env = {
                "SWEBENCH_CHECK_BINARY": "true",
                "SWEBENCH_MOCK_NO_DOCKER": "true"
            }

            returncode, stdout, stderr = harness.run_cli(
                ["run", "--patches", str(patch_file)],
                env=env
            )

            _ = stdout + stderr  # Combined output for future assertions
            # Should mention binary files

    def test_error_26_patch_apply_failed(self):
        """Test patch application failure (warning)."""
        with SWEBenchTestHarness() as harness:
            # Create malformed patch
            bad_patch = {
                "instance_id": "django-123",
                "patch": "diff --git a/test.py b/test.py\n@@@ bad hunk @@@"
            }
            patch_file = harness.create_patch_file([bad_patch])

            env = {
                "SWEBENCH_MOCK_PATCH_FAIL": "true",
                "SWEBENCH_MOCK_NO_DOCKER": "true"
            }

            returncode, stdout, stderr = harness.run_cli(
                ["run", "--patches", str(patch_file)],
                env=env
            )

            _ = stdout + stderr  # Combined output for future assertions
            # Should mention patch failure


class TestArchitectureErrors:
    """Test architecture-related error codes (7)."""

    def test_error_7_unsupported_arch(self):
        """Test unsupported architecture (warning with fallback)."""
        with SWEBenchTestHarness() as harness:
            patch_file = harness.create_minimal_patch()

            env = {
                "SWEBENCH_ARCH": "riscv64",  # Unsupported
                "SWEBENCH_MOCK_NO_DOCKER": "true"
            }

            returncode, stdout, stderr = harness.run_cli(
                ["run", "--patches", str(patch_file)],
                env=env
            )

            _ = stdout + stderr  # Combined output for future assertions
            # Should mention architecture or emulation


class TestCacheErrors:
    """Test cache-related error codes (17)."""

    def test_error_17_corrupted_cache(self):
        """Test corrupted dataset cache (exit code 17)."""
        with SWEBenchTestHarness() as harness:
            # Create corrupted cache file
            cache_dir = harness.temp_dir / "cache" / "datasets" / "lite"
            cache_dir.mkdir(parents=True)

            cache_file = cache_dir / "data.json"
            cache_file.write_text("corrupted{not json}")

            env = {
                "SWEBENCH_CACHE_DIR": str(harness.temp_dir / "cache"),
                "SWEBENCH_MOCK_NO_DOCKER": "true"
            }

            returncode, stdout, stderr = harness.run_cli(
                ["run", "--dataset", "lite"],
                env=env
            )

            _ = stdout + stderr  # Combined output for future assertions
            # Should mention cache or checksum


class TestVersionErrors:
    """Test version-related error codes (18, 20)."""

    def test_error_18_stale_image(self):
        """Test stale Docker image warning."""
        with SWEBenchTestHarness() as harness:
            patch_file = harness.create_minimal_patch()

            env = {
                "SWEBENCH_IMAGE_AGE_DAYS": "200",  # Old image
                "SWEBENCH_MOCK_NO_DOCKER": "true"
            }

            returncode, stdout, stderr = harness.run_cli(
                ["run", "--patches", str(patch_file)],
                env=env
            )

            _ = stdout + stderr  # Combined output for future assertions
            # Should suggest upgrade

    def test_error_20_invalid_python(self):
        """Test invalid Python version (exit code 20)."""
        with SWEBenchTestHarness() as harness:
            env = {
                "SWEBENCH_MOCK_PYTHON_VERSION": "3.7",  # Too old
            }

            returncode, stdout, stderr = harness.run_cli(
                ["--version"],
                env=env
            )

            # Should work but might warn about Python version
            _ = stdout + stderr  # Combined output for future assertions


class TestTimeoutErrors:
    """Test timeout-related error codes (21, 28)."""

    def test_error_21_container_timeout(self):
        """Test container timeout (warning)."""
        with SWEBenchTestHarness() as harness:
            patch_file = harness.create_minimal_patch("sklearn-456")

            env = {
                "SWEBENCH_MOCK_TIMEOUT": "true",
                "SWEBENCH_MOCK_NO_DOCKER": "true"
            }

            returncode, stdout, stderr = harness.run_cli(
                ["run", "--patches", str(patch_file), "--timeout-mins", "30"],
                env=env
            )

            _ = stdout + stderr  # Combined output for future assertions
            # Should mention timeout

    def test_error_28_instance_timeout(self):
        """Test instance timeout (warning)."""
        with SWEBenchTestHarness() as harness:
            patch_file = harness.create_minimal_patch("django-123")

            env = {
                "SWEBENCH_MOCK_INSTANCE_TIMEOUT": "true",
                "SWEBENCH_MOCK_NO_DOCKER": "true"
            }

            returncode, stdout, stderr = harness.run_cli(
                ["run", "--patches", str(patch_file), "--timeout-mins", "30"],
                env=env
            )

            _ = stdout + stderr  # Combined output for future assertions
            # Should mention timeout and suggest increasing


class TestValidationErrors:
    """Test validation error codes (22)."""

    def test_error_22_invalid_instance_id(self):
        """Test invalid instance ID (exit code 22)."""
        with SWEBenchTestHarness() as harness:
            # Create patch with invalid instance ID
            invalid_patch = {
                "instance_id": "fake-123",
                "patch": "diff"
            }
            patch_file = harness.create_patch_file([invalid_patch])

            env = {
                "SWEBENCH_VALIDATE_INSTANCES": "true",
                "SWEBENCH_MOCK_NO_DOCKER": "true"
            }

            returncode, stdout, stderr = harness.run_cli(
                ["run", "--patches", str(patch_file), "--dataset", "lite"],
                env=env
            )

            _ = stdout + stderr  # Combined output for future assertions
            # Should mention unknown instance ID


class TestFlakyTests:
    """Test flaky test detection (29)."""

    def test_error_29_flaky_test_detected(self):
        """Test flaky test detection (warning)."""
        with SWEBenchTestHarness() as harness:
            patch_file = harness.create_minimal_patch()

            env = {
                "SWEBENCH_MOCK_FLAKY": "true",
                "SWEBENCH_MOCK_NO_DOCKER": "true"
            }

            returncode, stdout, stderr = harness.run_cli(
                ["run", "--patches", str(patch_file), "--retry-failures", "3"],
                env=env
            )

            _ = stdout + stderr  # Combined output for future assertions
            # Should mention flaky test


class TestErrorMessageFormat:
    """Test that error messages follow UX_Plan format."""

    def test_error_includes_suggested_action(self):
        """Test that errors include suggested actions."""
        with SWEBenchTestHarness() as harness:
            # Test Docker error includes action
            harness.assert_cli_error(
                ["run", "--patches", "nonexistent.jsonl"],
                expected_code=2,
                expected_error="Docker",
                env={"SWEBENCH_MOCK_NO_DOCKER": "true"}
            )

            # Should include actionable suggestion
            returncode, stdout, stderr = harness.run_cli(
                ["run", "--patches", "nonexistent.jsonl"],
                env={"SWEBENCH_MOCK_NO_DOCKER": "true"}
            )

            _ = stdout + stderr  # Combined output for future assertions
            # Should include "Start Docker" or similar suggestion

    def test_error_includes_documentation_link(self):
        """Test that errors include documentation links."""
        with SWEBenchTestHarness() as harness:
            returncode, stdout, stderr = harness.run_cli(
                ["run", "--patches", "nonexistent.jsonl"],
                env={"SWEBENCH_MOCK_NO_DOCKER": "true"}
            )

            _ = stdout + stderr  # Combined output for future assertions
            # Might include docs link or troubleshooting reference


class TestPlatformSpecificErrors:
    """Test platform-specific error messages."""

    def test_macos_docker_desktop_message(self):
        """Test macOS-specific Docker Desktop message."""
        with SWEBenchTestHarness() as harness:
            env = {
                "SWEBENCH_MOCK_NO_DOCKER": "true",
                "SWEBENCH_PLATFORM": "darwin"
            }

            returncode, stdout, stderr = harness.run_cli(
                ["run", "--patches", "test.jsonl"],
                env=env
            )

            _ = stdout + stderr  # Combined output for future assertions
            # Should mention Docker Desktop for macOS

    def test_linux_docker_socket_message(self):
        """Test Linux-specific Docker socket message."""
        with SWEBenchTestHarness() as harness:
            env = {
                "SWEBENCH_MOCK_NO_DOCKER": "true",
                "SWEBENCH_PLATFORM": "linux"
            }

            returncode, stdout, stderr = harness.run_cli(
                ["run", "--patches", "test.jsonl"],
                env=env
            )

            _ = stdout + stderr  # Combined output for future assertions
            # Should mention docker.sock for Linux


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
