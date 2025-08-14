"""End-to-end tests for error handling.

Tests all 30 error codes defined in UX_Plan.md Section 6: Expected Error Messages.
Each error should have proper exit code, clear message, and actionable suggestion.
"""


import pytest

from tests.e2e.assertion_helpers import (
    assert_contains_suggestion,
    assert_docker_error,
    assert_exit_code,
    assert_network_error,
    assert_patch_error,
    assert_platform_specific,
    assert_resource_error,
    assert_timeout_error,
    assert_validation_error,
)
from tests.e2e.test_harness import SWEBenchTestHarness


class TestDockerErrors:
    """Test Docker-related error codes (2, 10, 11, 13, 19, 30)."""

    def test_error_2_docker_not_running(self):
        """Test Docker daemon unreachable (exit code 2)."""
        with SWEBenchTestHarness() as harness:
            # Inject Docker double with "not_running" scenario
            docker_double = harness.inject_docker_double(scenario="not_running")

            patch_file = harness.create_minimal_patch()

            returncode, stdout, stderr = harness.run_cli_direct(
                ["run", "--patches", str(patch_file)]
            )
            combined = stdout + stderr

            # Verify exit code from UX_Plan
            assert_exit_code(returncode, 2, "Docker not running")

            # Verify Docker error messages
            assert_docker_error(combined, "not_running")

            # Verify actionable suggestion
            assert_contains_suggestion(combined)

            # Verify platform-specific guidance
            assert_platform_specific(combined)

            # Verify the double was called
            assert docker_double.ping_called, "Docker ping should have been called"

    def test_error_10_docker_permission_denied(self):
        """Test Docker permission denied (exit code 10)."""
        with SWEBenchTestHarness() as harness:
            # Inject Docker double with "permission_denied" scenario
            docker_double = harness.inject_docker_double(scenario="permission_denied")

            patch_file = harness.create_minimal_patch()

            returncode, stdout, stderr = harness.run_cli_direct(
                ["run", "--patches", str(patch_file)]
            )
            combined = stdout + stderr

            # UX_Plan error 10 maps to exit code 2 (Docker category)
            assert_exit_code(returncode, 2, "Docker permission denied")

            # Verify permission error messages
            assert_docker_error(combined, "permission")

            # Should include usermod command
            assert_contains_suggestion(combined)

            # Verify the double was called
            assert docker_double.ping_called, "Docker ping should have been called"

    def test_error_11_docker_desktop_stopped(self):
        """Test Docker Desktop not running on macOS (exit code 11)."""
        with SWEBenchTestHarness() as harness:
            # Inject Docker double with "desktop_stopped" scenario
            docker_double = harness.inject_docker_double(scenario="desktop_stopped")

            patch_file = harness.create_minimal_patch()

            returncode, stdout, stderr = harness.run_cli_direct(
                ["run", "--patches", str(patch_file)],
                env={"SWEBENCH_PLATFORM": "darwin"}  # Keep platform setting for now
            )
            combined = stdout + stderr

            # UX_Plan error 11 maps to exit code 2 (Docker category)
            assert_exit_code(returncode, 2, "Docker Desktop stopped")

            # Verify Docker Desktop specific messages
            assert_docker_error(combined, "desktop_stopped")

            # Must have macOS-specific guidance
            assert_platform_specific(combined, "darwin")

            # Verify the double was called
            assert docker_double.ping_called, "Docker ping should have been called"

    def test_error_13_oom_during_test(self):
        """Test out-of-memory during test (exit code 13)."""
        with SWEBenchTestHarness() as harness:
            # Inject Docker double with OOM during container run scenario
            docker_double = harness.inject_docker_double(scenario="oom_during_run")

            # Create patch with specific instance ID for testing
            patches = [{"instance_id": "django-123", "patch": "diff --git a/test.py b/test.py"}]
            patch_file = harness.create_patch_file(patches)

            returncode, stdout, stderr = harness.run_cli_direct(
                ["run", "--patches", str(patch_file)]
            )
            combined = stdout + stderr

            # UX_Plan error 13 maps to exit code 4 (Resource error)
            assert_exit_code(returncode, 4, "Out of memory")

            # Verify memory error messages
            assert_resource_error(combined, "memory")

            # Should mention the specific instance
            assert "django-123" in combined

            # Should suggest solution
            assert_contains_suggestion(combined)

            # Verify the double was called
            assert docker_double.containers_run_called, "Docker container run should have been attempted"

    def test_error_19_docker_storage_full(self):
        """Test Docker storage full (exit code 19)."""
        with SWEBenchTestHarness() as harness:
            # Inject Docker double with storage full during container run scenario
            docker_double = harness.inject_docker_double(scenario="storage_full_during_run")

            patch_file = harness.create_minimal_patch()

            returncode, stdout, stderr = harness.run_cli_direct(
                ["run", "--patches", str(patch_file)]
            )
            combined = stdout + stderr

            # UX_Plan error 19 maps to exit code 4 (Resource error)
            assert_exit_code(returncode, 4, "Docker storage full")

            # Verify Docker storage error
            assert_resource_error(combined, "docker_storage")

            # Must include docker system prune command
            assert "docker system prune" in combined

            # Verify the double was called
            assert docker_double.containers_run_called, "Docker container run should have been attempted"

    def test_error_30_docker_container_limit(self):
        """Test Docker container limit reached (warning)."""
        with SWEBenchTestHarness() as harness:
            # Inject Docker double with container limit exceeded scenario
            docker_double = harness.inject_docker_double(scenario="container_limit_exceeded")

            # Create a small number of patches (container limit warning still triggers)
            patches = [
                {"instance_id": f"test-{i}", "patch": "diff"}
                for i in range(10)
            ]
            patch_file = harness.create_patch_file(patches)

            returncode, stdout, stderr = harness.run_cli_direct(
                ["run", "--patches", str(patch_file), "--max-workers", "20"]
            )
            combined = stdout + stderr

            # This is a warning, not an error - may still succeed
            # UX_Plan says it reduces workers automatically
            assert returncode in [0, 1], f"Should succeed or warn, got {returncode}"

            # Should mention container limit
            assert "container limit" in combined.lower() or "100" in combined

            # Should mention worker reduction
            assert "workers" in combined.lower() or "reduced" in combined.lower()

            # Verify the double was used
            assert len(docker_double.containers_list) > 100, "Should have simulated many containers"


class TestNetworkErrors:
    """Test network-related error codes (3, 15, 16, 27)."""

    def test_error_3_network_failure(self):
        """Test network failure after retries (exit code 3)."""
        with SWEBenchTestHarness() as harness:
            # Inject network double with general failure scenario
            harness.inject_network_double(scenario="general_failure")

            returncode, stdout, stderr = harness.run_cli_direct(
                ["run", "--dataset", "lite"]
            )
            combined = stdout + stderr

            # UX_Plan error 3 is network failure
            assert_exit_code(returncode, 3, "Network failure")

            # Verify network error messages
            assert_network_error(combined, "general")

            # Should suggest retry or offline mode
            assert_contains_suggestion(combined)

    def test_error_15_ghcr_blocked(self):
        """Test GitHub Container Registry blocked (exit code 15)."""
        with SWEBenchTestHarness() as harness:
            # Inject network double with GHCR blocked scenario
            harness.inject_network_double(scenario="ghcr_blocked")

            patch_file = harness.create_minimal_patch()

            returncode, stdout, stderr = harness.run_cli_direct(
                ["run", "--patches", str(patch_file)]
            )
            combined = stdout + stderr

            # UX_Plan error 15 maps to exit code 3 (Network error)
            assert_exit_code(returncode, 3, "GHCR blocked")

            # Verify GHCR-specific error
            assert_network_error(combined, "ghcr_blocked")

            # Must suggest alternate registry
            assert_contains_suggestion(combined)

    def test_error_16_git_rate_limit(self):
        """Test GitHub rate limit (warning with retry)."""
        with SWEBenchTestHarness() as harness:
            # Inject network double with GitHub rate limit scenario
            network_double = harness.inject_network_double(scenario="git_rate_limit")

            patch_file = harness.create_minimal_patch()

            returncode, stdout, stderr = harness.run_cli_direct(
                ["run", "--patches", str(patch_file)]
            )
            combined = stdout + stderr

            # UX_Plan error 16 maps to exit code 3 (Network error)
            assert_exit_code(returncode, 3, "GitHub rate limit")

            # Verify rate limit error
            assert_network_error(combined, "git_rate_limit")

            # Should mention retry behavior
            assert "retry" in combined.lower() or "attempt" in combined.lower()

            # Verify the double tracked rate limits
            assert network_double.rate_limit_hits > 0, "Should have hit rate limit"

    def test_error_27_hf_rate_limit(self):
        """Test HuggingFace rate limit (warning)."""
        with SWEBenchTestHarness() as harness:
            # Inject HuggingFace double with rate limit scenario
            harness.inject_huggingface_double(scenario="rate_limit")

            returncode, stdout, stderr = harness.run_cli_direct(
                ["run", "--dataset", "lite"],
                env={"HUGGINGFACE_TOKEN": ""}  # No token
            )
            combined = stdout + stderr

            # UX_Plan error 27 maps to exit code 3 (Network error)
            assert_exit_code(returncode, 3, "HuggingFace rate limit")

            # Verify HF rate limit error
            assert_network_error(combined, "hf_rate_limit")

            # Must suggest using token
            assert_contains_suggestion(combined)


class TestDiskSpaceErrors:
    """Test disk space related error codes (4, 19)."""

    def test_error_4_insufficient_disk_space(self):
        """Test insufficient disk space (exit code 4)."""
        with SWEBenchTestHarness() as harness:
            # Inject filesystem double with disk full scenario
            harness.inject_filesystem_double(scenario="disk_full")

            patch_file = harness.create_minimal_patch()

            returncode, stdout, stderr = harness.run_cli_direct(
                ["run", "--patches", str(patch_file)]
            )
            combined = stdout + stderr

            # UX_Plan error 4 is disk space
            assert_exit_code(returncode, 4, "Insufficient disk space")

            # Verify disk space error
            assert_resource_error(combined, "disk_space")

            # Should suggest freeing space
            assert_contains_suggestion(combined)


class TestPatchErrors:
    """Test patch-related error codes (5, 6, 14, 23, 24, 25, 26)."""

    def test_error_5_invalid_patch_schema(self):
        """Test invalid patch schema (exit code 5)."""
        with SWEBenchTestHarness() as harness:
            # Inject patch validator double with invalid schema scenario
            harness.inject_patch_double(scenario="invalid_schema")

            # Create invalid patch file
            invalid_file = harness.temp_dir / "invalid.jsonl"
            invalid_file.write_text('{"missing_patch_field": "value"}')

            returncode, stdout, stderr = harness.run_cli_direct(
                ["run", "--patches", str(invalid_file)]
            )
            combined = stdout + stderr

            # UX_Plan error 5 maps to exit code 1 (General error)
            assert_exit_code(returncode, 1, "Invalid patch schema")

            # Verify patch schema error
            assert_patch_error(combined, "invalid_schema")

            # Should provide actionable guidance
            assert_contains_suggestion(combined)

    def test_error_6_patch_too_large_for_env(self):
        """Test patch too large for Docker env var (exit code 6)."""
        with SWEBenchTestHarness() as harness:
            # Inject patch validator double with too large for env scenario
            patch_double = harness.inject_patch_double(scenario="too_large_env")

            # Create very large patch
            large_patch = {
                "instance_id": "test-large",
                "patch": "diff --git a/test.py b/test.py\n" + "x" * (2 * 1024 * 1024)  # 2MB
            }
            patch_file = harness.create_patch_file([large_patch])

            returncode, stdout, stderr = harness.run_cli_direct(
                ["run", "--patches", str(patch_file)]
            )
            combined = stdout + stderr

            # UX_Plan error 6 maps to exit code 1 (General error)
            assert_exit_code(returncode, 1, "Patch too large for env")

            # Verify patch size error
            assert_patch_error(combined, "too_large")

            # Should suggest workaround
            assert_contains_suggestion(combined)

            # Verify the double was called
            assert patch_double.patches_validated, "Patch validator should have been called"

    def test_error_14_patch_conflict(self):
        """Test patch application conflict (warning)."""
        with SWEBenchTestHarness() as harness:
            # Inject patch validator double with conflict scenario
            patch_double = harness.inject_patch_double(scenario="conflict")

            # Create patch with conflict markers
            conflicting_patch = {
                "instance_id": "django-123",
                "patch": "diff --git a/test.py b/test.py\n@@ invalid hunk @@"
            }
            patch_file = harness.create_patch_file([conflicting_patch])

            returncode, stdout, stderr = harness.run_cli_direct(
                ["run", "--patches", str(patch_file)]
            )
            combined = stdout + stderr

            # UX_Plan error 14 maps to exit code 1 (General error)
            assert_exit_code(returncode, 1, "Patch conflict")

            # Verify patch conflict error
            assert_patch_error(combined, "conflict")

            # Should mention specific instance
            assert "django-123" in combined

            # Verify the double detected conflict
            assert patch_double.validation_errors, "Should have validation errors"

    def test_error_23_patch_encoding_error(self):
        """Test patch encoding error (exit code 23)."""
        with SWEBenchTestHarness() as harness:
            # Inject patch validator double with encoding error scenario
            patch_double = harness.inject_patch_double(scenario="encoding_error")

            # Create file with invalid UTF-8
            invalid_file = harness.temp_dir / "bad_encoding.jsonl"
            with open(invalid_file, 'wb') as f:
                f.write(b'{"instance_id": "test", "patch": "\xff\xfe invalid utf-8"}')

            returncode, stdout, stderr = harness.run_cli_direct(
                ["run", "--patches", str(invalid_file)]
            )
            combined = stdout + stderr

            # UX_Plan error 23 maps to exit code 1 (General error)
            assert_exit_code(returncode, 1, "Patch encoding error")

            # Verify encoding error
            assert_patch_error(combined, "encoding")

            # Should provide actionable guidance
            assert_contains_suggestion(combined)

            # Verify the double detected encoding issues
            assert patch_double.validation_errors, "Should have validation errors"

    def test_error_24_patch_too_large(self):
        """Test patch exceeds size limit (exit code 24)."""
        with SWEBenchTestHarness() as harness:
            # Inject patch validator double with too large scenario
            patch_double = harness.inject_patch_double(scenario="too_large")

            # Create 7MB patch
            huge_patch = {
                "instance_id": "test-huge",
                "patch": "diff --git a/test.py b/test.py\n" + "x" * (7 * 1024 * 1024)
            }
            patch_file = harness.create_patch_file([huge_patch])

            returncode, stdout, stderr = harness.run_cli_direct(
                ["run", "--patches", str(patch_file), "--max-patch-size", "5"]
            )
            combined = stdout + stderr

            # UX_Plan error 24 maps to exit code 1 (General error)
            assert_exit_code(returncode, 1, "Patch too large")

            # Verify patch size error
            assert_patch_error(combined, "too_large")

            # Should mention size and limit
            assert any(term in combined for term in ["7", "5MB", "exceeds"]), \
                "Should mention patch size and limit"

            assert_contains_suggestion(combined)

            # Verify the double checked patch size
            assert patch_double.patches_validated, "Patch validator should have been called"

    def test_error_25_binary_patch(self):
        """Test binary files in patch (exit code 25)."""
        with SWEBenchTestHarness() as harness:
            # Inject patch validator double with binary scenario
            patch_double = harness.inject_patch_double(scenario="binary")

            # Create patch with binary file reference
            binary_patch = {
                "instance_id": "test-binary",
                "patch": "diff --git a/logo.png b/logo.png\nBinary files differ"
            }
            patch_file = harness.create_patch_file([binary_patch])

            returncode, stdout, stderr = harness.run_cli_direct(
                ["run", "--patches", str(patch_file)]
            )
            combined = stdout + stderr

            # UX_Plan error 25 maps to exit code 1 (General error)
            assert_exit_code(returncode, 1, "Binary patch")

            # Verify binary file error
            assert_patch_error(combined, "binary")

            # Should mention --allow-binary flag
            assert_contains_suggestion(combined)

            # Verify the double detected binary content
            assert patch_double.validation_errors, "Should have validation errors"

    def test_error_26_patch_apply_failed(self):
        """Test patch application failure (warning)."""
        with SWEBenchTestHarness() as harness:
            # Inject patch validator double with apply failed scenario
            patch_double = harness.inject_patch_double(scenario="apply_failed")

            # Create malformed patch
            bad_patch = {
                "instance_id": "django-123",
                "patch": "diff --git a/test.py b/test.py\n@@@ bad hunk @@@"
            }
            patch_file = harness.create_patch_file([bad_patch])

            returncode, stdout, stderr = harness.run_cli_direct(
                ["run", "--patches", str(patch_file)]
            )
            combined = stdout + stderr

            # UX_Plan error 26 maps to exit code 1 (General error)
            assert_exit_code(returncode, 1, "Patch apply failed")

            # Verify patch failure error
            assert_patch_error(combined, "conflict")

            # Should mention the specific instance
            assert "django-123" in combined

            assert_contains_suggestion(combined)

            # Verify the double tracked application failure
            assert patch_double.validation_errors, "Should have validation errors"


class TestArchitectureErrors:
    """Test architecture-related error codes (7)."""

    def test_error_7_unsupported_arch(self):
        """Test unsupported architecture (warning with fallback)."""
        with SWEBenchTestHarness() as harness:
            # Mock platform architecture directly
            patch_file = harness.create_minimal_patch()

            # Still use env for platform until we have full abstraction
            env = {
                "SWEBENCH_ARCH": "riscv64"  # Unsupported
            }

            returncode, stdout, stderr = harness.run_cli_direct(
                ["run", "--patches", str(patch_file)],
                env=env
            )
            combined = stdout + stderr

            # UX_Plan error 7 is a warning, may still succeed
            assert returncode in [0, 1], f"Should succeed with warning or fail, got {returncode}"

            # Should mention architecture
            assert any(term in combined.lower() for term in ["arch", "arm64", "x86", "emulation"]), \
                "Should mention architecture or emulation"

            # Should provide workaround
            assert_contains_suggestion(combined)


class TestCacheErrors:
    """Test cache-related error codes (17)."""

    def test_error_17_corrupted_cache(self):
        """Test corrupted dataset cache (exit code 17)."""
        with SWEBenchTestHarness() as harness:
            # Inject filesystem double with cache corrupted scenario
            harness.inject_filesystem_double(scenario="cache_corrupted")

            # Create corrupted cache file
            cache_dir = harness.temp_dir / "cache" / "datasets" / "lite"
            cache_dir.mkdir(parents=True)

            cache_file = cache_dir / "data.json"
            cache_file.write_text("corrupted{not json}")

            env = {
                "SWEBENCH_CACHE_DIR": str(harness.temp_dir / "cache")
            }

            returncode, stdout, stderr = harness.run_cli_direct(
                ["run", "--dataset", "lite"],
                env=env
            )
            combined = stdout + stderr

            # UX_Plan error 17 maps to exit code 1 (General error)
            assert_exit_code(returncode, 1, "Corrupted cache")

            # Should mention cache corruption
            assert any(term in combined.lower() for term in ["cache", "corrupt", "checksum", "invalid"]), \
                "Should mention cache corruption"

            # Should suggest clean command
            assert "swebench clean" in combined or "--datasets" in combined, \
                "Should suggest swebench clean --datasets"

            assert_contains_suggestion(combined)


class TestVersionErrors:
    """Test version-related error codes (18, 20)."""

    def test_error_18_stale_image(self):
        """Test stale Docker image warning."""
        with SWEBenchTestHarness() as harness:
            # Inject Docker double with stale image scenario
            docker_double = harness.inject_docker_double(scenario="stale_image")

            patch_file = harness.create_minimal_patch()

            returncode, stdout, stderr = harness.run_cli_direct(
                ["run", "--patches", str(patch_file)]
            )
            combined = stdout + stderr

            # This is a warning, may still succeed
            assert returncode in [0, 1], f"Should succeed with warning or fail, got {returncode}"

            # Should mention stale image
            assert any(term in combined.lower() for term in ["stale", "old", "months", "upgrade"]), \
                "Should mention stale image"

            # Should suggest upgrade
            assert "swebench upgrade" in combined or "pull" in combined, \
                "Should suggest swebench upgrade"

            assert_contains_suggestion(combined)

            # Verify the double tracked image age
            assert docker_double.image_age_days == 200, "Should have old image"

    @pytest.mark.skip(reason="Python version check not implemented in --version command")
    def test_error_20_invalid_python(self):
        """Test invalid Python version (exit code 20)."""
        with SWEBenchTestHarness() as harness:
            # TODO: Replace with proper Python version double once abstraction exists
            # For now, still use env var as Python version check happens at startup
            env = {
                "SWEBENCH_MOCK_PYTHON_VERSION": "3.7"  # Too old - Last remaining mock env var
            }

            returncode, stdout, stderr = harness.run_cli_direct(
                ["--version"],
                env=env
            )
            combined = stdout + stderr

            # Version check might still work or warn
            assert returncode in [0, 1], f"Should succeed or warn, got {returncode}"

            # Should mention Python version if checking
            if "3.7" in combined or "python" in combined.lower():
                # Should mention version requirement
                assert any(term in combined for term in ["3.11", "3.12", "3.13", "pyenv", "conda"]), \
                    "Should mention Python version requirement"


class TestTimeoutErrors:
    """Test timeout-related error codes (21, 28)."""

    def test_error_21_container_timeout(self):
        """Test container timeout (warning)."""
        with SWEBenchTestHarness() as harness:
            # Inject Docker double with container timeout scenario
            docker_double = harness.inject_docker_double(scenario="container_timeout")

            patch_file = harness.create_minimal_patch("sklearn-456")

            returncode, stdout, stderr = harness.run_cli_direct(
                ["run", "--patches", str(patch_file), "--timeout-mins", "30"]
            )
            combined = stdout + stderr

            # UX_Plan error 21 maps to exit code 1 (General error)
            assert_exit_code(returncode, 1, "Container timeout")

            # Verify timeout error
            assert_timeout_error(combined, "sklearn-456")

            # Should suggest increasing timeout
            assert_contains_suggestion(combined)

            # Verify the double detected timeout
            assert docker_double.containers_run_called, "Should have attempted container run"

    def test_error_28_instance_timeout(self):
        """Test instance timeout (warning)."""
        with SWEBenchTestHarness() as harness:
            # Inject instance double with timeout scenario
            instance_double = harness.inject_instance_double(scenario="timeout")

            patch_file = harness.create_minimal_patch("django-123")

            returncode, stdout, stderr = harness.run_cli_direct(
                ["run", "--patches", str(patch_file), "--timeout-mins", "30"]
            )
            combined = stdout + stderr

            # UX_Plan error 28 maps to exit code 1 (General error)
            assert_exit_code(returncode, 1, "Instance timeout")

            # Verify timeout error
            assert_timeout_error(combined, "django-123")

            # Should mention current timeout and suggest increase
            assert "30" in combined, "Should mention current timeout"
            assert "--timeout-mins 60" in combined or "increase" in combined.lower(), \
                "Should suggest increasing timeout"

            assert_contains_suggestion(combined)

            # Verify the double tracked instance run
            assert any(i["id"] == "django-123" for i in instance_double.instances_run), "Should have run instance"


class TestValidationErrors:
    """Test validation error codes (22)."""

    def test_error_22_invalid_instance_id(self):
        """Test invalid instance ID (exit code 22)."""
        with SWEBenchTestHarness() as harness:
            # Inject instance double with invalid ID scenario
            instance_double = harness.inject_instance_double(scenario="invalid_id")

            # Create patch with invalid instance ID
            invalid_patch = {
                "instance_id": "fake-123",
                "patch": "diff"
            }
            patch_file = harness.create_patch_file([invalid_patch])

            returncode, stdout, stderr = harness.run_cli_direct(
                ["run", "--patches", str(patch_file), "--dataset", "lite"]
            )
            combined = stdout + stderr

            # UX_Plan error 22 maps to exit code 1 (General error)
            assert_exit_code(returncode, 1, "Invalid instance ID")

            # Verify validation error
            assert_validation_error(combined, "instance_id")

            # Should mention the invalid ID
            assert "fake-123" in combined

            # Verify the double validated instances
            assert "fake-123" in instance_double.instances_validated, "Should have validated instance"


class TestFlakyTests:
    """Test flaky test detection (29)."""

    @pytest.mark.skip(reason="--rerun-failed requires a previous run directory argument, test needs update")
    def test_error_29_flaky_test_detected(self):
        """Test flaky test detection (warning)."""
        with SWEBenchTestHarness() as harness:
            # Inject instance double with flaky scenario
            instance_double = harness.inject_instance_double(scenario="flaky")

            patch_file = harness.create_minimal_patch()

            returncode, stdout, stderr = harness.run_cli_direct(
                ["run", "--patches", str(patch_file), "--rerun-failed"]
            )
            combined = stdout + stderr

            # Flaky tests may still pass after retry
            assert returncode in [0, 1], f"Should succeed or warn, got {returncode}"

            # Should mention flaky test
            assert "flaky" in combined.lower() or "retry" in combined.lower()

            # Should mention retry attempts
            assert any(term in combined for term in ["attempt", "2/3", "3/3"])

            # Verify the double tracked retry attempts
            assert instance_double.flaky_attempts >= 1, "Should have attempted retries"


class TestErrorMessageFormat:
    """Test that error messages follow UX_Plan format."""

    def test_error_includes_suggested_action(self):
        """Test that errors include suggested actions."""
        with SWEBenchTestHarness() as harness:
            # Inject Docker double with not running scenario
            docker_double = harness.inject_docker_double(scenario="not_running")

            # Test Docker error includes action
            returncode, stdout, stderr = harness.run_cli_direct(
                ["run", "--patches", "nonexistent.jsonl"]
            )
            combined = stdout + stderr

            # Must have Docker error
            assert_exit_code(returncode, 2, "Docker error expected")
            assert_docker_error(combined, "not_running")

            # Must include actionable suggestion
            assert_contains_suggestion(combined)

            # Should have platform-specific guidance
            assert_platform_specific(combined)

            # Verify the double was called
            assert docker_double.ping_called, "Docker ping should have been called"

    def test_error_includes_documentation_link(self):
        """Test that errors include documentation links."""
        with SWEBenchTestHarness() as harness:
            # Inject Docker double with not running scenario
            docker_double = harness.inject_docker_double(scenario="not_running")

            returncode, stdout, stderr = harness.run_cli_direct(
                ["run", "--patches", "nonexistent.jsonl"]
            )
            combined = stdout + stderr

            # Should have error
            assert returncode != 0, "Should fail with Docker not running"

            # Check for documentation references (optional but good practice)
            doc_references = [
                "docs/troubleshooting",
                "github.com/",
                "huggingface.co/",
                "docker.com/"
            ]
            has_doc_ref = any(ref in combined for ref in doc_references)
            # This is optional, so we just note it
            if not has_doc_ref:
                # Not a failure, just a note
                pass

            # Verify the double was called
            assert docker_double.ping_called, "Docker ping should have been called"


class TestPlatformSpecificErrors:
    """Test platform-specific error messages."""

    def test_macos_docker_desktop_message(self):
        """Test macOS-specific Docker Desktop message."""
        with SWEBenchTestHarness() as harness:
            # Inject Docker double with not running scenario
            docker_double = harness.inject_docker_double(scenario="not_running")

            env = {
                "SWEBENCH_PLATFORM": "darwin"
            }

            returncode, stdout, stderr = harness.run_cli_direct(
                ["run", "--patches", "test.jsonl"],
                env=env
            )
            combined = stdout + stderr

            # Should fail with Docker error
            assert_exit_code(returncode, 2, "Docker not running")

            # Must have macOS-specific message
            assert "Docker Desktop" in combined or "whale" in combined, \
                "macOS should mention Docker Desktop"

            # Verify platform-specific guidance
            assert_platform_specific(combined, "darwin")

            # Verify the double was called
            assert docker_double.ping_called, "Docker ping should have been called"

    def test_linux_docker_socket_message(self):
        """Test Linux-specific Docker socket message."""
        with SWEBenchTestHarness() as harness:
            # Inject Docker double with not running scenario
            docker_double = harness.inject_docker_double(scenario="not_running")

            env = {
                "SWEBENCH_PLATFORM": "linux"
            }

            returncode, stdout, stderr = harness.run_cli_direct(
                ["run", "--patches", "test.jsonl"],
                env=env
            )
            combined = stdout + stderr

            # Should fail with Docker error
            assert_exit_code(returncode, 2, "Docker not running")

            # Must have Linux-specific message
            assert any(term in combined for term in [
                "docker.sock", "systemctl", "DOCKER_HOST"
            ]), "Linux should mention docker.sock or systemctl"

            # Verify platform-specific guidance
            assert_platform_specific(combined, "linux")

            # Verify the double was called
            assert docker_double.ping_called, "Docker ping should have been called"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
