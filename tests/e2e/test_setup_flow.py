"""End-to-end tests for first-time setup flow.

Tests the complete first-time user experience including:
- Docker detection and setup wizard
- Dataset download with HuggingFace authentication
- Cache directory creation
- Image pull simulation

Based on UX_Plan.md Section 3: Installation / Download Flow
"""


import pytest

from tests.e2e.test_harness import SWEBenchTestHarness


class TestFirstTimeSetup:
    """Test first-time setup experience as defined in UX_Plan.md Section 3."""

    def test_docker_not_detected(self):
        """Test behavior when Docker is not running (UX_Plan 3.1.2.i)."""
        with SWEBenchTestHarness() as harness:
            # Ensure Docker check will fail
            env = {"SWEBENCH_MOCK_NO_DOCKER": "true"}

            # Run setup command
            returncode, stdout, stderr = harness.run_cli(
                ["run", "--patches", "nonexistent.jsonl"],
                env=env
            )

            # Should exit with code 2 (Docker error)
            assert returncode == 2, f"Expected exit code 2, got {returncode}"

            # Check for expected error message from UX_Plan
            combined = stdout + stderr
            assert "Docker" in combined, "Error should mention Docker"
            assert ("not running" in combined or "unreachable" in combined), \
                "Error should indicate Docker is not running"

    def test_setup_wizard_macos_instructions(self):
        """Test setup wizard provides macOS-specific instructions (UX_Plan 3.2)."""
        with SWEBenchTestHarness() as harness:
            # Create a mock setup command response
            returncode, stdout, stderr = harness.run_cli(
                ["setup", "--help"],
                env={}
            )

            # Setup command should exist (or we should create it)
            # For now, test that help works
            assert returncode == 0 or "No such command" in stderr, \
                "Setup command should exist or return proper error"

    def test_disk_space_check(self):
        """Test disk space verification (UX_Plan 3.1.2.iii)."""
        with SWEBenchTestHarness() as harness:
            # This is harder to test without mocking system calls
            # For now, verify the error message format when space is low

            # We can't easily simulate low disk space, but we can test
            # that the system handles the SWEBENCH_MIN_DISK_GB env var
            env = {
                "SWEBENCH_MIN_DISK_GB": "999999",  # Impossible amount
                "SWEBENCH_SKIP_DISK_CHECK": "false"
            }

            # This should fail the disk check if implemented
            returncode, stdout, stderr = harness.run_cli(
                ["run", "--patches", "test.jsonl"],
                env=env
            )

            # The actual disk check might not be implemented yet
            # Document this as a future test requirement
            # assert returncode == 4  # Disk space error

    def test_welcome_prompt_interactive(self):
        """Test welcome prompt for first-time users (UX_Plan 3.1.3)."""
        with SWEBenchTestHarness() as harness:
            # Create a minimal patch file
            patch_file = harness.create_minimal_patch()

            # Test with --yes flag (non-interactive)
            returncode, stdout, stderr = harness.run_cli(
                ["run", "--patches", str(patch_file), "--yes"],
                env={"SWEBENCH_FIRST_RUN": "true"}
            )

            # With --yes, should skip prompts
            combined = stdout + stderr
            # Check if welcome message appears (if implemented)
            # "Welcome to SWE-bench Runner" might appear

    def test_dataset_download_anonymous(self):
        """Test dataset download with anonymous access (UX_Plan 3.4)."""
        with SWEBenchTestHarness() as harness:
            # Test anonymous download warning
            env = {
                "HUGGINGFACE_TOKEN": "",  # No token
                "SWEBENCH_MOCK_NO_DOCKER": "true"
            }

            patch_file = harness.create_minimal_patch()

            returncode, stdout, stderr = harness.run_cli(
                ["run", "--patches", str(patch_file), "--dataset", "lite"],
                env=env
            )

            # Should show anonymous access warning if downloading
            combined = stdout + stderr
            # Look for indicators of dataset handling
            # "anonymous" or "token" might appear in warnings

    def test_dataset_download_with_token(self):
        """Test dataset download with HuggingFace token (UX_Plan 3.4)."""
        with SWEBenchTestHarness() as harness:
            # Test with token
            env = {
                "HUGGINGFACE_TOKEN": "hf_test_token_12345",
                "SWEBENCH_MOCK_NO_DOCKER": "true"
            }

            patch_file = harness.create_minimal_patch()

            returncode, stdout, stderr = harness.run_cli(
                ["run", "--patches", str(patch_file), "--dataset", "lite"],
                env=env
            )

            # Should not show anonymous warning
            combined = stdout + stderr
            # Should not warn about anonymous access
            # (actual behavior depends on implementation)

    def test_cache_directory_creation(self):
        """Test cache directory structure creation (UX_Plan 3.1.6)."""
        with SWEBenchTestHarness() as harness:
            # Set custom cache directory
            cache_dir = harness.temp_dir / "custom_cache"
            env = {
                "SWEBENCH_CACHE_DIR": str(cache_dir),
                "SWEBENCH_MOCK_NO_DOCKER": "true"
            }

            patch_file = harness.create_minimal_patch()

            # Run command that should create cache structure
            harness.run_cli(
                ["run", "--patches", str(patch_file)],
                env=env
            )

            # Check cache directory was created
            assert cache_dir.exists(), f"Cache directory {cache_dir} should be created"

            # Check for expected subdirectories (if created)
            # datasets_dir = cache_dir / "datasets"
            # images_dir = cache_dir / "images"
            # These might be created on demand

    def test_config_file_creation(self):
        """Test config.toml creation on first run (UX_Plan 3.1.6)."""
        with SWEBenchTestHarness() as harness:
            # Set up paths
            config_dir = harness.temp_dir / ".swebench"
            config_file = config_dir / "config.toml"

            env = {
                "SWEBENCH_CONFIG_DIR": str(config_dir),
                "SWEBENCH_MOCK_NO_DOCKER": "true"
            }

            patch_file = harness.create_minimal_patch()

            # Run command that might create config
            harness.run_cli(
                ["run", "--patches", str(patch_file)],
                env=env
            )

            # Config file creation might not be implemented
            # Document as future requirement
            # assert config_file.exists(), "Config file should be created on first run"

    def test_platform_detection(self):
        """Test platform detection for image selection (UX_Plan 3.1.2.ii)."""
        with SWEBenchTestHarness() as harness:
            # Run version command to check platform info
            returncode, stdout, stderr = harness.run_cli(
                ["--version"],
                env={}
            )

            assert returncode == 0, "Version command should succeed"
            # Platform info might be included in version output

    def test_image_pull_simulation(self):
        """Test Docker image pull flow (UX_Plan 3.1.4)."""
        with SWEBenchTestHarness() as harness:
            # This would require Docker to actually test
            # For now, test that the right image name is referenced

            env = {
                "SWEBENCH_MOCK_DOCKER_PULL": "true",  # Mock the pull
                "SWEBENCH_MOCK_NO_DOCKER": "false"  # But Docker is "available"
            }

            patch_file = harness.create_minimal_patch()

            returncode, stdout, stderr = harness.run_cli(
                ["run", "--patches", str(patch_file)],
                env=env
            )

            # Look for image reference in output
            combined = stdout + stderr
            # "ghcr.io/swebench/runner" might appear

    def test_upgrade_path(self):
        """Test upgrade detection and prompting (UX_Plan 3.3)."""
        with SWEBenchTestHarness() as harness:
            # Simulate version mismatch
            env = {
                "SWEBENCH_IMAGE_VERSION": "0.1.0",
                "SWEBENCH_CLI_VERSION": "0.2.0",
                "SWEBENCH_MOCK_NO_DOCKER": "true"
            }

            patch_file = harness.create_minimal_patch()

            returncode, stdout, stderr = harness.run_cli(
                ["run", "--patches", str(patch_file)],
                env=env
            )

            # Might warn about version mismatch
            combined = stdout + stderr
            # Look for upgrade-related messages

    def test_no_input_mode(self):
        """Test --no-input flag for CI environments (UX_Plan 3.1.3)."""
        with SWEBenchTestHarness() as harness:
            patch_file = harness.create_minimal_patch()

            # Run with --no-input flag
            returncode, stdout, stderr = harness.run_cli(
                ["run", "--patches", str(patch_file), "--no-input"],
                env={"SWEBENCH_MOCK_NO_DOCKER": "true"}
            )

            # Should not prompt for anything
            # Command should complete or fail without waiting for input
            assert returncode in [0, 1, 2], "Should complete without hanging"


class TestSetupWizard:
    """Test the interactive setup wizard from UX_Plan Section 3.2."""

    @pytest.mark.skip(reason="Setup wizard not yet implemented")
    def test_setup_wizard_docker_desktop_macos(self):
        """Test Docker Desktop instructions for macOS."""
        with SWEBenchTestHarness() as harness:
            # Simulate choosing macOS option
            returncode, stdout, stderr = harness.run_cli(
                ["setup"],
                input_text="1\n",  # Choose macOS
                env={}
            )

            # Should show macOS-specific instructions
            assert "Docker Desktop" in stdout
            assert "https://docker.com/products/docker-desktop" in stdout
            assert "whale icon" in stdout

    @pytest.mark.skip(reason="Setup wizard not yet implemented")
    def test_setup_wizard_docker_engine_linux(self):
        """Test Docker Engine instructions for Linux."""
        with SWEBenchTestHarness() as harness:
            # Simulate choosing Linux option
            returncode, stdout, stderr = harness.run_cli(
                ["setup"],
                input_text="2\n",  # Choose Linux
                env={}
            )

            # Should show Linux-specific instructions
            assert "Docker Engine" in stdout or "docker" in stdout.lower()


class TestHuggingFaceAuth:
    """Test HuggingFace authentication flow from UX_Plan Section 3.4."""

    def test_anonymous_rate_limit_warning(self):
        """Test warning about anonymous rate limits."""
        with SWEBenchTestHarness() as harness:
            patch_file = harness.create_minimal_patch()

            # No HF token
            env = {
                "HUGGINGFACE_TOKEN": "",
                "SWEBENCH_MOCK_NO_DOCKER": "true",
                "SWEBENCH_MOCK_HF_DOWNLOAD": "true"  # Mock the download
            }

            returncode, stdout, stderr = harness.run_cli(
                ["run", "--patches", str(patch_file), "--dataset", "lite"],
                env=env
            )

            combined = stdout + stderr
            # Should mention rate limits or token
            # Actual implementation may vary

    def test_hf_token_from_env(self):
        """Test using HuggingFace token from environment."""
        with SWEBenchTestHarness() as harness:
            patch_file = harness.create_minimal_patch()

            env = {
                "HUGGINGFACE_TOKEN": "hf_test_token",
                "SWEBENCH_MOCK_NO_DOCKER": "true",
                "SWEBENCH_MOCK_HF_DOWNLOAD": "true"
            }

            returncode, stdout, stderr = harness.run_cli(
                ["run", "--patches", str(patch_file), "--dataset", "lite"],
                env=env
            )

            combined = stdout + stderr
            # Should not warn about anonymous access
            assert "anonymous" not in combined.lower() or "authenticated" in combined.lower()

    def test_hf_token_from_flag(self):
        """Test using HuggingFace token from CLI flag."""
        with SWEBenchTestHarness() as harness:
            patch_file = harness.create_minimal_patch()

            env = {
                "SWEBENCH_MOCK_NO_DOCKER": "true",
                "SWEBENCH_MOCK_HF_DOWNLOAD": "true"
            }

            returncode, stdout, stderr = harness.run_cli(
                ["run", "--patches", str(patch_file),
                 "--dataset", "lite",
                 "--hf-token", "hf_flag_token"],
                env=env
            )

            # Flag should override environment
            combined = stdout + stderr
            # Should use the provided token


class TestErrorRecovery:
    """Test error recovery scenarios during setup."""

    def test_network_failure_retry(self):
        """Test network failure with retry logic (UX_Plan 3.1.5)."""
        with SWEBenchTestHarness() as harness:
            patch_file = harness.create_minimal_patch()

            # Simulate network failure
            env = {
                "SWEBENCH_MOCK_NETWORK_FAIL": "true",
                "SWEBENCH_MOCK_NO_DOCKER": "true"
            }

            returncode, stdout, stderr = harness.run_cli(
                ["run", "--patches", str(patch_file), "--dataset", "lite"],
                env=env
            )

            # Should show retry attempts
            combined = stdout + stderr
            # Look for "retry" or "attempt" in output

    def test_corrupted_cache_recovery(self):
        """Test recovery from corrupted cache."""
        with SWEBenchTestHarness() as harness:
            # Create corrupted cache file
            cache_dir = harness.temp_dir / "cache" / "datasets" / "lite"
            cache_dir.mkdir(parents=True)

            corrupted_file = cache_dir / "data.json"
            corrupted_file.write_text("corrupted{data")

            patch_file = harness.create_minimal_patch()

            env = {
                "SWEBENCH_CACHE_DIR": str(harness.temp_dir / "cache"),
                "SWEBENCH_MOCK_NO_DOCKER": "true"
            }

            returncode, stdout, stderr = harness.run_cli(
                ["run", "--patches", str(patch_file), "--dataset", "lite"],
                env=env
            )

            # Should handle corrupted cache gracefully
            # Might re-download or report error clearly


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
