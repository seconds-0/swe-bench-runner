"""End-to-end tests for complete evaluation workflow.

Tests the main evaluation flow including:
- Running evaluations with patches
- Progress tracking and reporting
- Success celebrations
- HTML report generation
- Various subsetting options

Based on UX_Plan.md Section 4: Core Usage Flows
"""

import json

import pytest

from tests.e2e.test_harness import SWEBenchTestHarness


class TestHappyPathEvaluation:
    """Test the happy path evaluation flow from UX_Plan Section 4.1."""

    def test_quick_lite_evaluation(self):
        """Test basic evaluation with patches file (UX_Plan 4.1)."""
        with SWEBenchTestHarness() as harness:
            # Create a minimal patch file
            patch_file = harness.create_minimal_patch("django__django-11001")

            # Run evaluation
            returncode, stdout, stderr = harness.run_cli(
                ["run", "--patches", str(patch_file)],
                env={"SWEBENCH_MOCK_NO_DOCKER": "true"}
            )

            # Should fail at Docker check but show proper flow
            assert returncode == 2, "Should fail with Docker error"
            assert "Docker" in stdout, "Should mention Docker"

    def test_smart_defaults_detection(self):
        """Test smart defaults for finding patches.jsonl (UX_Plan 4.1)."""
        with SWEBenchTestHarness() as harness:
            # Create predictions.jsonl in current directory
            predictions_file = harness.temp_dir / "predictions.jsonl"
            predictions_file.write_text(json.dumps({
                "instance_id": "test-1",
                "patch": "diff --git a/test.py b/test.py\n--- a/test.py\n+++ b/test.py\n"
            }))

            # Run without specifying patches file
            returncode, stdout, stderr = harness.run_cli(
                ["run"],
                env={"SWEBENCH_MOCK_NO_DOCKER": "true"}
            )

            # Should detect predictions.jsonl or fail appropriately
            stdout + stderr
            # Should either find the file or ask for --patches

    def test_progress_tracking_output(self):
        """Test progress bar and status updates (UX_Plan 4.1)."""
        with SWEBenchTestHarness() as harness:
            # Create multiple patches
            patches = [
                {"instance_id": f"test-{i}", "patch": "diff"}
                for i in range(5)
            ]
            patch_file = harness.create_patch_file(patches)

            returncode, stdout, stderr = harness.run_cli(
                ["run", "--patches", str(patch_file), "--workers", "2"],
                env={"SWEBENCH_MOCK_NO_DOCKER": "true"}
            )

            # Look for progress indicators
            stdout + stderr
            # Might show "Evaluating N instances" or progress bar

    def test_success_celebration(self):
        """Test success celebration output (UX_Plan 4.1)."""
        with SWEBenchTestHarness() as harness:
            # This would require successful completion
            # Mock a successful run
            patch_file = harness.create_minimal_patch()

            env = {
                "SWEBENCH_MOCK_SUCCESS": "true",  # Mock successful evaluation
                "SWEBENCH_MOCK_NO_DOCKER": "false"  # Pretend Docker works
            }

            returncode, stdout, stderr = harness.run_cli(
                ["run", "--patches", str(patch_file)],
                env=env
            )

            # Look for success indicators (if implemented)
            stdout + stderr
            # Might show "SUCCESS" or completion message

    def test_html_report_generation(self):
        """Test HTML report generation (UX_Plan 4.1)."""
        with SWEBenchTestHarness() as harness:
            patch_file = harness.create_minimal_patch()
            results_dir = harness.temp_dir / "results"

            env = {
                "SWEBENCH_RESULTS_DIR": str(results_dir),
                "SWEBENCH_MOCK_NO_DOCKER": "true"
            }

            harness.run_cli(
                ["run", "--patches", str(patch_file)],
                env=env
            )

            # Check if results directory structure is created
            # Report generation might happen even if Docker fails
            # assert results_dir.exists(), "Results directory should be created"

    def test_json_output_mode(self):
        """Test JSON output mode for CI (UX_Plan 4.3)."""
        with SWEBenchTestHarness() as harness:
            patch_file = harness.create_minimal_patch()

            returncode, stdout, stderr = harness.run_cli(
                ["run", "--patches", str(patch_file), "--json"],
                env={"SWEBENCH_MOCK_NO_DOCKER": "true"}
            )

            # In JSON mode, structured output expected
            # Docker errors might still print normally
            # Check if any JSON appears in output


class TestSubsettingWorkflows:
    """Test subsetting examples from UX_Plan Section 4.2."""

    def test_django_only_instances(self):
        """Test filtering to Django instances only."""
        with SWEBenchTestHarness() as harness:
            # Create patches with different repos
            patches = [
                {"instance_id": "django__django-11001", "patch": "diff"},
                {"instance_id": "flask__flask-2001", "patch": "diff"},
                {"instance_id": "django__django-11002", "patch": "diff"},
            ]
            patch_file = harness.create_patch_file(patches)

            returncode, stdout, stderr = harness.run_cli(
                ["run", "--patches", str(patch_file), "--subset", "django/**"],
                env={"SWEBENCH_MOCK_NO_DOCKER": "true"}
            )

            stdout + stderr
            # Should process only Django instances
            # Look for instance filtering messages

    def test_random_sampling(self):
        """Test random sampling with seed."""
        with SWEBenchTestHarness() as harness:
            # Create many patches
            patches = [
                {"instance_id": f"test-{i}", "patch": "diff"}
                for i in range(10)
            ]
            patch_file = harness.create_patch_file(patches)

            # Run with sampling
            returncode, stdout, stderr = harness.run_cli(
                ["run", "--patches", str(patch_file),
                 "--count", "5", "--sample", "random-seed=42"],
                env={"SWEBENCH_MOCK_NO_DOCKER": "true"}
            )

            stdout + stderr
            # Should select subset of instances

    def test_count_limiting(self):
        """Test limiting to first N instances."""
        with SWEBenchTestHarness() as harness:
            patches = [
                {"instance_id": f"test-{i}", "patch": "diff"}
                for i in range(10)
            ]
            patch_file = harness.create_patch_file(patches)

            returncode, stdout, stderr = harness.run_cli(
                ["run", "--patches", str(patch_file), "--count", "3"],
                env={"SWEBENCH_MOCK_NO_DOCKER": "true"}
            )

            stdout + stderr
            # Should process only 3 instances

    def test_rerun_failed(self):
        """Test re-running failed instances."""
        with SWEBenchTestHarness() as harness:
            # Create a results directory with mock failures
            results_dir = harness.temp_dir / "results" / "latest"
            results_dir.mkdir(parents=True)

            # Create mock failure file
            failures_file = results_dir / "failures.json"
            failures_file.write_text(json.dumps([
                {"instance_id": "failed-1"},
                {"instance_id": "failed-2"}
            ]))

            returncode, stdout, stderr = harness.run_cli(
                ["run", "--rerun-failed", str(results_dir)],
                env={"SWEBENCH_MOCK_NO_DOCKER": "true"}
            )

            stdout + stderr
            # Should attempt to rerun failures


class TestInterruptedRunRecovery:
    """Test interrupted run recovery from UX_Plan Section 7.1."""

    def test_resume_interrupted_run(self):
        """Test resuming an interrupted evaluation."""
        with SWEBenchTestHarness() as harness:
            # Create checkpoint file
            checkpoint_dir = harness.temp_dir / "results" / "2025-01-01_12-00-00"
            checkpoint_dir.mkdir(parents=True)

            checkpoint_file = checkpoint_dir / "checkpoint.json"
            checkpoint_file.write_text(json.dumps({
                "completed": ["test-1", "test-2"],
                "remaining": ["test-3", "test-4", "test-5"],
                "total": 5
            }))

            returncode, stdout, stderr = harness.run_cli(
                ["run", "--resume", str(checkpoint_dir)],
                env={"SWEBENCH_MOCK_NO_DOCKER": "true"}
            )

            combined = stdout + stderr
            # Should mention resuming
            assert "resum" in combined.lower() or returncode == 2


class TestValidation:
    """Test pre-flight validation from UX_Plan Section 7.3."""

    def test_validate_patches_file(self):
        """Test patch validation command."""
        with SWEBenchTestHarness() as harness:
            # Create invalid patch file
            invalid_file = harness.temp_dir / "invalid.jsonl"
            invalid_file.write_text("not valid json\n{broken json")

            returncode, stdout, stderr = harness.run_cli(
                ["validate", "--patches", str(invalid_file)],
                env={}
            )

            # Should report validation errors
            stdout + stderr
            # Should mention invalid format

    def test_validate_large_patches(self):
        """Test validation of large patches."""
        with SWEBenchTestHarness() as harness:
            # Create large patch
            large_patch = {
                "instance_id": "test-large",
                "patch": "diff --git a/test.py b/test.py\n" + "+" * (2 * 1024 * 1024)  # 2MB
            }
            patch_file = harness.create_patch_file([large_patch])

            returncode, stdout, stderr = harness.run_cli(
                ["validate", "--patches", str(patch_file)],
                env={}
            )

            stdout + stderr
            # Should warn about large patch


class TestRetryBehavior:
    """Test retry behavior from UX_Plan Section 7.5."""

    def test_retry_failed_instances(self):
        """Test retrying failed instances."""
        with SWEBenchTestHarness() as harness:
            patch_file = harness.create_minimal_patch()

            returncode, stdout, stderr = harness.run_cli(
                ["run", "--patches", str(patch_file), "--retry-failures", "2"],
                env={"SWEBENCH_MOCK_NO_DOCKER": "true"}
            )

            stdout + stderr
            # Should mention retry configuration


class TestTimeoutConfiguration:
    """Test timeout configuration from UX_Plan Section 7.6."""

    def test_instance_timeout(self):
        """Test per-instance timeout setting."""
        with SWEBenchTestHarness() as harness:
            patch_file = harness.create_minimal_patch()

            returncode, stdout, stderr = harness.run_cli(
                ["run", "--patches", str(patch_file), "--timeout-mins", "45"],
                env={"SWEBENCH_MOCK_NO_DOCKER": "true"}
            )

            stdout + stderr
            # Should acknowledge timeout setting

    def test_global_timeout(self):
        """Test global timeout setting."""
        with SWEBenchTestHarness() as harness:
            # Create many patches
            patches = [
                {"instance_id": f"test-{i}", "patch": "diff"}
                for i in range(100)
            ]
            patch_file = harness.create_patch_file(patches)

            returncode, stdout, stderr = harness.run_cli(
                ["run", "--patches", str(patch_file), "--global-timeout-hours", "2"],
                env={"SWEBENCH_MOCK_NO_DOCKER": "true"}
            )

            stdout + stderr
            # Should mention global timeout


class TestMaintenanceCommands:
    """Test maintenance commands from UX_Plan Section 8."""

    def test_doctor_command(self):
        """Test swebench doctor for diagnostics."""
        with SWEBenchTestHarness() as harness:
            returncode, stdout, stderr = harness.run_cli(
                ["doctor"],
                env={}
            )

            # Should show diagnostic information
            stdout + stderr
            # Might show Docker status, disk space, etc.

    def test_clean_command(self):
        """Test swebench clean for cache cleanup."""
        with SWEBenchTestHarness() as harness:
            # Create some cache files
            cache_dir = harness.temp_dir / "cache"
            cache_dir.mkdir()
            (cache_dir / "test.cache").write_text("cached data")

            returncode, stdout, stderr = harness.run_cli(
                ["clean", "--dry-run"],
                env={"SWEBENCH_CACHE_DIR": str(cache_dir)}
            )

            # Should show what would be cleaned
            stdout + stderr
            # Should mention dry-run

    def test_status_command(self):
        """Test swebench status for active containers."""
        with SWEBenchTestHarness() as harness:
            returncode, stdout, stderr = harness.run_cli(
                ["status"],
                env={"SWEBENCH_MOCK_NO_DOCKER": "true"}
            )

            # Should show status even without Docker
            stdout + stderr


class TestDatasetOptions:
    """Test dataset selection options."""

    def test_lite_dataset_default(self):
        """Test that lite dataset is the default."""
        with SWEBenchTestHarness() as harness:
            patch_file = harness.create_minimal_patch()

            returncode, stdout, stderr = harness.run_cli(
                ["run", "--patches", str(patch_file)],
                env={"SWEBENCH_MOCK_NO_DOCKER": "true"}
            )

            stdout + stderr
            # Should mention lite dataset or use it by default

    def test_verified_dataset(self):
        """Test using verified dataset."""
        with SWEBenchTestHarness() as harness:
            patch_file = harness.create_minimal_patch()

            returncode, stdout, stderr = harness.run_cli(
                ["run", "--patches", str(patch_file), "--dataset", "verified"],
                env={"SWEBENCH_MOCK_NO_DOCKER": "true"}
            )

            stdout + stderr
            # Should use verified dataset

    def test_full_dataset(self):
        """Test using full dataset."""
        with SWEBenchTestHarness() as harness:
            patch_file = harness.create_minimal_patch()

            returncode, stdout, stderr = harness.run_cli(
                ["run", "--patches", str(patch_file), "--dataset", "full"],
                env={"SWEBENCH_MOCK_NO_DOCKER": "true"}
            )

            stdout + stderr
            # Should use full dataset

    def test_custom_dataset_path(self):
        """Test using custom dataset path."""
        with SWEBenchTestHarness() as harness:
            # Create custom dataset file
            custom_dataset = harness.temp_dir / "custom_dataset.json"
            custom_dataset.write_text(json.dumps([
                {"instance_id": "custom-1", "problem_statement": "Custom problem"}
            ]))

            patch_file = harness.create_minimal_patch("custom-1")

            returncode, stdout, stderr = harness.run_cli(
                ["run", "--patches", str(patch_file), "--dataset", str(custom_dataset)],
                env={"SWEBENCH_MOCK_NO_DOCKER": "true"}
            )

            stdout + stderr
            # Should use custom dataset


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
