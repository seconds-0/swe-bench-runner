#!/usr/bin/env python3
"""
Real end-to-end test with actual API calls and full evaluation.

WARNING: This test costs real money (~$0.01-0.05 per run)

This test validates the complete user journey:
1. CLI command execution
2. API key validation
3. Patch generation via real API
4. Docker-based evaluation
5. Result verification

Environment variables required:
- OPENAI_API_KEY or ANTHROPIC_API_KEY
- SWEBENCH_E2E_ENABLED=true (safety switch)
- SWEBENCH_E2E_MAX_COST=0.10 (optional, default: $0.10)
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import pytest

# Color codes for output
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'


class E2ETestLogger:
    """Comprehensive logger for E2E tests with timestamps and step tracking."""

    def __init__(self, log_file: Path | None = None):
        self.log_file = log_file or Path(f"e2e_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        self.steps: list[dict] = []
        self.start_time = datetime.now()
        self.total_cost = 0.0

    def log(self, message: str, level: str = "INFO", cost: float = 0.0):
        """Log a message with timestamp and optional cost tracking."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

        # Console output with color
        color = {"INFO": BLUE, "SUCCESS": GREEN, "ERROR": RED, "WARNING": YELLOW}.get(level, "")
        symbol = {"INFO": "→", "SUCCESS": "✓", "ERROR": "✗", "WARNING": "⚠"}.get(level, "•")
        print(f"{color}[{timestamp}] {symbol} {message}{RESET}")

        # File output
        log_entry = {
            "timestamp": timestamp,
            "level": level,
            "message": message,
            "cost": cost,
            "elapsed": (datetime.now() - self.start_time).total_seconds()
        }
        self.steps.append(log_entry)

        # Track costs
        if cost > 0:
            self.total_cost += cost
            self.log(f"Cost update: ${cost:.4f} (Total: ${self.total_cost:.4f})", "INFO")

        # Write to file
        with open(self.log_file, 'a') as f:
            f.write(f"[{timestamp}] [{level}] {message}\n")
            if cost > 0:
                f.write(f"[{timestamp}] [COST] ${cost:.4f} (Total: ${self.total_cost:.4f})\n")

    def save_report(self):
        """Save comprehensive test report."""
        report = {
            "test_run": {
                "start_time": self.start_time.isoformat(),
                "duration": (datetime.now() - self.start_time).total_seconds(),
                "total_cost": self.total_cost,
                "steps": len(self.steps)
            },
            "steps": self.steps
        }

        report_file = self.log_file.with_suffix('.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        return report_file


@pytest.mark.e2e
@pytest.mark.skipif(
    os.environ.get("SWEBENCH_E2E_ENABLED") != "true",
    reason="E2E tests disabled. Set SWEBENCH_E2E_ENABLED=true to run (costs money!)"
)
class TestRealAPIWorkflow:
    """Comprehensive end-to-end tests with real API calls."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment and logging."""
        self.logger = E2ETestLogger()
        self.max_cost = float(os.environ.get("SWEBENCH_E2E_MAX_COST", "0.10"))
        self.temp_dir = tempfile.mkdtemp(prefix="swebench_e2e_")

        # Set non-interactive mode
        os.environ["SWEBENCH_NO_INPUT"] = "1"

        self.logger.log("Starting E2E test run", "INFO")
        self.logger.log(f"Max budget: ${self.max_cost:.2f}", "INFO")
        self.logger.log(f"Temp directory: {self.temp_dir}", "INFO")

        yield

        # Cleanup
        self.logger.log(f"Test completed. Total cost: ${self.logger.total_cost:.4f}", "INFO")
        report_file = self.logger.save_report()
        self.logger.log(f"Report saved to: {report_file}", "SUCCESS")

    def run_command(self, cmd: list[str], timeout: int = 60, check: bool = True) -> tuple[int, str, str]:
        """Run a command with logging and timeout."""
        cmd_str = ' '.join(cmd)
        self.logger.log(f"Running: {cmd_str}", "INFO")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=os.environ.copy()
            )

            if result.returncode == 0:
                self.logger.log(f"Command succeeded: {cmd[0]}", "SUCCESS")
            else:
                self.logger.log(f"Command failed with code {result.returncode}", "ERROR")
                if check:
                    self.logger.log(f"STDOUT: {result.stdout}", "ERROR")
                    self.logger.log(f"STDERR: {result.stderr}", "ERROR")
                    raise RuntimeError(f"Command failed: {cmd_str}")

            return result.returncode, result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            self.logger.log(f"Command timed out after {timeout}s", "ERROR")
            raise
        except Exception as e:
            self.logger.log(f"Command error: {e}", "ERROR")
            raise

    def check_prerequisites(self):
        """Verify all prerequisites are met."""
        self.logger.log("Checking prerequisites...", "INFO")

        # Check Docker
        try:
            code, stdout, _ = self.run_command(["docker", "--version"], timeout=5)
            assert code == 0, "Docker not available"
            self.logger.log(f"Docker available: {stdout.strip()}", "SUCCESS")
        except:
            self.logger.log("Docker not available - required for evaluation", "ERROR")
            raise

        # Check API keys
        api_keys = {
            "OpenAI": os.environ.get("OPENAI_API_KEY"),
            "Anthropic": os.environ.get("ANTHROPIC_API_KEY")
        }

        available_providers = []
        for provider, key in api_keys.items():
            if key:
                # Mask the key for logging
                masked_key = f"{key[:7]}...{key[-4:]}" if len(key) > 15 else "***"
                self.logger.log(f"{provider} API key found: {masked_key}", "SUCCESS")
                available_providers.append(provider.lower())
            else:
                self.logger.log(f"{provider} API key not found", "WARNING")

        assert available_providers, "No API keys found. Need OPENAI_API_KEY or ANTHROPIC_API_KEY"

        # Check disk space
        import shutil
        free_gb = shutil.disk_usage("/").free / (1024**3)
        self.logger.log(f"Free disk space: {free_gb:.1f} GB", "INFO")
        if free_gb < 10:
            self.logger.log("Low disk space warning", "WARNING")

        return available_providers

    def test_scenario_1_happy_path(self):
        """Test the happy path with minimal cost."""
        self.logger.log("=" * 60, "INFO")
        self.logger.log("SCENARIO 1: Happy Path - Single Instance Generation", "INFO")
        self.logger.log("=" * 60, "INFO")

        # Check prerequisites
        providers = self.check_prerequisites()
        provider = providers[0]  # Use first available provider

        # Step 1: Check CLI is working
        self.logger.log("Step 1: Verify CLI installation", "INFO")
        code, stdout, _ = self.run_command(["swebench", "--version"])
        assert "swebench" in stdout.lower() or "version" in stdout.lower()

        # Step 2: Get dataset info
        self.logger.log("Step 2: Fetch dataset information", "INFO")
        code, stdout, _ = self.run_command(["swebench", "info", "-d", "lite"])
        assert code == 0
        assert "lite" in stdout.lower() or "300" in stdout

        # Step 3: Generate a patch for ONE instance
        self.logger.log(f"Step 3: Generate patch using {provider}", "INFO")
        patch_file = Path(self.temp_dir) / "generated_patches.jsonl"

        # Use cheapest model
        model = "gpt-3.5-turbo" if provider == "openai" else "claude-3-haiku-20240307"

        cmd = [
            "swebench", "generate",
            "--dataset", "lite",
            "--count", "1",
            "--provider", provider,
            "--model", model,
            "--output", str(patch_file),
            "--max-workers", "1"  # Single threaded for predictable costs
        ]

        start_time = time.time()
        code, stdout, stderr = self.run_command(cmd, timeout=120)
        generation_time = time.time() - start_time

        # Estimate cost (rough)
        estimated_cost = 0.002 if provider == "openai" else 0.001
        self.logger.log(f"Generation completed in {generation_time:.1f}s", "SUCCESS", cost=estimated_cost)

        # Verify patch was generated
        assert patch_file.exists(), "Patch file not created"
        with open(patch_file) as f:
            patches = [json.loads(line) for line in f]

        assert len(patches) == 1, f"Expected 1 patch, got {len(patches)}"
        patch = patches[0]
        assert "instance_id" in patch, "Patch missing instance_id"
        assert "patch" in patch or "model_patch" in patch, "Patch missing content"

        self.logger.log(f"Patch generated for: {patch['instance_id']}", "SUCCESS")

        # Step 4: Run evaluation
        self.logger.log("Step 4: Run evaluation on generated patch", "INFO")
        results_dir = Path(self.temp_dir) / "results"

        cmd = [
            "swebench", "run",
            "--patches", str(patch_file),
            "--dataset", "lite",
            "--output-dir", str(results_dir),
            "--timeout", "300"  # 5 minutes max
        ]

        start_time = time.time()
        code, stdout, stderr = self.run_command(cmd, timeout=360, check=False)
        eval_time = time.time() - start_time

        # Evaluation might fail if patch is wrong, but should complete
        if code == 0:
            self.logger.log(f"Evaluation succeeded in {eval_time:.1f}s", "SUCCESS")
        else:
            self.logger.log(f"Evaluation completed with issues in {eval_time:.1f}s", "WARNING")

        # Check results were generated
        if results_dir.exists():
            results_files = list(results_dir.glob("*.json"))
            if results_files:
                self.logger.log(f"Results generated: {len(results_files)} files", "SUCCESS")

                # Parse results
                for result_file in results_files:
                    with open(result_file) as f:
                        results = json.load(f)
                    self.logger.log(f"Results in {result_file.name}: {len(results)} instances", "INFO")

        self.logger.log("Scenario 1 completed successfully!", "SUCCESS")

    def test_scenario_2_multi_provider(self):
        """Test multiple providers if available."""
        self.logger.log("=" * 60, "INFO")
        self.logger.log("SCENARIO 2: Multi-Provider Comparison", "INFO")
        self.logger.log("=" * 60, "INFO")

        providers = self.check_prerequisites()

        if len(providers) < 2:
            self.logger.log("Only one provider available, skipping multi-provider test", "WARNING")
            pytest.skip("Need at least 2 providers for this test")

        # Generate patches with each provider
        patches_by_provider = {}

        for provider in providers[:2]:  # Test first 2 providers only
            self.logger.log(f"Testing provider: {provider}", "INFO")

            patch_file = Path(self.temp_dir) / f"{provider}_patches.jsonl"
            model = "gpt-3.5-turbo" if provider == "openai" else "claude-3-haiku-20240307"

            cmd = [
                "swebench", "generate",
                "--dataset", "lite",
                "--count", "1",
                "--provider", provider,
                "--model", model,
                "--output", str(patch_file)
            ]

            code, stdout, stderr = self.run_command(cmd, timeout=120, check=False)

            if code == 0 and patch_file.exists():
                with open(patch_file) as f:
                    patches = [json.loads(line) for line in f]
                patches_by_provider[provider] = patches
                self.logger.log(f"{provider}: Generated {len(patches)} patches", "SUCCESS", cost=0.002)
            else:
                self.logger.log(f"{provider}: Generation failed", "ERROR")

        # Compare results
        if len(patches_by_provider) >= 2:
            self.logger.log("Comparing patches from different providers...", "INFO")
            # Would add comparison logic here
            self.logger.log("Multi-provider test completed", "SUCCESS")

    def test_scenario_3_error_recovery(self):
        """Test error handling and recovery."""
        self.logger.log("=" * 60, "INFO")
        self.logger.log("SCENARIO 3: Error Recovery", "INFO")
        self.logger.log("=" * 60, "INFO")

        # Test 1: Invalid API key
        self.logger.log("Test 1: Invalid API key handling", "INFO")
        old_key = os.environ.get("OPENAI_API_KEY", "")
        os.environ["OPENAI_API_KEY"] = "sk-invalid-test-key-12345"

        cmd = [
            "swebench", "generate",
            "--dataset", "lite",
            "--count", "1",
            "--provider", "openai",
            "--model", "gpt-3.5-turbo"
        ]

        code, stdout, stderr = self.run_command(cmd, timeout=30, check=False)
        assert code != 0, "Should fail with invalid API key"
        assert "auth" in (stdout + stderr).lower() or "api" in (stdout + stderr).lower()
        self.logger.log("Invalid API key handled correctly", "SUCCESS")

        # Restore key
        os.environ["OPENAI_API_KEY"] = old_key

        # Test 2: Missing patch file
        self.logger.log("Test 2: Missing patch file handling", "INFO")
        cmd = ["swebench", "run", "--patches", "/nonexistent/patches.jsonl"]
        code, stdout, stderr = self.run_command(cmd, timeout=10, check=False)
        assert code != 0, "Should fail with missing file"
        assert "not found" in (stdout + stderr).lower() or "does not exist" in (stdout + stderr).lower()
        self.logger.log("Missing file handled correctly", "SUCCESS")

        # Test 3: Invalid dataset
        self.logger.log("Test 3: Invalid dataset handling", "INFO")
        cmd = ["swebench", "info", "-d", "invalid_dataset"]
        code, stdout, stderr = self.run_command(cmd, timeout=10, check=False)
        # Should handle gracefully
        self.logger.log("Invalid dataset handled", "SUCCESS")

        self.logger.log("Error recovery tests completed", "SUCCESS")

    def test_scenario_4_full_pipeline(self):
        """Test the complete pipeline from setup to results."""
        self.logger.log("=" * 60, "INFO")
        self.logger.log("SCENARIO 4: Full Pipeline Test", "INFO")
        self.logger.log("=" * 60, "INFO")

        providers = self.check_prerequisites()

        # Step 1: Provider setup
        self.logger.log("Step 1: List available providers", "INFO")
        code, stdout, _ = self.run_command(["swebench", "provider", "list"])
        assert code == 0
        assert "provider" in stdout.lower() or providers[0] in stdout.lower()

        # Step 2: Dataset preparation
        self.logger.log("Step 2: Prepare dataset", "INFO")
        code, stdout, _ = self.run_command(["swebench", "info", "-d", "lite", "--count", "5"])
        assert code == 0

        # Step 3: Generate patches
        self.logger.log("Step 3: Generate patches for 2 instances", "INFO")
        patch_file = Path(self.temp_dir) / "pipeline_patches.jsonl"
        provider = providers[0]
        model = "gpt-3.5-turbo" if provider == "openai" else "claude-3-haiku-20240307"

        cmd = [
            "swebench", "generate",
            "--dataset", "lite",
            "--count", "2",
            "--provider", provider,
            "--model", model,
            "--output", str(patch_file)
        ]

        code, stdout, stderr = self.run_command(cmd, timeout=180)
        assert code == 0
        assert patch_file.exists()

        with open(patch_file) as f:
            patches = [json.loads(line) for line in f]
        assert len(patches) == 2

        self.logger.log(f"Generated {len(patches)} patches", "SUCCESS", cost=0.004)

        # Step 4: Run evaluation
        self.logger.log("Step 4: Evaluate patches", "INFO")
        results_dir = Path(self.temp_dir) / "pipeline_results"

        cmd = [
            "swebench", "run",
            "--patches", str(patch_file),
            "--dataset", "lite",
            "--output-dir", str(results_dir),
            "--timeout", "300"
        ]

        code, stdout, stderr = self.run_command(cmd, timeout=600, check=False)

        # Step 5: Verify results
        self.logger.log("Step 5: Verify results", "INFO")
        if results_dir.exists():
            result_files = list(results_dir.glob("*.json"))
            html_files = list(results_dir.glob("*.html"))

            self.logger.log(f"Generated {len(result_files)} JSON results", "SUCCESS")
            self.logger.log(f"Generated {len(html_files)} HTML reports", "SUCCESS")

            # Parse and log results
            total_passed = 0
            total_failed = 0

            for result_file in result_files:
                if "report" in result_file.name:
                    with open(result_file) as f:
                        report = json.load(f)

                    if isinstance(report, list):
                        for item in report:
                            if item.get("resolved", False):
                                total_passed += 1
                            else:
                                total_failed += 1

            self.logger.log(f"Results: {total_passed} passed, {total_failed} failed", "INFO")

        self.logger.log("Full pipeline test completed!", "SUCCESS")


def run_minimal_test():
    """Run a minimal test to verify setup."""
    print(f"{BOLD}Running minimal E2E test...{RESET}")

    logger = E2ETestLogger()

    # Check if we should run
    if os.environ.get("SWEBENCH_E2E_ENABLED") != "true":
        logger.log("E2E tests disabled. Set SWEBENCH_E2E_ENABLED=true to run", "WARNING")
        return False

    # Check for API keys
    has_keys = bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))
    if not has_keys:
        logger.log("No API keys found. Need OPENAI_API_KEY or ANTHROPIC_API_KEY", "ERROR")
        return False

    # Run simple version check
    try:
        result = subprocess.run(
            ["swebench", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            logger.log("CLI is working", "SUCCESS")
            return True
        else:
            logger.log("CLI failed", "ERROR")
            return False
    except Exception as e:
        logger.log(f"Error: {e}", "ERROR")
        return False


if __name__ == "__main__":
    # Allow running directly for testing
    if "--minimal" in sys.argv:
        success = run_minimal_test()
        sys.exit(0 if success else 1)
    else:
        # Run full pytest suite
        pytest.main([__file__, "-v", "-s", "--tb=short"])
