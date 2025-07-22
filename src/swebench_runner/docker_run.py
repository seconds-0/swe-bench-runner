"""Docker execution engine for SWE-bench evaluations."""

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import docker
from docker.errors import APIError

from . import exit_codes
from .bootstrap import show_memory_warning, show_resource_warning
from .cache import get_cache_dir, get_logs_dir
from .models import EvaluationResult, Patch


def check_docker_running() -> None:
    """Check if Docker daemon is accessible."""
    try:
        client = docker.from_env()
        client.ping()
    except APIError as e:
        error_msg = str(e).lower()
        # Connection refused to Docker daemon means Docker is not running
        if ("connection refused" in error_msg or
            "cannot connect to the docker daemon" in error_msg):
            if platform.system() == "Darwin":
                print("⛔ Docker Desktop not running. Start it from Applications "
                      "and wait for whale icon.")
            else:
                print("⛔ Docker daemon unreachable at /var/run/docker.sock")
            print("ℹ️  Start Docker and try again.")
            sys.exit(exit_codes.DOCKER_NOT_FOUND)
        elif any(term in error_msg for term in [
            "network", "connection", "timeout", "unreachable",
            "resolve", "dns", "timed out"
        ]):
            print("❌ Network error connecting to Docker daemon")
            print("   Check internet connection and Docker network settings")
            sys.exit(exit_codes.NETWORK_ERROR)
        else:
            if platform.system() == "Darwin":
                print("⛔ Docker Desktop not running. Start it from Applications "
                      "and wait for whale icon.")
            else:
                print("⛔ Docker daemon unreachable at /var/run/docker.sock")
            print("ℹ️  Start Docker and try again.")
            sys.exit(exit_codes.DOCKER_NOT_FOUND)
    except Exception as e:
        error_msg = str(e).lower()
        # Connection refused to Docker daemon means Docker is not running
        if ("connection refused" in error_msg or
            "cannot connect to the docker daemon" in error_msg):
            if platform.system() == "Darwin":
                print("⛔ Docker Desktop not running. Start it from Applications "
                      "and wait for whale icon.")
            else:
                print("⛔ Docker daemon unreachable at /var/run/docker.sock")
            print("ℹ️  Start Docker and try again.")
            sys.exit(exit_codes.DOCKER_NOT_FOUND)
        elif any(term in error_msg for term in [
            "network", "connection", "timeout", "unreachable",
            "resolve", "dns"
        ]):
            print(f"❌ Network error connecting to Docker: {e}")
            print("   Check internet connection and Docker network settings")
            sys.exit(exit_codes.NETWORK_ERROR)
        else:
            print(f"❌ Error connecting to Docker: {e}")
            sys.exit(exit_codes.DOCKER_NOT_FOUND)


def load_first_patch(patch_source: str, max_size_mb: int = 5) -> Patch:
    """Load the first patch from a JSONL file or patch directory."""
    try:
        patch_path = Path(patch_source)

        if patch_path.is_file():
            # JSONL file
            with patch_path.open('r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        patch = Patch(
                            instance_id=data['instance_id'],
                            patch=data['patch']
                        )
                        # Validate using the configurable limit
                        patch.validate(max_size_mb=max_size_mb)

                        # Add informational warning for Docker environment
                        # variable limits (~500KB for environment variables)
                        patch_size_kb = len(patch.patch.encode('utf-8')) / 1024
                        if patch_size_kb > 500:
                            print(f"ℹ️  Note: {patch_size_kb:.0f}KB patch may exceed "
                                  "Docker environment limits")
                            print("   Consider using smaller patches or "
                                  "file-based patch application")

                        return patch
            raise ValueError("No patches found in file")

        elif patch_path.is_dir():
            # Patch directory
            patch_files = list(patch_path.glob("*.patch"))
            if not patch_files:
                raise ValueError("No .patch files found in directory")

            # Load the first patch file
            first_patch_file = sorted(patch_files)[0]
            instance_id = first_patch_file.stem  # Remove .patch extension

            with first_patch_file.open('r', encoding='utf-8') as f:
                patch_content = f.read()

            patch = Patch(
                instance_id=instance_id,
                patch=patch_content
            )
            # Validate using the configurable limit
            patch.validate(max_size_mb=max_size_mb)

            # Add informational warning for Docker environment variable limits
            patch_size_kb = len(patch.patch.encode('utf-8')) / 1024
            if patch_size_kb > 500:
                print(f"ℹ️  Note: {patch_size_kb:.0f}KB patch may exceed "
                      "Docker environment limits")
                print("   Consider using smaller patches or "
                      "file-based patch application")

            return patch

        else:
            raise ValueError(f"Path {patch_source} is neither a file nor a directory")

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        error_msg = str(e)
        # Check for our specific error codes
        if error_msg.startswith("PATCH_TOO_LARGE:"):
            # Extract the actual message
            size_info = error_msg.replace("PATCH_TOO_LARGE: ", "")
            print("❌ Patch Too Large")
            print()
            print(f"Your patch {size_info}.")
            print()
            print("Please reduce your patch size and try again.")
        else:
            print(f"Error: Invalid patch format: {e}")
        sys.exit(exit_codes.GENERAL_ERROR)
    except FileNotFoundError:
        print(f"Error: Patch source not found: {patch_source}")
        sys.exit(exit_codes.GENERAL_ERROR)
    except UnicodeDecodeError:
        print("Error: Patch file contains invalid UTF-8 encoding")
        sys.exit(exit_codes.GENERAL_ERROR)


def check_resources() -> None:
    """Check system resources with full CI configurability.

    Environment Variables:
    - CI: Set to "true" to enable CI mode
    - SWEBENCH_SKIP_RESOURCE_CHECK: Set to "true" to skip all checks
    - SWEBENCH_CI_MIN_MEMORY_GB: Override minimum memory for CI (default: 4)
    - SWEBENCH_CI_MIN_DISK_GB: Override minimum disk for CI (default: 20)
    - SWEBENCH_MIN_MEMORY_GB: Override minimum memory for normal mode (default: 8)
    - SWEBENCH_MIN_DISK_GB: Override minimum disk for normal mode (default: 50)
    """
    # Environment checks
    is_ci = os.environ.get("CI") == "true"
    skip_all = os.environ.get("SWEBENCH_SKIP_RESOURCE_CHECK") == "true"

    if skip_all:
        return

    # Fully configurable requirements
    if is_ci:
        min_memory_gb = int(os.environ.get("SWEBENCH_CI_MIN_MEMORY_GB", "4"))
        min_disk_gb = int(os.environ.get("SWEBENCH_CI_MIN_DISK_GB", "20"))
        memory_recommended = int(os.environ.get("SWEBENCH_CI_REC_MEMORY_GB", "8"))
        disk_recommended = int(os.environ.get("SWEBENCH_CI_REC_DISK_GB", "50"))
    else:
        min_memory_gb = int(os.environ.get("SWEBENCH_MIN_MEMORY_GB", "8"))
        min_disk_gb = int(os.environ.get("SWEBENCH_MIN_DISK_GB", "50"))
        memory_recommended = int(os.environ.get("SWEBENCH_REC_MEMORY_GB", "16"))
        disk_recommended = int(os.environ.get("SWEBENCH_REC_DISK_GB", "120"))

    # Check memory
    try:
        import psutil
        mem_gb = psutil.virtual_memory().available / (1024**3)

        if mem_gb < min_memory_gb:
            if is_ci:
                print(f"⚠️  CI Warning: Only {mem_gb:.1f}GB RAM available")
                print(f"   Minimum: {min_memory_gb}GB "
                      f"(configurable via SWEBENCH_CI_MIN_MEMORY_GB)")
                print(f"   Recommended: {memory_recommended}GB")
            else:
                print(f"❌ Critical: Only {mem_gb:.1f}GB RAM available")
                print(f"   Minimum {min_memory_gb}GB required")
                sys.exit(exit_codes.RESOURCE_ERROR)
        elif mem_gb < memory_recommended:
            # Show warning for non-critical low memory
            if is_ci:
                print(f"⚠️  CI Memory: {mem_gb:.1f}GB available, "
                      f"{memory_recommended}GB recommended")
            else:
                show_memory_warning(mem_gb)
    except ImportError:
        pass  # Skip if psutil not available

    # Check disk space
    try:
        free_gb = shutil.disk_usage(".").free / (1024**3)

        if free_gb < min_disk_gb:
            if is_ci:
                print(f"⚠️  CI Warning: Only {free_gb:.1f}GB disk space available")
                print(f"   Minimum: {min_disk_gb}GB "
                      f"(configurable via SWEBENCH_CI_MIN_DISK_GB)")
                print("   Some evaluations may fail due to insufficient space")
            else:
                print(f"❌ Critical: Only {free_gb:.1f}GB free disk space")
                print(f"   Minimum {min_disk_gb}GB required")
                print("   Run: swebench clean --all")
                sys.exit(exit_codes.RESOURCE_ERROR)
        elif free_gb < disk_recommended:
            # Show warning for non-critical low disk space
            if is_ci:
                print(f"⚠️  CI Disk: {free_gb:.1f}GB available, "
                      f"{disk_recommended}GB recommended")
            else:
                show_resource_warning(free_gb)
    except Exception as e:
        # Skip resource checks on error - not critical for operation
        # This allows running in restricted environments where psutil
        # may not be available or system calls may be restricted
        if os.getenv("SWEBENCH_DEBUG"):
            print(f"Debug: Resource check skipped due to: {type(e).__name__}",
                  file=sys.stderr)
        pass  # noqa: S110


def check_swebench_installed() -> bool:
    """Check if SWE-bench harness is installed."""
    try:
        # Safe subprocess call: sys.executable is trusted Python interpreter
        # All arguments are hardcoded strings, no user input
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-m", "swebench.harness.run_evaluation", "--help"],
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, subprocess.SubprocessError):
        return False


def install_swebench() -> None:
    """Install SWE-bench harness package."""
    print("Installing SWE-bench harness...")
    try:
        # Safe subprocess call - using sys.executable and pip
        # All arguments are hardcoded strings, no user input
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-m", "pip", "install", "swebench"],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout
        )
        if result.returncode != 0:
            stderr = result.stderr.lower()
            # Check for network-related errors
            if any(term in stderr for term in [
                "network", "connection", "timeout", "unreachable",
                "resolve", "dns", "timed out", "connection refused"
            ]):
                print("❌ Network error during SWE-bench installation")
                print("   Check internet connection and try again")
                sys.exit(exit_codes.NETWORK_ERROR)
            else:
                print(f"❌ Failed to install SWE-bench: {result.stderr}")
                print("   Try: pip install swebench")
                sys.exit(exit_codes.GENERAL_ERROR)
        print("✅ SWE-bench harness installed successfully")
    except subprocess.TimeoutExpired:
        print("❌ SWE-bench installation timed out")
        print("   Check internet connection and try again")
        sys.exit(exit_codes.GENERAL_ERROR)
    except Exception as e:
        error_msg = str(e).lower()
        if any(term in error_msg for term in [
            "network", "connection", "timeout", "unreachable",
            "resolve", "dns"
        ]):
            print(f"❌ Network error installing SWE-bench: {e}")
            print("   Check internet connection and try again")
            sys.exit(exit_codes.NETWORK_ERROR)
        else:
            print(f"❌ Error installing SWE-bench: {e}")
            print("   Try: pip install swebench")
            sys.exit(exit_codes.GENERAL_ERROR)


def create_predictions_file(patch: Patch, temp_dir: Path) -> Path:
    """Create SWE-bench predictions file."""
    predictions_file = temp_dir / "predictions.jsonl"

    prediction_data = {
        "instance_id": patch.instance_id,
        "model": "swebench-runner-mvp",
        "prediction": patch.patch
    }

    with predictions_file.open('w', encoding='utf-8') as f:
        json.dump(prediction_data, f)
        f.write('\n')

    return predictions_file


def detect_platform() -> str:
    """Detect platform architecture for Docker image selection."""
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "x86_64"
    elif machine in ("arm64", "aarch64"):
        return "arm64"
    else:
        # Default to x86_64 with warning
        print(f"⚠️  Warning: Unknown architecture {machine}, defaulting to x86_64")
        return "x86_64"


def run_swebench_harness(predictions_file: Path, temp_dir: Path,
                        patch: Patch) -> subprocess.CompletedProcess[str]:
    """Run SWE-bench harness evaluation."""
    run_id = f"mvp-{patch.instance_id}-{uuid.uuid4().hex[:8]}"
    arch = detect_platform()

    # Build harness command
    cmd = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "--predictions_path", str(predictions_file),
        "--max_workers", "1",
        "--run_id", run_id,
        "--timeout", "3600",  # 60 minutes
        "--cache_level", "env"
    ]

    # Add namespace for x86_64 (Epoch AI optimized images)
    if arch == "x86_64":
        cmd.extend(["--namespace", "ghcr.io/epoch-research"])

    print(f"Running SWE-bench harness for {patch.instance_id}...")
    print(f"Command: {' '.join(cmd)}")

    try:
        # Safe subprocess call - cmd built from validated inputs
        # All components are either hardcoded or validated file paths
        result = subprocess.run(  # noqa: S603
            cmd,
            cwd=temp_dir,
            capture_output=True,
            text=True,
            timeout=4200  # 70 minutes (10 minutes buffer over harness timeout)
        )
        return result
    except subprocess.TimeoutExpired:
        print("❌ SWE-bench harness timed out after 70 minutes")
        raise


def parse_harness_results(temp_dir: Path, patch: Patch) -> EvaluationResult:
    """Parse results from SWE-bench harness evaluation."""
    results_dir = temp_dir / "evaluation_results"

    if not results_dir.exists():
        return EvaluationResult(
            instance_id=patch.instance_id,
            passed=False,
            error="No evaluation results directory found"
        )

    # Look for result files
    result_files = list(results_dir.glob("*.json"))
    if not result_files:
        return EvaluationResult(
            instance_id=patch.instance_id,
            passed=False,
            error="No result files found in evaluation_results/"
        )

    # Parse the first result file
    result_file = result_files[0]
    try:
        with result_file.open('r', encoding='utf-8') as f:
            results_data = json.load(f)

        # Check if our instance is in the results
        if patch.instance_id not in results_data:
            return EvaluationResult(
                instance_id=patch.instance_id,
                passed=False,
                error=f"Instance {patch.instance_id} not found in results"
            )

        instance_result = results_data[patch.instance_id]

        # Check resolution status
        resolved = instance_result.get("resolved", False)
        error_msg = instance_result.get("error")

        return EvaluationResult(
            instance_id=patch.instance_id,
            passed=resolved,
            error=error_msg if not resolved else None
        )

    except (json.JSONDecodeError, KeyError) as e:
        return EvaluationResult(
            instance_id=patch.instance_id,
            passed=False,
            error=f"Failed to parse results: {e}"
        )


def run_evaluation(
    patch_source: str, no_input: bool = False, max_patch_size_mb: int = 5
) -> EvaluationResult:
    """Run a single SWE-bench evaluation.

    Args:
        patch_source: Path to JSONL file or directory containing patches
        no_input: If True, fail on prompts instead of waiting for user input
        max_patch_size_mb: Maximum allowed patch size in MB
    """
    # Pre-flight checks
    check_docker_running()
    check_resources()

    # Load patch (includes size validation)
    patch = load_first_patch(patch_source, max_size_mb=max_patch_size_mb)

    # Ensure SWE-bench is installed
    if not check_swebench_installed():
        install_swebench()

    # Create temporary directory for evaluation
    # Also ensure cache directory exists
    get_cache_dir()
    get_logs_dir()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        try:
            # Create predictions file
            predictions_file = create_predictions_file(patch, temp_path)

            # Run harness
            result = run_swebench_harness(predictions_file, temp_path, patch)

            # Check for subprocess errors
            if result.returncode != 0:
                stderr = result.stderr.lower()
                # Check for network-related errors
                if any(term in stderr for term in [
                    "network", "connection", "timeout", "unreachable",
                    "resolve", "dns", "timed out", "connection refused",
                    "failed to pull", "registry", "pull access denied"
                ]):
                    return EvaluationResult(
                        instance_id=patch.instance_id,
                        passed=False,
                        error=f"Network error during harness execution: {result.stderr}"
                    )
                else:
                    return EvaluationResult(
                        instance_id=patch.instance_id,
                        passed=False,
                        error=f"Harness failed with code {result.returncode}: "
                              f"{result.stderr}"
                    )

            # Parse results
            return parse_harness_results(temp_path, patch)

        except subprocess.TimeoutExpired:
            return EvaluationResult(
                instance_id=patch.instance_id,
                passed=False,
                error="Evaluation timed out after 70 minutes"
            )
        except Exception as e:
            return EvaluationResult(
                instance_id=patch.instance_id,
                passed=False,
                error=f"Evaluation failed: {e}"
            )
