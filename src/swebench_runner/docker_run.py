"""Docker execution engine for SWE-bench evaluations."""

import json
import platform
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import docker
from docker.errors import APIError

from .bootstrap import show_memory_warning, show_resource_warning
from .cache import get_cache_dir, get_logs_dir
from .models import EvaluationResult, Patch


def check_docker_running() -> None:
    """Check if Docker daemon is accessible."""
    try:
        client = docker.from_env()
        client.ping()  # type: ignore[no-untyped-call]
    except APIError as e:
        error_msg = str(e).lower()
        if any(term in error_msg for term in [
            "network", "connection", "timeout", "unreachable",
            "resolve", "dns", "timed out", "connection refused"
        ]):
            print("❌ Network error connecting to Docker daemon")
            print("   Check internet connection and Docker network settings")
            sys.exit(3)  # PRD specified exit code for network failure
        else:
            if platform.system() == "Darwin":
                print("⛔ Docker Desktop not running. Start it from Applications "
                      "and wait for whale icon.")
            else:
                print("⛔ Docker daemon unreachable at /var/run/docker.sock")
            print("ℹ️  Start Docker and try again.")
            sys.exit(2)  # PRD specified exit code for Docker missing
    except Exception as e:
        error_msg = str(e).lower()
        if any(term in error_msg for term in [
            "network", "connection", "timeout", "unreachable",
            "resolve", "dns"
        ]):
            print(f"❌ Network error connecting to Docker: {e}")
            print("   Check internet connection and Docker network settings")
            sys.exit(3)  # PRD specified exit code for network failure
        else:
            print(f"❌ Error connecting to Docker: {e}")
            sys.exit(2)  # PRD specified exit code for Docker missing


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
                        patch.validate(max_size_mb=max_size_mb)
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
            patch.validate(max_size_mb=max_size_mb)
            return patch

        else:
            raise ValueError(f"Path {patch_source} is neither a file nor a directory")

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Error: Invalid patch format: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: Patch source not found: {patch_source}")
        sys.exit(1)
    except UnicodeDecodeError:
        print("Error: Patch file contains invalid UTF-8 encoding")
        sys.exit(1)


def check_resources() -> None:
    """Check if system has sufficient resources."""
    try:
        # Check memory (8GB minimum, 16GB recommended)
        try:
            import psutil
            memory_info = psutil.virtual_memory()
            mem_gb = memory_info.available / (1024**3)
            if mem_gb < 8:
                print(f"❌ Critical: Only {mem_gb:.1f}GB RAM available, "
                      "minimum 8GB required")
                print("   Increase system memory or close other applications")
                sys.exit(4)  # PRD specified exit code for resource issues
            elif mem_gb < 16:
                show_memory_warning(mem_gb)
        except ImportError:
            # psutil not available, skip memory check
            pass

        # Check disk space (50GB minimum for lite)
        try:
            disk_usage = shutil.disk_usage(".")
            free_gb = disk_usage.free / (1024**3)
            if free_gb < 50:
                print(f"❌ Critical: Only {free_gb:.1f}GB free disk space, "
                      "minimum 50GB required")
                print("   Free up disk space and try again")
                sys.exit(4)  # PRD specified exit code for resource issues
            elif free_gb < 120:
                show_resource_warning(free_gb)
        except Exception:
            # Can't check disk space, skip
            pass

    except Exception:
        # Non-critical, just skip if we can't check
        pass


def check_swebench_installed() -> bool:
    """Check if SWE-bench harness is installed."""
    try:
        result = subprocess.run(
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
        result = subprocess.run(
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
                sys.exit(3)  # PRD specified exit code for network failure
            else:
                print(f"❌ Failed to install SWE-bench: {result.stderr}")
                print("   Try: pip install swebench")
                sys.exit(1)
        print("✅ SWE-bench harness installed successfully")
    except subprocess.TimeoutExpired:
        print("❌ SWE-bench installation timed out")
        print("   Check internet connection and try again")
        sys.exit(3)  # PRD specified exit code for network failure
    except Exception as e:
        error_msg = str(e).lower()
        if any(term in error_msg for term in [
            "network", "connection", "timeout", "unreachable",
            "resolve", "dns"
        ]):
            print(f"❌ Network error installing SWE-bench: {e}")
            print("   Check internet connection and try again")
            sys.exit(3)  # PRD specified exit code for network failure
        else:
            print(f"❌ Error installing SWE-bench: {e}")
            print("   Try: pip install swebench")
            sys.exit(1)


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
        result = subprocess.run(
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

    # Load patch
    patch = load_first_patch(patch_source, max_size_mb=max_patch_size_mb)

    # Check patch size (environment variable limit)
    patch_bytes = patch.patch.encode('utf-8')
    if len(patch_bytes) > 500 * 1024:  # 500KB limit
        return EvaluationResult(
            instance_id=patch.instance_id,
            passed=False,
            error=f"Patch too large: {len(patch_bytes) / 1024:.1f}KB (max 500KB)"
        )

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
