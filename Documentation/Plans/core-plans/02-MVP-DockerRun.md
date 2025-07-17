# Work Plan: MVP-DockerRun - Execute Single Instance in Docker

**Task ID**: MVP-DockerRun  
**Status**: Not Started

## Problem Statement

We need to execute a single SWE-bench instance using the official SWE-bench harness with Epoch AI's optimized Docker images. This is the core functionality - actually running the evaluation. Without this, we have no product. The implementation must handle harness execution, predictions file creation, and result extraction.

## Proposed Solution

Create a minimal SWE-bench harness execution pipeline that:
1. Installs the official SWE-bench package
2. Creates a predictions file in SWE-bench format
3. Runs evaluation using `python -m swebench.harness.run_evaluation`
4. Parses results from the evaluation_results directory
5. Handles basic errors (Docker not running, harness failures)

### Technical Approach
- **UPDATED BASED ON VALIDATION**: Use official SWE-bench harness with Epoch AI images
- Install `swebench` package via pip and use `python -m swebench.harness.run_evaluation`
- Use Epoch AI's optimized images via `--namespace ghcr.io/epoch-research`
- Create predictions file in JSONL format with instance_id, model, and prediction fields
- Parse JSON output from `evaluation_results/` directory
- Use subprocess to execute harness with proper timeout handling
- Platform-specific Docker detection and ARM64 support

### Expected Implementation
```python
# docker_run.py - UPDATED TO USE OFFICIAL SWE-BENCH HARNESS
import json
import sys
import platform
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Optional
import docker
from docker.errors import APIError
from .models import Patch, EvaluationResult

def check_docker_running() -> None:
    """Check if Docker daemon is accessible."""
    try:
        client.ping()
    except APIError:
        if platform.system() == "Darwin":
            print("⛔ Docker Desktop not running. Start it from Applications and wait for whale icon.")
        else:
            print("⛔ Docker daemon unreachable at /var/run/docker.sock")
        print("ℹ️  Start Docker and try again.")
        sys.exit(2)  # PRD specified exit code for Docker missing

def load_first_patch(patch_file: str) -> Patch:
    \"\"\"Load the first patch from a JSONL file.\"\"\"
    try:
        with open(patch_file, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    patch = Patch(
                        instance_id=data['instance_id'],
                        patch=data['patch']
                    )
                    patch.validate()
                    return patch
        raise ValueError("No patches found in file")
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error: Invalid patch file format: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: Patch file not found: {patch_file}")
        sys.exit(1)

def check_resources(client: docker.DockerClient) -> None:
    \"\"\"Check if Docker has sufficient resources.\"\"\"
    try:
        info = client.info()
        
        # Check memory (16GB+ recommended)
        mem_total = info.get('MemTotal', 0)
        if mem_total > 0:
            mem_gb = mem_total / (1024**3)
            if mem_gb < 16:
                print(f"⚠️  Warning: {mem_gb:.1f}GB RAM available, 16GB+ recommended")
                print("   Large evaluations may fail or run slowly")
            elif mem_gb < 8:
                print(f"❌ Critical: Only {mem_gb:.1f}GB RAM available, minimum 8GB required")
                print("   Increase Docker memory limit in Docker Desktop settings")
        
        # Check disk space (120GB needed for image cache)
        try:
            disk_usage = shutil.disk_usage("/var/lib/docker")
            free_gb = disk_usage.free / (1024**3)
            if free_gb < 120:
                print(f"⚠️  Warning: {free_gb:.1f}GB free disk space, 120GB+ recommended")
                print("   SWE-bench images require substantial disk space")
        except:
            # Docker might be using different location, check current directory
            try:
                disk_usage = shutil.disk_usage(".")
                free_gb = disk_usage.free / (1024**3)
                if free_gb < 50:
                    print(f"⚠️  Warning: {free_gb:.1f}GB free disk space")
                    print("   Ensure sufficient space for Docker images")
            except:
                pass
                
    except:
        # Non-critical, just skip if we can't check
        pass

def ensure_image_exists(client: docker.DockerClient, instance_id: str) -> str:
    \"\"\"Ensure Docker image exists, pull if needed. Returns image name.\"\"\"
    # Try Epoch AI's optimized images first
    epoch_image = f"ghcr.io/epoch-research/swe-bench.eval.x86_64.{instance_id}"
    
    try:
        client.images.get(epoch_image)
        print(f"Using existing image: {epoch_image}")
        return epoch_image
    except NotFound:
        print(f"Pulling image {epoch_image} (this may take a few minutes)...")
        try:
            # Epoch AI images are public and don't require authentication
            client.images.pull(epoch_image)
            print("Image pulled successfully")
            return epoch_image
        except APIError as e:
            print(f"Error: Failed to pull Epoch AI image: {e}")
            if "not found" in str(e).lower():
                print("💡 This instance is not available in Epoch AI's registry")
                print("   Falling back to official SWE-bench harness...")
                
                # TODO: Implement fallback to official SWE-bench harness
                # For MVP, we'll error out with helpful message
                print("❌ Fallback to official harness not implemented in MVP")
                print("   Available instances: https://github.com/epoch-research/SWE-bench")
                print("   You may need to use the official SWE-bench harness for this instance")
                sys.exit(1)
            else:
                print(f"❌ Network or authentication error: {e}")
                print("   Check your internet connection and try again")
                sys.exit(1)

def parse_container_output(instance_id: str, output: str, 
                          wait_result: dict) -> EvaluationResult:
    \"\"\"Parse container output to determine success/failure.\"\"\"
    exit_code = wait_result.get('StatusCode', 1)
    
    # Try to parse JSON output (multiple possible formats)
    try:
        # Find JSON in output (might have other text)
        for line in output.strip().split('\n'):
            line = line.strip()
            if line.startswith('{'):
                try:
                    result_data = json.loads(line)
                    
                    # Check for resolution_status format (verified format)
                    if 'resolution_status' in result_data:
                        passed = result_data['resolution_status'] == 'PASSED' and exit_code == 0
                        return EvaluationResult(
                            instance_id=instance_id,
                            passed=passed,
                            error=None if passed else f"Test failed: {result_data.get('test_output', 'No output')}"
                        )
                    
                    # Fallback: check for 'passed' boolean format
                    elif 'passed' in result_data:
                        passed = result_data.get('passed', False) and exit_code == 0
                        return EvaluationResult(
                            instance_id=instance_id,
                            passed=passed,
                            error=None if passed else "Test failed"
                        )
                        
                except json.JSONDecodeError:
                    continue
                    
    except Exception:
        pass
    
    # Fallback: non-zero exit = failure
    return EvaluationResult(
        instance_id=instance_id,
        passed=(exit_code == 0),
        error=f"Exit code: {exit_code}, could not parse output"
    )

def run_evaluation(patch_file: str) -> EvaluationResult:
    """Run a single SWE-bench evaluation."""
    client = docker.from_env()
    check_docker_running(client)
    check_resources(client)  # Warn if insufficient resources
    
    # Load first patch from JSONL
    patch = load_first_patch(patch_file)
    
    # Check patch size for env var limit (accounting for base64 encoding)
    patch_bytes = patch.patch.encode('utf-8')
    # Base64 encoding increases size by ~33%, so check against 375KB to stay under 500KB limit
    base64_size_estimate = len(patch_bytes) * 4 // 3  # More accurate base64 size calculation
    if base64_size_estimate > 500 * 1024:  # 500KB threshold for environment variables
        print("⚠️  Large patch detected, would use volume mount (not implemented in MVP)")
        print(f"   Patch size: {len(patch_bytes) / 1024:.1f}KB, Base64 size: {base64_size_estimate / 1024:.1f}KB")
        # For MVP, we'll error out
        print("Error: Patch too large for MVP (>500KB after base64 encoding)")
        sys.exit(1)
    
    print(f"Running evaluation for {patch.instance_id}...")
    
    try:
        # Pull image if needed (returns actual image name)
        image = ensure_image_exists(client, patch.instance_id)
        
        # Encode patch as base64 for environment variable
        patch_b64 = base64.b64encode(patch.patch.encode('utf-8')).decode('ascii')
        
        # Run container with SWE-bench Docker interface
        container = client.containers.run(
            image=image,
            environment={
                "SWE_INSTANCE_ID": patch.instance_id,
                "SWE_PATCH": patch_b64
            },
            mem_limit="8g",  # PRD requirement
            detach=True,
            remove=False,  # Keep for debugging if needed
            working_dir="/testbed"  # Standard SWE-bench working directory
        )
        
        # Wait for completion with timeout (60 minutes - some evaluations take longer)
        try:
            result = container.wait(timeout=3600)  # 60 minutes
        except docker.errors.APIError as e:
            # Timeout or other API error
            print(f"Container timeout or error: {e}")
            try:
                container.kill()
            except:
                pass  # Container might already be dead
            return EvaluationResult(
                instance_id=patch.instance_id,
                passed=False,
                error="Evaluation timed out after 60 minutes"
            )
        
        # Get output from multiple sources
        logs = container.logs(stdout=True, stderr=True)
        logs_output = logs.decode('utf-8', errors='replace')
        
        # Try to extract result files from container (SWE-bench may generate files)
        file_output = None
        try:
            # Common locations for SWE-bench results
            for result_path in ['/testbed/result.json', '/testbed/evaluation_result.json', '/result.json']:
                try:
                    bits, _ = container.get_archive(result_path)
                    # Extract and read the file content
                    tar_stream = io.BytesIO(b''.join(bits))
                    with tarfile.open(fileobj=tar_stream, mode='r') as tar:
                        for member in tar.getmembers():
                            if member.isfile():
                                file_content = tar.extractfile(member).read()
                                file_output = file_content.decode('utf-8', errors='replace')
                                break
                    if file_output:
                        break
                except:
                    continue
        except:
            pass
        
        # Use file output if available, otherwise use logs
        output = file_output or logs_output
        
        # Parse result
        return parse_container_output(patch.instance_id, output, result)
        
    except ContainerError as e:
        return EvaluationResult(
            instance_id=patch.instance_id,
            passed=False,
            error=f"Container failed: {e}"
        )
    finally:
        # Cleanup
        if 'container' in locals():
            container.remove(force=True)
```

## Automated Test Plan

1. **Unit Tests** (`tests/test_docker_run.py`):
   ```python
   import pytest
   from unittest.mock import Mock, patch
   from swebench_runner.docker_run import run_evaluation, check_docker_running
   
   @pytest.fixture
   def mock_docker_client():
       """Mock Docker client for testing."""
       client = Mock()
       client.ping.return_value = True
       client.images.pull.return_value = Mock()
       
       # Mock container
       container = Mock()
       container.wait.return_value = {"StatusCode": 0}
       container.logs.return_value = b'{"passed": true}'
       client.containers.run.return_value = container
       
       return client
   
   def test_docker_not_running():
       """Test handling when Docker is not running."""
       client = Mock()
       client.ping.side_effect = docker.errors.APIError("Cannot connect")
       
       with pytest.raises(SystemExit) as exc_info:
           check_docker_running(client)
       assert exc_info.value.code == 2  # Docker missing exit code
   ```
   - Test Docker client creation
   - Test platform-specific Docker detection
   - Test container configuration
   - Test output parsing
   - Test patch size validation
   - Mock container execution

2. **Integration Tests** (with real Docker):
   - Test with minimal test instance
   - Test timeout handling
   - Test error scenarios
   - Test image pull progress

## Components Involved

- `src/swebench_runner/docker_run.py` - Docker execution logic
- `src/swebench_runner/models.py` - Data classes for patches/results
- `tests/test_docker_run.py` - Docker execution tests
- `tests/fixtures/` - Test patches and expected results

### Data Models
```python
# models.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class Patch:
    """Represents a single patch to evaluate."""
    instance_id: str
    patch: str
    
    def validate(self) -> None:
        """Basic validation for MVP."""
        if not self.instance_id:
            raise ValueError("instance_id cannot be empty")
        if not self.patch:
            raise ValueError("patch cannot be empty")

@dataclass
class EvaluationResult:
    """Result of a single evaluation."""
    instance_id: str
    passed: bool
    error: Optional[str] = None
```

## Dependencies

- **External**: 
  - docker>=6.1.0 (Docker SDK for Python - for daemon checking)
  - swebench (official SWE-bench harness - auto-installed)
  - Docker daemon running (20.10.0+ recommended)
  - Epoch AI optimized images (ghcr.io/epoch-research)
  - pytest>=7.0 (for testing)
  - pytest-timeout>=2.0 (for test timeouts)
- **Internal**: 
  - 01-MVP-CLI (need CLI structure)
  - models.py (data structures)
- **Knowledge**: 
  - [SWE-bench documentation](https://www.swebench.com/SWE-bench/guides/evaluation/)
  - [SWE-bench harness usage](https://github.com/swe-bench/SWE-bench)
  - [Epoch AI optimized images](https://github.com/epoch-research/SWE-bench)

## Implementation Checklist

### Phase 1: Research & Validation (CRITICAL - Lessons from CI Implementation)
- [ ] **Pre-implementation Research**
  - [ ] Research Docker SDK cross-platform behavior (macOS vs Linux daemon paths)
  - [ ] Validate SWE-bench harness installation requirements and Python version compatibility
  - [ ] Test subprocess execution patterns on target platforms
  - [ ] Verify Epoch AI image availability and sizing (ARM64 vs x86_64)
  - [ ] Document platform-specific Docker daemon detection methods
  - [ ] Research environment variable limits for different platforms
- [ ] **Dependency Compatibility Research**
  - [ ] Check docker>=6.1.0 compatibility with Python 3.8-3.12
  - [ ] Verify swebench package installation requirements
  - [ ] Test subprocess timeout handling across platforms
  - [ ] Validate pathlib.Path vs os.path for cross-platform compatibility
- [ ] **Risk Assessment**
  - [ ] Identify high-risk integration points (Docker daemon, subprocess, image pulling)
  - [ ] Plan comprehensive mocking strategy for external dependencies
  - [ ] Define error categories and exit codes before implementation
  - [ ] Create test matrix for platform-specific scenarios

### Phase 2: Implementation
- [ ] **Set up prerequisites**
  - [ ] Add docker>=6.1.0 to pyproject.toml dependencies (for daemon checking)
  - [ ] Create docker_run.py module
  - [ ] Implement platform-specific Docker daemon availability checking
  - [ ] Add cross-platform Docker daemon detection with proper error messages
  - [ ] Implement check_docker_running() with exit code 2
- [ ] **Create data models first**
  - [ ] Create models.py
  - [ ] Define Patch dataclass with validation
  - [ ] Define EvaluationResult dataclass
  - [ ] Add basic validation methods with clear error messages
- [ ] **Implement JSONL loading**
  - [ ] Create load_first_patch() function with robust error handling
  - [ ] Parse JSONL file with proper encoding handling
  - [ ] Handle file not found with user-friendly messages
  - [ ] Handle invalid JSON with line number information
  - [ ] Extract first patch only for MVP
- [ ] **Implement SWE-bench harness integration**
  - [ ] Create check_swebench_installed() function
  - [ ] Implement install_swebench() function with version compatibility checks
  - [ ] Create predictions file in SWE-bench format using pathlib.Path
  - [ ] Implement robust platform detection (ARM64 vs x86_64)
  - [ ] Build subprocess command for harness execution with cross-platform considerations
  - [ ] Run harness with proper timeout handling and resource monitoring
  - [ ] Parse results from evaluation_results directory with error recovery
- [ ] **Add comprehensive error handling**
  - [ ] Handle Docker daemon not running (exit code 2) with platform-specific guidance
  - [ ] Handle swebench installation failures with actionable error messages
  - [ ] Handle harness execution failures with diagnostic information
  - [ ] Handle harness timeout with cleanup procedures
  - [ ] Handle malformed results with fallback parsing
  - [ ] Handle resource constraints (memory, disk space) with clear warnings

### Phase 3: Testing & Validation
- [ ] **Write comprehensive tests**
  - [ ] Create mock subprocess fixture for cross-platform testing
  - [ ] Test Docker availability checking on multiple platforms
  - [ ] Test swebench installation with version compatibility scenarios
  - [ ] Test successful evaluation with mocked external dependencies
  - [ ] Test timeout scenarios without waiting for real timeouts
  - [ ] Test platform detection with mocked platform.system()
  - [ ] Test results parsing with various output formats
  - [ ] Test error scenarios with proper exit codes
- [ ] **Cross-platform validation**
  - [ ] Test on macOS (Docker Desktop) and Linux (docker.sock)
  - [ ] Validate subprocess behavior differences
  - [ ] Test pathlib.Path vs os.path compatibility
  - [ ] Verify environment variable handling across platforms
- [ ] **Integration with CLI**
  - [ ] Update cli.py run command to call run_evaluation()
  - [ ] Pass patches file path using pathlib.Path
  - [ ] Display progress messages with platform-appropriate formatting
  - [ ] Show results (pass/fail) with proper exit codes

### Phase 4: Pre-commit Validation
- [ ] **Local testing checklist**
  - [ ] Run exact CI linting commands locally (ruff check, mypy)
  - [ ] Test with sample patches on local Docker setup
  - [ ] Verify all error scenarios produce expected exit codes
  - [ ] Test timeout handling without waiting for real timeouts
  - [ ] Validate cross-platform path handling
- [ ] **Integration testing**
  - [ ] Test with real Docker daemon (if available)
  - [ ] Test with mock Docker daemon for CI scenarios
  - [ ] Verify SWE-bench harness integration (if swebench available)
  - [ ] Test resource constraint scenarios

## Verification Steps

1. **Manual Testing**:
   ```bash
   # Ensure Docker is running
   docker version
   
   # Test with sample patch
   echo '{"instance_id": "astropy__astropy-12907", "patch": "diff --git..."}' > test.jsonl
   swebench run --patches test.jsonl
   
   # Should show swebench installation, harness execution, and result
   ```

2. **SWE-bench Harness Verification**:
   ```bash
   # Check swebench installation
   python -m swebench.harness.run_evaluation --help
   
   # Verify Epoch AI images are used
   docker images | grep epoch-research
   
   # Check evaluation results
   ls evaluation_results/
   ```

3. **Success Criteria**:
   - [ ] Successfully connects to Docker
   - [ ] Automatically installs swebench if needed
   - [ ] Creates predictions file in correct format
   - [ ] Executes harness with proper arguments
   - [ ] Uses Epoch AI images (x86_64) or builds locally (ARM64)
   - [ ] Parses results from evaluation_results directory
   - [ ] Returns pass/fail result
   - [ ] Handles common errors gracefully

## Decision Authority

**Can decide independently**:
- Container configuration details
- Timeout duration for MVP (30 min)
- Output parsing logic
- Error message formatting

**Need user input**:
- Exact Docker image to use
- Whether to pull image automatically
- Result format structure

## Questions/Uncertainties

**✅ Resolved** (Validation Complete):
- ✅ Container interface: Use official SWE-bench harness instead of direct container execution
- ✅ Environment variables: Not needed - harness handles container communication
- ✅ Output format: Parse results from `evaluation_results/` directory
- ✅ Image availability: Epoch AI provides optimized images, harness handles fallback
- ✅ Platform support: ARM64 builds locally, x86_64 uses Epoch AI images

**✅ Non-blocking** (Decisions Made):
- ✅ Progress indication: Harness provides built-in progress output
- ✅ Memory/CPU limits: Harness handles resource management automatically
- ✅ Timeout handling: Use harness --timeout flag (60 minutes default)
- ✅ Installation: Auto-install swebench package if not present

## Acceptable Tradeoffs

- **Single instance only** - no parallel execution (use --max_workers 1)
- **Fixed timeout** - 60 minutes, not configurable in MVP
- **Minimal output** - just pass/fail, no detailed logs saved
- **Harness dependency** - relies on official SWE-bench harness installation
- **Auto-installation** - installs swebench package automatically
- **First patch only** - from JSONL file, ignore others
- **Basic validation** - just check patch exists and format
- **No retry logic** - fail immediately on errors
- **Temporary files** - creates temp directory for each evaluation

## Critical Lessons from CI Implementation

### Cross-Platform Compatibility Issues
**Problem**: Platform-specific assumptions cause failures in CI/CD environments
**Solutions**:
- Use `pathlib.Path` instead of string paths for cross-platform compatibility
- Avoid platform-specific shell commands (e.g., `stat -c%s` on Linux only)
- Use Python built-ins for cross-platform operations (file size, path manipulation)
- Test subprocess behavior on both macOS and Linux

### Dependency Version Conflicts
**Problem**: Security fixes may require newer versions than supported Python versions
**Solutions**:
- Research compatibility matrices before adding dependencies
- Use conditional requirements when necessary
- Validate all dependency versions against target Python versions (3.8-3.12)
- Consider build vs runtime dependency differences

### Error Handling & User Experience
**Problem**: Cryptic error messages lead to poor user experience
**Solutions**:
- Provide platform-specific guidance ("Start Docker Desktop" vs "Check docker.sock")
- Include actionable next steps in all error messages
- Use appropriate exit codes for different error categories
- Warn about resource constraints (memory, disk space) before failures

### Testing & Validation
**Problem**: Local testing doesn't match CI environment behavior
**Solutions**:
- Run exact CI linting commands locally before committing
- Mock external dependencies comprehensively
- Test platform-specific code paths explicitly
- Validate error scenarios produce expected exit codes

### Pre-commit Validation Checklist
Based on CI implementation experience:
- [ ] Run `ruff check src/swebench_runner tests` locally
- [ ] Run `mypy src/swebench_runner` locally
- [ ] Test on multiple platforms (macOS, Linux if available)
- [ ] Verify all error paths produce appropriate exit codes
- [ ] Test timeout scenarios without waiting for real timeouts
- [ ] Validate cross-platform path and subprocess handling

## Notes

This is the core execution engine. **UPDATED** to use the official SWE-bench harness for MVP:
- One instance at a time (--max_workers 1)
- Uses official SWE-bench harness via subprocess
- Leverages Epoch AI optimized images
- Basic pass/fail output
- Auto-installs dependencies

The architecture allows for enhancement in later phases (parallel execution, detailed logging, etc.) but focuses on getting a working execution first using the proven SWE-bench harness.

### SWE-bench Harness Interface (VERIFIED)

**✅ CRITICAL UPDATE**: We now use the official SWE-bench harness instead of direct container execution.

**Harness Command Format**:
```bash
python -m swebench.harness.run_evaluation \
  --predictions_path predictions.jsonl \
  --max_workers 1 \
  --run_id mvp-{instance_id}-{uuid} \
  --timeout 3600 \
  --cache_level env \
  --namespace ghcr.io/epoch-research  # x86_64 only
```

**Input Format** (predictions.jsonl):
```json
{
  "instance_id": "astropy__astropy-12907",
  "model": "swebench-runner-mvp", 
  "prediction": "diff --git a/file.py b/file.py\n..."
}
```

**Output Format** (evaluation_results/{run_id}.json):
```json
{
  "astropy__astropy-12907": {
    "resolved": true,
    "error": null
  }
}
```

**Platform Support**:
- **x86_64**: Uses Epoch AI optimized images (`--namespace ghcr.io/epoch-research`)
- **ARM64**: Builds images locally (`--namespace ""`)

**Success Detection**:
- `"resolved": true` = success
- `"resolved": false` or missing = failure
- Parse from `evaluation_results/` directory

### MVP Scope Clarification

**What we DO in MVP**:
- Run one patch from a JSONL file
- Basic validation (file exists, can parse, size check)
- Show progress during image pull and execution
- Return simple pass/fail result
- Handle basic Docker errors with proper exit codes

**What we DON'T do in MVP**:
- No parallel execution
- No detailed logging (just stdout)
- No patch validation beyond size
- No retry on failures
- No volume mounting for large patches
- No configuration options

### Patch Size Handling

**Environment Variable Approach** (MVP default):
- Patches up to 500KB passed via PATCH environment variable
- Check size before passing: `len(patch.patch.encode('utf-8'))`
- Error if >500KB (PRD mentions 1MB Docker limit, we use 500KB for safety)

**Future Volume Mount Approach** (not in MVP):
```python
# For patches >500KB, would use:
volumes = {
    '/tmp/patch.diff': {
        'bind': '/patch.diff',
        'mode': 'ro'
    }
}
# Then container reads from /patch.diff instead of env var
```

For MVP, we simply error on large patches with a clear message.

### Documentation References

- [docker-py Quick Start](https://docker-py.readthedocs.io/en/stable/client.html)
- [Container run() API](https://docker-py.readthedocs.io/en/stable/containers.html#docker.models.containers.ContainerCollection.run)
- [Docker SDK Error Types](https://docker-py.readthedocs.io/en/stable/api.html#errors)
- [Container wait() timeout](https://docker-py.readthedocs.io/en/stable/containers.html#docker.models.containers.Container.wait)
- [Platform Detection in Python](https://docs.python.org/3/library/platform.html)

### Research Findings (COMPLETED)

1. **SWE-bench Docker Architecture**:
   - Uses multi-layered approach: base → environment → instance images
   - No single "runner" image - each instance has its own image
   - Official harness at `swebench.harness.run_evaluation` module
   - Alternative: Epoch AI's optimized images at `ghcr.io/epoch-research/swe-bench.eval.<arch>.<instance_id>`

2. **Container Interface**:
   - **Input**: Base64-encoded environment variable for patch data
   - **Alternative**: Volume mounting for large patches (>500KB) via `SWEBENCH_DOCKER_FORK_DIR`
   - **Execution**: Runs `/eval.sh` script in container with `DOCKER_WORKDIR` and `DOCKER_USER`
   - **Output**: JSON report with test runtime, output, and resolution status

3. **System Requirements**:
   - 120GB free disk space (for image cache)
   - 16GB+ RAM recommended
   - 8+ CPU cores
   - Workers: `min(0.75 * cpu_count(), 24)`

4. **Timeout Handling**:
   - Configurable timeout per instance
   - Docker containers are forcibly killed on timeout
   - Cleanup required for failed containers

### CLI Integration

The CLI will call this module like:
```python
# In cli.py
from .docker_run import run_evaluation

@cli.command()
@click.option('--patches', required=True, type=click.Path(exists=True))
def run(patches):
    """Run SWE-bench evaluation."""
    try:
        result = run_evaluation(patches)
        if result.passed:
            click.echo(f"✓ {result.instance_id}: PASSED")
        else:
            click.echo(f"✗ {result.instance_id}: FAILED")
            if result.error:
                click.echo(f"  Error: {result.error}")
        sys.exit(0 if result.passed else 1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
```