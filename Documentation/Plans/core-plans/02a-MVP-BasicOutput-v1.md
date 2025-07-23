# MVP-BasicOutput-v1 - Minimal Output Implementation

**Task ID**: MVP-BasicOutput-v1
**Status**: Not Started
**Priority**: Critical Path
**Estimated Effort**: 2-3 hours
**Dependencies**: MVP-DockerRun

## Problem Statement

After running the harness, we need to:
- Display clear pass/fail results
- Save results for later analysis
- Exit with correct codes
- Auto-detect patches file

This is the absolute minimum for a functional tool.

## Research Phase

### Research Checklist
- [ ] How does the harness output results? (exact format and location)
- [ ] What's the structure of the run_evaluation output?
- [ ] Where are result files written?
- [ ] What progress messages does harness emit?
- [ ] How does MVP-DockerRun integrate with harness?
- [x] What exit codes are required? (from PRD)
- [x] How to detect patches files? (from UX Plan)

### Research Tasks (MUST DO BEFORE IMPLEMENTATION)

```markdown
## Quick Validation Script
# Create a test script to run harness and capture ALL outputs

import subprocess
import json
from pathlib import Path

# Run a simple evaluation
cmd = [
    "python", "-m", "swebench.harness.run_evaluation",
    "--predictions_path", "test_patches.jsonl",
    "--instance_ids", "astropy__astropy-12907",  # quick instance
    "--run_id", "test_run"
]

# Capture everything
process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

stdout, stderr = process.communicate()

# Log all outputs
print("=== STDOUT ===")
print(stdout)
print("\n=== STDERR ===")
print(stderr)
print("\n=== EXIT CODE ===")
print(process.returncode)

# Check for output files
print("\n=== OUTPUT FILES ===")
for path in Path(".").rglob("*"):
    if "test_run" in str(path):
        print(path)
        if path.suffix == ".json":
            print(json.dumps(json.loads(path.read_text()), indent=2))
```

### Research Findings (COMPLETED)

1. **Harness Output Structure**:
   - **Result Location**: `evaluation_results/{run_id}.json` in the working directory
   - **Result Schema**:
     ```json
     {
       "instance_id": {
         "resolved": true/false,
         "error": null or "error message"
       }
     }
     ```
   - **Progress Messages** (stdout):
     - "Downloading tomli-2.0.1-py3-none-any.whl.metadata"
     - "Building image sweb.env.py.x86_64..."
     - "Installing build dependencies..."
     - Progress indicators: "0%| | 0/1 [06:42<?, ?it/s]"
   - **Log Files**:
     - Docker build logs: `logs/build_images/`
     - Evaluation logs: `logs/run_evaluation/{run_id}/{model}/{instance_id}/`

2. **MVP-DockerRun Integration** (ALREADY IMPLEMENTED):
   - `docker_run.py` already handles harness execution via subprocess
   - `run_swebench_harness()` executes: `python -m swebench.harness.run_evaluation`
   - `parse_harness_results()` reads from `evaluation_results/*.json`
   - Auto-installs swebench package if missing
   - Returns `EvaluationResult` with passed/error fields

3. **Smart Defaults**:
   - Patches detection already implemented in `load_first_patch()`
   - Searches for JSONL files and patch directories
   - Validates patch size (default 5MB limit)

## Proposed Solution (UPDATED WITH RESEARCH)

### 1. Core Components (2 modules, ~150 lines total)

```python
# output.py - All output handling
from datetime import datetime
from pathlib import Path
from typing import Optional
from .models import EvaluationResult

def detect_patches_file() -> Optional[Path]:
    """Auto-detect patches file in priority order."""
    # Note: This is already implemented in docker_run.load_first_patch()
    # We just need to expose it for CLI usage
    for name in ["predictions.jsonl", "patches.jsonl"]:
        path = Path(name)
        if path.exists() and path.stat().st_size > 0:
            return path
    return None

def display_result(result: EvaluationResult, output_dir: Optional[Path] = None):
    """Display evaluation result with proper formatting."""
    if result.passed:
        print(f"✅ {result.instance_id}: PASSED")
    else:
        print(f"❌ {result.instance_id}: FAILED")
        if result.error:
            print(f"   Error: {result.error}")

    # Save result if output directory specified
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "summary.json"

        result_data = {
            "instance_id": result.instance_id,
            "passed": result.passed,
            "error": result.error,
            "timestamp": datetime.now().isoformat()
        }

        with open(summary_path, 'w') as f:
            json.dump(result_data, f, indent=2)

        print(f"\n📁 Results saved to: {output_dir}")

# cli.py - CLI update (minimal changes needed)
from .docker_run import run_evaluation as docker_run_evaluation
from .output import detect_patches_file, display_result
from .error_utils import classify_error

@cli.command()
@click.option('--instance-id', help='SWE-bench instance ID (auto-detected from patches file if not provided)')
@click.option('--patches-file', type=click.Path(exists=True), help='Path to patches file')
def run(instance_id: Optional[str], patches_file: Optional[str]):
    """Run evaluation on single instance."""
    # Auto-detect patches file if not provided
    if not patches_file:
        detected = detect_patches_file()
        if not detected:
            click.echo("❌ No patches file found (looked for predictions.jsonl and patches.jsonl)", err=True)
            sys.exit(exit_codes.GENERAL_ERROR)
        patches_file = detected
        click.echo(f"💡 Using {patches_file}")

    try:
        # Use existing docker_run.run_evaluation
        result = docker_run_evaluation(patches_file)

        # Display results using our formatter
        display_result(result, Path(f"results/{result.instance_id}"))

        # Exit with correct code
        sys.exit(0 if result.passed else exit_codes.GENERAL_ERROR)

    except Exception as e:
        # Use existing error classification
        exit_code = classify_error(e)
        sys.exit(exit_code)
```

### 2. Exit Codes (Simple)
- 0: Success (test passed)
- 1: Test failed
- 2: Docker error
- 3: Network error
- 4: Disk space error

### 3. Output Format (Minimal)
```
💡 Using predictions.jsonl
Running SWE-bench harness for django__django-11999...
Command: python -m swebench.harness.run_evaluation --predictions_path /tmp/... --max_workers 1 ...

✅ django__django-11999: PASSED

📁 Results saved to: results/django__django-11999
```

For failed evaluations:
```
💡 Using patches.jsonl
Running SWE-bench harness for astropy__astropy-12907...

❌ astropy__astropy-12907: FAILED
   Error: Tests failed during evaluation

📁 Results saved to: results/astropy__astropy-12907
```

### 4. Error Handling (Basic)
```python
try:
    result = run_evaluation(instance_id, patches_file)
except DockerException:
    click.echo("❌ Docker error: Is Docker running?", err=True)
    sys.exit(2)
except Exception as e:
    click.echo(f"❌ Error: {e}", err=True)
    sys.exit(1)
```

## Implementation Checklist

- [ ] Create output.py module
  - [ ] Add detect_patches_file() function
  - [ ] Add display_result() function
  - [ ] Import EvaluationResult from models.py
- [ ] Update cli.py
  - [ ] Import output functions
  - [ ] Import error_utils.classify_error
  - [ ] Update run command to use new output formatting
  - [ ] Remove instance-id requirement (auto-detect from patch)
- [ ] Add basic tests
  - [ ] Test patch detection with various file names
  - [ ] Test result display (both pass/fail)
  - [ ] Test JSON output saving
  - [ ] Test exit codes
- [ ] Manual testing
  - [ ] Test with real instance
  - [ ] Test auto-detection
  - [ ] Test error scenarios

## Test Plan

### Unit Tests
```python
def test_detect_patches_file(tmp_path):
    """Test auto-detection."""
    # Create predictions.jsonl
    (tmp_path / "predictions.jsonl").touch()

    # Should find it
    with chdir(tmp_path):
        assert detect_patches_file() == Path("predictions.jsonl")

def test_display_results(capsys):
    """Test result display."""
    results = [{"instance_id": "test-1", "resolved": True}]
    display_results(results, Path("output"))

    captured = capsys.readouterr()
    assert "test-1: ✅ PASSED" in captured.out
```

### Integration Test
```python
def test_cli_run_success(runner, mock_harness):
    """Test successful run."""
    mock_harness.return_value = {"resolved": True}

    result = runner.invoke(cli, ['run', '--instance-id', 'test-1'])
    assert result.exit_code == 0
    assert "✅ PASSED" in result.output
```

## Success Criteria

1. ✅ Shows clear pass/fail
2. ✅ Auto-detects patches file
3. ✅ Saves results to known location
4. ✅ Exits with correct code
5. ✅ <200 lines of code

## Dependencies

- Click (already installed)
- pathlib (stdlib)
- json (stdlib)
- sys (stdlib)

## Questions/Uncertainties

### ✅ Resolved (From Research)
1. **Harness Integration**: MVP-DockerRun handles everything via `run_evaluation()`
2. **Output Structure**: Results in `evaluation_results/{run_id}.json` with `resolved` boolean
3. **Progress Tracking**: Harness prints to stdout, docker_run already shows it
4. **Error Detection**: `error_utils.classify_error()` already maps exceptions to exit codes

### Non-blocking
- Whether to show execution time → Nice to have, not critical
- Whether to parse harness progress messages → v2 feature
- HTML report generation → Definitely v2

## Notes

This is the absolute minimum viable output. Most functionality already exists:
1. `docker_run.run_evaluation()` handles harness execution
2. We just need to format the output nicely
3. Save a simple JSON summary
4. Exit with proper codes

Key insights from research:
- MVP-DockerRun already does the heavy lifting
- We just need a thin output formatting layer
- Error handling is already implemented in error_utils
- Harness progress shows automatically via subprocess

Total implementation: ~100 lines in output.py + minor CLI updates
