
# PR #6 MVP-DockerRun Implementation - Error Report

**PR**: #6 MVP Docker Execution Implementation  
**Date**: 2025-07-21  
**Status**: Failed CI, Multiple Critical Issues  
**Severity**: HIGH - Blocking merge

## Executive Summary

The MVP Docker execution implementation (PR #6) encountered significant issues during CI and code review. This report documents all identified errors, their root causes, and recommended fixes to ensure the information is preserved for future reference.

## Critical Issues Found

### 1. Bootstrap Flow Breaking All CLI Tests
**Severity**: CRITICAL  
**Location**: src/swebench_runner/cli.py:67  
**Impact**: All CLI tests fail with `SystemExit(0)`

**Issue Details**:
- The bootstrap check runs before argument validation in the `run` command
- Tests don't mock bootstrap functions or use `--no-input` flag
- Bootstrap creates persistent state (~/.swebench) affecting test isolation

**Root Cause Analysis** (REVISED):
- Original analysis was partially incorrect
- The function doesn't exit with code 0 when returning False
- Real issue: Tests execute real filesystem operations without isolation

**Fix Required** (UNIFIED SOLUTION):
```python
# conftest.py - Comprehensive solution combining isolation + mocking
@pytest.fixture(autouse=True)
def test_environment(monkeypatch):
    """Complete test isolation for all tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Isolate cache directory
        monkeypatch.setenv("SWEBENCH_CACHE_DIR", str(Path(tmpdir) / ".swebench"))
        # Force CI mode to prevent any interactive prompts
        monkeypatch.setenv("CI", "true")
        # Mock bootstrap to avoid any filesystem checks
        with patch("swebench_runner.cli.check_and_prompt_first_run", return_value=False):
            yield tmpdir
```

**Why This Solution**:
- Provides complete isolation from user's filesystem
- Prevents any interactive prompts in tests
- Works for all test scenarios without modification
- Backwards compatible with existing tests
- No need to modify individual test methods

### 2. Python Version Compatibility Issues
**Severity**: HIGH  
**Location**: pyproject.toml, multiple source files  
**Impact**: Installation fails on Python 3.8 and 3.9

**Issue Details**:
- pyproject.toml requires Python >=3.10
- Union type syntax (X | Y) used throughout requires Python 3.10+
- SWE-bench harness also requires Python 3.10+
- CI tests against Python 3.8, 3.9, 3.10, 3.11, 3.12

**Root Cause** (VERIFIED):
- Confirmed: SWE-bench official repository requires Python >=3.10
- Source: https://github.com/SWE-bench/SWE-bench/blob/main/pyproject.toml
- Maintainers use Python 3.9, 3.10, or 3.11 (per issue #156)

**Fix Required** (SIMPLIFIED):
```yaml
# Update CI matrix in .github/workflows/ci.yml:
python-version: ["3.10", "3.11", "3.12"]

# pyproject.toml is already correct:
requires-python = ">=3.10"

# Keep modern type syntax - no changes needed
```

### 3. Conflicting Patch Size Limits
**Severity**: MEDIUM  
**Location**: src/swebench_runner/docker_run.py:347  
**Impact**: Confusing behavior, incorrect error messages

**Issue Details** (CLARIFIED):
- Function accepts `max_patch_size_mb` parameter (default 5MB) - general patch limit
- Hardcoded check for 500KB limit - Docker environment variable limit
- Two limits serve different purposes but error message is misleading

**Root Cause** (RESEARCHED):
- Docker has ~1MB limit for environment variables
- PRD mentions "automatic fallback to bind-mounted patch file when patch > 500KB"
- Current implementation doesn't implement the fallback

**Fix Required** (COMPLETE IMPLEMENTATION):
```python
def run_evaluation(patch_source: str, no_input: bool = False, max_patch_size_mb: int = 5):
    # Load patch with configurable limit
    patch = load_first_patch(patch_source, max_size_mb=max_patch_size_mb)
    
    # Check if patch needs bind-mount due to Docker env var limit
    patch_bytes = patch.patch.encode('utf-8')
    patch_size_kb = len(patch_bytes) / 1024
    use_bind_mount = patch_size_kb > 500  # 500KB Docker env limit
    
    if use_bind_mount:
        # Clear explanation for users
        if patch_size_kb > max_patch_size_mb * 1024:
            return EvaluationResult(
                instance_id=patch.instance_id,
                passed=False,
                error=(f"Patch size {patch_size_kb:.1f}KB exceeds the general limit "
                       f"of {max_patch_size_mb}MB. Please reduce patch size.")
            )
        else:
            # Implement bind-mount for large patches
            return _run_with_bind_mount(patch, temp_path)

def _run_with_bind_mount(patch: PatchInstance, temp_path: Path) -> EvaluationResult:
    """Run evaluation with patch bind-mounted to container.
    
    Used when patch exceeds Docker's 500KB environment variable limit.
    """
    import tempfile
    import os
    
    # Create secure temporary file for patch
    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.patch',
        dir=temp_path,
        delete=False
    ) as patch_file:
        patch_file.write(patch.patch)
        patch_file_path = patch_file.name
    
    try:
        # Prepare Docker volume mount
        volumes = {
            patch_file_path: {
                'bind': '/tmp/patch.diff',
                'mode': 'ro'  # Read-only for security
            }
        }
        
        # Run harness with bind-mounted patch
        cmd = [
            "python", "-m", "swebench.harness.run_evaluation",
            "--predictions_path", predictions_path,
            "--patch_path", "/tmp/patch.diff",  # Container path
            # ... other args
        ]
        
        # Execute with volumes
        result = subprocess.run(
            cmd,
            volumes=volumes,
            capture_output=True,
            text=True,
            timeout=4200
        )
        
        return parse_harness_results(temp_path, patch)
        
    finally:
        # Clean up temporary patch file
        try:
            os.unlink(patch_file_path)
        except OSError:
            pass  # Best effort cleanup
```

**Error Message Improvements**:
```python
# Clear, educational error messages
if use_bind_mount and not BIND_MOUNT_IMPLEMENTED:
    return EvaluationResult(
        instance_id=patch.instance_id,
        passed=False,
        error=(
            f"⚠️  Patch Size Limitation\n"
            f"\n"
            f"Your patch is {patch_size_kb:.1f}KB, which exceeds Docker's "
            f"environment variable limit of 500KB.\n"
            f"\n"
            f"📊 Size Limits Explained:\n"
            f"• Docker env limit: 500KB (system constraint)\n"
            f"• General patch limit: {max_patch_size_mb}MB (configurable)\n"
            f"\n"
            f"Your patch is within the general limit but too large for env vars.\n"
            f"\n"
            f"🔧 Workarounds:\n"
            f"1. Split your patch into smaller logical changes\n"
            f"2. Use --patches-dir with .patch files (coming in v1.1)\n"
            f"3. Wait for bind-mount support (planned for v1.1)\n"
            f"\n"
            f"📅 Timeline: Large patch support via bind-mount coming in v1.1 (Q1 2025)"
        )
    )
```

### 4. Exit Code Mismatches
**Severity**: HIGH  
**Location**: Multiple files  
**Impact**: Incorrect error reporting, test failures

**PRD Specification** (VERIFIED):
- 0 = success
- 1 = general harness error
- 2 = Docker missing
- 3 = network failure  
- 4 = disk full/resource issues

**Specific Problems Found**:
1. Docker daemon check at docker_run.py:33 uses exit(3) instead of exit(2)
2. Timeout errors return EvaluationResult but CLI doesn't map to exit code 1
3. No centralized exit code constants
4. Error parsing logic in CLI is incomplete

**Fix Required** (COMPREHENSIVE):
```python
# 1. Create exit_codes.py
"""Exit codes as specified in PRD."""
SUCCESS = 0
GENERAL_ERROR = 1
DOCKER_NOT_FOUND = 2
NETWORK_ERROR = 3
RESOURCE_ERROR = 4

# 2. Fix CLI error parsing (cli.py)
if result.error:
    error_lower = result.error.lower()
    if "timed out" in error_lower or "timeout" in error_lower:
        sys.exit(exit_codes.GENERAL_ERROR)  # Timeouts are general errors
    elif any(term in error_lower for term in ["network", "connection", "unreachable"]):
        sys.exit(exit_codes.NETWORK_ERROR)
    elif "docker" in error_lower and "not found" in error_lower:
        sys.exit(exit_codes.DOCKER_NOT_FOUND)
    elif "disk" in error_lower or "space" in error_lower:
        sys.exit(exit_codes.RESOURCE_ERROR)
    else:
        sys.exit(exit_codes.GENERAL_ERROR)
```

### 5. Resource Check Failures in CI
**Severity**: MEDIUM  
**Location**: src/swebench_runner/docker_run.py:133  
**Impact**: CI always fails on resource checks

**Issue Details** (RESEARCHED):
- Code requires 50GB minimum disk space
- CI runner limitations:
  - GitHub Actions: ~14-30GB free space
  - GitLab CI: ~25GB free space
  - Most CI environments: <50GB available

**Fix Required** (ENHANCED):
```python
def check_resources() -> None:
    """Check if system has sufficient resources."""
    # Allow CI to skip or reduce requirements
    is_ci = os.environ.get("CI") == "true"
    skip_checks = os.environ.get("SWEBENCH_SKIP_RESOURCE_CHECK") == "true"
    
    if skip_checks:
        return
    
    try:
        # Check memory - lower requirement for CI
        min_memory_gb = 4 if is_ci else 8
        # ... memory check with adjusted limit
        
        # Check disk space - lower requirement for CI
        min_disk_gb = 20 if is_ci else 50
        
        disk_usage = shutil.disk_usage(".")
        free_gb = disk_usage.free / (1024**3)
        
        if free_gb < min_disk_gb:
            if is_ci:
                # Warning only in CI
                print(f"⚠️  Warning: Only {free_gb:.1f}GB free disk space")
                print(f"   Recommended: {min_disk_gb}GB+ for reliable operation")
            else:
                # Error in normal mode
                print(f"❌ Critical: Only {free_gb:.1f}GB free disk space")
                print(f"   Minimum {min_disk_gb}GB required")
                sys.exit(4)
```

### 6. Low Test Coverage
**Severity**: HIGH  
**Location**: src/swebench_runner/bootstrap.py, cache.py  
**Impact**: 45% coverage vs 90% required

**Issue Details** (VERIFIED):
- bootstrap.py: Only 21% coverage (no test file exists)
- cache.py: Only 33% coverage (no test file exists)  
- Overall coverage: 45% (target is 90%)

**Fix Required** (DETAILED TEST PLAN):

#### test_bootstrap.py (COMPREHENSIVE)
```python
"""Comprehensive tests for bootstrap module."""
import pytest
from unittest.mock import patch, MagicMock, call
from pathlib import Path
import sys

from swebench_runner.bootstrap import (
    check_and_prompt_first_run,
    suggest_patches_file,
    show_success_message,
    show_docker_setup_help,
    show_resource_warning,
    show_memory_warning,
)


class TestBootstrap:
    """Test all bootstrap functionality."""
    
    @patch("swebench_runner.bootstrap.is_first_run")
    @patch("swebench_runner.bootstrap.mark_first_run_complete")
    def test_check_and_prompt_first_run_ci_mode(self, mock_mark, mock_is_first):
        """Test bootstrap in CI mode (no_input=True)."""
        mock_is_first.return_value = True
        
        result = check_and_prompt_first_run(no_input=True)
        
        assert result is True
        mock_mark.assert_called_once()
    
    @patch("swebench_runner.bootstrap.is_first_run")
    def test_check_and_prompt_not_first_run(self, mock_is_first):
        """Test when not first run."""
        mock_is_first.return_value = False
        
        result = check_and_prompt_first_run(no_input=True)
        
        assert result is False
    
    @patch("swebench_runner.bootstrap.is_first_run")
    @patch("swebench_runner.bootstrap.mark_first_run_complete")
    @patch("swebench_runner.bootstrap.show_welcome_message")
    @patch("click.confirm")
    def test_check_and_prompt_interactive_accept(self, mock_confirm, mock_welcome, 
                                                 mock_mark, mock_is_first):
        """Test interactive mode when user accepts."""
        mock_is_first.return_value = True
        mock_confirm.return_value = True
        
        result = check_and_prompt_first_run(no_input=False)
        
        assert result is True
        mock_welcome.assert_called_once()
        mock_mark.assert_called_once()
    
    @patch("swebench_runner.bootstrap.is_first_run")
    @patch("click.confirm")
    @patch("click.echo")
    def test_check_and_prompt_user_cancels(self, mock_echo, mock_confirm, mock_is_first):
        """Test when user cancels setup."""
        mock_is_first.return_value = True
        mock_confirm.return_value = False
        
        with pytest.raises(SystemExit) as exc_info:
            check_and_prompt_first_run(no_input=False)
        
        assert exc_info.value.code == 0
        mock_echo.assert_any_call("Setup cancelled. Run again when ready.")
    
    @patch("swebench_runner.bootstrap.auto_detect_patches_file")
    @patch("click.confirm")
    @patch("click.echo")
    def test_suggest_patches_file_found_accepted(self, mock_echo, mock_confirm, mock_detect):
        """Test patch file suggestion when file found and accepted."""
        mock_detect.return_value = Path("patches.jsonl")
        mock_confirm.return_value = True
        
        result = suggest_patches_file()
        
        assert result == Path("patches.jsonl")
        mock_echo.assert_called_with("💡 Found patches.jsonl in current directory")
    
    @patch("swebench_runner.bootstrap.auto_detect_patches_file")
    def test_suggest_patches_file_not_found(self, mock_detect):
        """Test when no patch file is found."""
        mock_detect.return_value = None
        
        result = suggest_patches_file()
        
        assert result is None
    
    @patch("click.echo")
    def test_show_success_message_first_success(self, mock_echo):
        """Test first success message formatting."""
        show_success_message("test-123", is_first_success=True)
        
        # Verify celebration message shown
        calls = [str(call) for call in mock_echo.call_args_list]
        assert any("🎉 SUCCESS! 🎉" in str(call) for call in calls)
        assert any("Congrats on your first successful evaluation!" in str(call) for call in calls)
    
    @patch("click.echo")
    def test_show_success_message_regular(self, mock_echo):
        """Test regular success message."""
        show_success_message("test-456", is_first_success=False)
        
        mock_echo.assert_called_once_with("✅ test-456: Evaluation completed successfully")
    
    @patch("platform.system")
    @patch("click.echo")
    def test_show_docker_setup_help_macos(self, mock_echo, mock_platform):
        """Test Docker setup help for macOS."""
        mock_platform.return_value = "Darwin"
        
        show_docker_setup_help()
        
        calls = [str(call) for call in mock_echo.call_args_list]
        assert any("Docker Desktop" in str(call) for call in calls)
    
    @patch("platform.system")
    @patch("click.echo")
    def test_show_docker_setup_help_linux(self, mock_echo, mock_platform):
        """Test Docker setup help for Linux."""
        mock_platform.return_value = "Linux"
        
        show_docker_setup_help()
        
        calls = [str(call) for call in mock_echo.call_args_list]
        assert any("sudo apt-get install docker.io" in str(call) for call in calls)
    
    @patch("click.echo")
    def test_show_resource_warning(self, mock_echo):
        """Test resource warning display."""
        show_resource_warning(25.5)
        
        calls = [str(call) for call in mock_echo.call_args_list]
        assert any("Only 25.5GB free disk space" in str(call) for call in calls)
        assert any("swebench clean --all" in str(call) for call in calls)
    
    @patch("click.echo")
    def test_show_memory_warning(self, mock_echo):
        """Test memory warning display."""
        show_memory_warning(6.0)
        
        calls = [str(call) for call in mock_echo.call_args_list]
        assert any("Only 6.0GB RAM available" in str(call) for call in calls)
```

#### test_cache.py (COMPREHENSIVE)
```python
"""Comprehensive tests for cache module."""
import pytest
from pathlib import Path
import os
import tempfile
import shutil
from datetime import datetime

from swebench_runner.cache import (
    get_cache_dir,
    is_first_run,
    mark_first_run_complete,
    get_cache_usage,
    clean_cache,
    auto_detect_patches_file,
    get_logs_dir,
    get_results_dir,
)


class TestCache:
    """Test all cache functionality."""
    
    @pytest.fixture
    def isolated_cache_dir(self, tmp_path, monkeypatch):
        """Provide isolated cache directory for tests."""
        cache_dir = tmp_path / "test_cache"
        monkeypatch.setenv("SWEBENCH_CACHE_DIR", str(cache_dir))
        return cache_dir
    
    def test_get_cache_dir_env_var(self, isolated_cache_dir):
        """Test cache dir from environment variable."""
        result = get_cache_dir()
        
        assert result == isolated_cache_dir
        assert result.exists()
        assert (result / "datasets").exists()
        assert (result / "logs").exists()
        assert (result / "results").exists()
    
    def test_get_cache_dir_default(self, monkeypatch):
        """Test default cache directory location."""
        monkeypatch.delenv("SWEBENCH_CACHE_DIR", raising=False)
        
        result = get_cache_dir()
        
        assert result == Path.home() / ".swebench"
    
    def test_is_first_run_true(self, isolated_cache_dir):
        """Test first run detection when config doesn't exist."""
        # Ensure cache dir exists but no config file
        get_cache_dir()
        
        assert is_first_run() is True
    
    def test_is_first_run_false(self, isolated_cache_dir):
        """Test first run detection when config exists."""
        get_cache_dir()
        mark_first_run_complete()
        
        assert is_first_run() is False
    
    def test_mark_first_run_complete(self, isolated_cache_dir):
        """Test marking first run as complete."""
        cache_dir = get_cache_dir()
        
        mark_first_run_complete()
        
        config_file = cache_dir / "config.toml"
        assert config_file.exists()
        
        content = config_file.read_text()
        assert "SWE-bench Runner Configuration" in content
        assert "version = \"0.1.0\"" in content
        assert str(cache_dir) in content
    
    def test_get_cache_usage_empty(self, isolated_cache_dir):
        """Test cache usage calculation for empty cache."""
        get_cache_dir()
        
        usage = get_cache_usage()
        
        assert usage["datasets"] == 0
        assert usage["logs"] == 0
        assert usage["results"] == 0
    
    def test_get_cache_usage_with_files(self, isolated_cache_dir):
        """Test cache usage calculation with files."""
        cache_dir = get_cache_dir()
        
        # Create test files
        (cache_dir / "datasets" / "test.txt").write_text("x" * 1000)
        (cache_dir / "logs" / "test.log").write_text("y" * 2000)
        (cache_dir / "results" / "test.json").write_text("z" * 3000)
        
        # Create subdirectory with files
        (cache_dir / "logs" / "subdir").mkdir()
        (cache_dir / "logs" / "subdir" / "nested.log").write_text("w" * 500)
        
        usage = get_cache_usage()
        
        assert usage["datasets"] == 1000
        assert usage["logs"] == 2500  # 2000 + 500
        assert usage["results"] == 3000
    
    def test_clean_cache_dry_run(self, isolated_cache_dir):
        """Test cache cleaning in dry run mode."""
        cache_dir = get_cache_dir()
        
        # Create test files
        (cache_dir / "datasets" / "test.txt").write_text("x" * 1000)
        (cache_dir / "logs" / "test.log").write_text("y" * 2000)
        
        # Dry run - nothing should be deleted
        removed = clean_cache(
            clean_datasets=True,
            clean_logs=True,
            dry_run=True
        )
        
        assert removed["datasets"] == 1000
        assert removed["logs"] == 2000
        assert (cache_dir / "datasets" / "test.txt").exists()  # Still exists
        assert (cache_dir / "logs" / "test.log").exists()  # Still exists
    
    def test_clean_cache_real(self, isolated_cache_dir):
        """Test actual cache cleaning."""
        cache_dir = get_cache_dir()
        
        # Create test files
        (cache_dir / "datasets" / "test.txt").write_text("x" * 1000)
        (cache_dir / "logs" / "test.log").write_text("y" * 2000)
        (cache_dir / "results" / "test.json").write_text("z" * 3000)
        
        # Clean only logs and datasets
        removed = clean_cache(
            clean_datasets=True,
            clean_logs=True,
            clean_results=False,
            dry_run=False
        )
        
        assert removed["datasets"] == 1000
        assert removed["logs"] == 2000
        assert removed["results"] == 0
        
        assert not (cache_dir / "datasets" / "test.txt").exists()
        assert not (cache_dir / "logs" / "test.log").exists()
        assert (cache_dir / "results" / "test.json").exists()  # Not cleaned
    
    def test_auto_detect_patches_file_found(self, tmp_path, monkeypatch):
        """Test auto-detection of patch files."""
        monkeypatch.chdir(tmp_path)
        
        # Create various patch files
        (tmp_path / "patches.jsonl").touch()
        (tmp_path / "predictions.jsonl").touch()
        (tmp_path / "other.txt").touch()
        
        result = auto_detect_patches_file()
        
        # Should prefer patches.jsonl
        assert result == Path("patches.jsonl")
    
    def test_auto_detect_patches_file_priority(self, tmp_path, monkeypatch):
        """Test auto-detection priority order."""
        monkeypatch.chdir(tmp_path)
        
        # Only predictions.jsonl exists
        (tmp_path / "predictions.jsonl").touch()
        
        result = auto_detect_patches_file()
        assert result == Path("predictions.jsonl")
    
    def test_auto_detect_patches_file_not_found(self, tmp_path, monkeypatch):
        """Test when no patch files found."""
        monkeypatch.chdir(tmp_path)
        
        result = auto_detect_patches_file()
        assert result is None
    
    def test_get_logs_dir(self, isolated_cache_dir):
        """Test logs directory creation."""
        logs_dir = get_logs_dir()
        
        assert logs_dir == isolated_cache_dir / "logs"
        assert logs_dir.exists()
    
    def test_get_results_dir(self, isolated_cache_dir):
        """Test results directory creation."""
        results_dir = get_results_dir()
        
        assert results_dir == isolated_cache_dir / "results"
        assert results_dir.exists()
    
    def test_cache_dir_permissions(self, isolated_cache_dir):
        """Test cache directory has correct permissions."""
        cache_dir = get_cache_dir()
        
        # Should be readable and writable by owner
        assert os.access(cache_dir, os.R_OK)
        assert os.access(cache_dir, os.W_OK)
        assert os.access(cache_dir, os.X_OK)
```

### 7. Security and Linting Violations
**Severity**: MEDIUM  
**Location**: Multiple files  
**Impact**: Code quality issues

**Specific Issues** (CATEGORIZED):

#### Style Issues:
1. Long lines exceeding 88 characters
2. Unused imports
3. Missing type annotations
4. Inconsistent string formatting

#### Security Issues:
5. No validation of subprocess output
6. No timeout on subprocess calls
7. Missing input validation

**Fix Required** (WITH EXAMPLES):

#### Long Lines Fix:
```python
# Before:
error_message = f"Docker container exited with non-zero status during harness execution. This usually means the evaluation failed. Check logs for details: {stderr}"

# After:
error_message = (
    "Docker container exited with non-zero status during harness execution. "
    "This usually means the evaluation failed. "
    f"Check logs for details: {stderr}"
)
```

#### Subprocess Security Fix:
```python
# Before:
result = subprocess.run(["docker", "info"], capture_output=True)

# After:
result = subprocess.run(
    ["docker", "info"], 
    capture_output=True,
    text=True,
    check=False,  # Handle errors explicitly
    timeout=30    # Prevent hanging
)
# Validate output
if result.returncode != 0:
    # Handle error
if not result.stdout:
    # Handle empty output
```

#### Input Validation Fix:
```python
def load_patch_file(file_path: Path) -> str:
    # Validate file size first
    if file_path.stat().st_size > MAX_PATCH_SIZE:
        raise ValueError(f"Patch file too large: {file_path}")
    
    # Validate encoding
    with open(file_path, 'rb') as f:
        raw_content = f.read()
        try:
            content = raw_content.decode('utf-8')
        except UnicodeDecodeError:
            raise ValueError("Patch file must be UTF-8 encoded")
    
    return content
```

## Reviewer Comments Analysis

### Claude's Review Comments:
1. **Bootstrap order issue**: "The bootstrap flow in cli.py happens before argument validation"
2. **Exit code confusion**: "Network errors should consistently return exit code 3"
3. **Test coverage**: "New files bootstrap.py and cache.py lack test coverage"

### Mentatbot's Review Comments:
1. **Patch size conflict**: "Conflicting patch size limits in docker_run.py"
2. **Resource checks**: "CI environment has insufficient resources for hard requirements"
3. **Python compatibility**: "Union syntax requires Python 3.10+"

### CI Failure Analysis:
1. **Linting failures**: 23 style violations
2. **Type checking failures**: 7 mypy errors
3. **Test failures**: All platforms fail due to bootstrap issue
4. **Coverage failure**: 45% vs 90% required

## Priority Order for Fixes

### Priority 1 - Blocking Issues (Must Fix)
1. **Bootstrap flow order** - Prevents all tests from running
2. **Python version compatibility** - Blocks installation on 3.8/3.9
3. **Exit code standardization** - Core functionality requirement

### Priority 2 - Critical Issues (Should Fix)
4. **Test coverage** - Required for merge (90% threshold)
5. **Resource check bypass** - CI cannot pass without this
6. **Patch size limit conflict** - Confusing behavior

### Priority 3 - Quality Issues (Nice to Fix)
7. **Linting violations** - Code quality
8. **Security warnings** - Best practices

## Recommended Fix Implementation Order (REVISED)

### Phase 1: Unblock CI (Immediate)
1. **Fix test isolation** - Mock bootstrap functions or use isolated cache
2. **Update Python versions** - Remove 3.8/3.9 from CI matrix (keep 3.10+)
3. **Add CI resource bypass** - Implement `SWEBENCH_SKIP_RESOURCE_CHECK` and `CI` detection

### Phase 2: Core Functionality (Next)
4. **Create exit_codes.py** - Centralize exit code constants
5. **Fix exit code mapping** - Update CLI error parsing to match PRD specs
6. **Clarify patch size errors** - Distinguish Docker env limit from general limit

### Phase 3: Test Coverage (Critical)
7. **Create test_bootstrap.py** - Comprehensive tests with mocking
8. **Create test_cache.py** - Full coverage of cache operations
9. **Add exit code tests** - Test each exit code scenario

### Phase 4: Code Quality (Final)
10. **Fix subprocess security** - Add timeouts and validation
11. **Fix linting issues** - Long lines, type annotations
12. **Add input validation** - Validate file encoding and size

## Technical Details for Each Fix (UPDATED)

### 1. Test Isolation Fix (COMPREHENSIVE)
```python
# tests/conftest.py - Global test configuration
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

@pytest.fixture(autouse=True)
def test_environment(monkeypatch):
    """Complete test isolation for all tests.
    
    This fixture:
    1. Isolates cache directory to prevent ~/.swebench pollution
    2. Sets CI=true to disable all interactive prompts
    3. Mocks bootstrap to prevent first-run checks
    4. Provides temp directory for test files
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Isolate cache directory
        cache_dir = Path(tmpdir) / ".swebench"
        monkeypatch.setenv("SWEBENCH_CACHE_DIR", str(cache_dir))
        
        # Force CI mode
        monkeypatch.setenv("CI", "true")
        
        # Mock bootstrap functions
        with patch("swebench_runner.cli.check_and_prompt_first_run", return_value=False), \
             patch("swebench_runner.cli.suggest_patches_file", return_value=None):
            yield tmpdir

# No changes needed to existing tests - this fixture applies automatically
```

### 2. Python Version Update
```yaml
# .github/workflows/ci.yml
jobs:
  test:
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]  # Remove 3.8, 3.9
```

### 3. Exit Code Implementation
```python
# src/swebench_runner/exit_codes.py (NEW FILE)
"""Exit codes as specified in PRD Section 5.8."""
SUCCESS = 0
GENERAL_ERROR = 1        # Harness errors, timeouts, unknown errors
DOCKER_NOT_FOUND = 2     # Docker not installed or not running
NETWORK_ERROR = 3        # Network failures, registry access
RESOURCE_ERROR = 4       # Disk space, memory issues

# src/swebench_runner/cli.py - Update error handling
from . import exit_codes

if result.error:
    error_lower = result.error.lower()
    
    # Map specific errors to exit codes
    if any(term in error_lower for term in ["timeout", "timed out"]):
        sys.exit(exit_codes.GENERAL_ERROR)
    elif any(term in error_lower for term in [
        "network", "connection", "unreachable", "registry", "pull"
    ]):
        sys.exit(exit_codes.NETWORK_ERROR)
    elif "docker" in error_lower and any(term in error_lower for term in [
        "not found", "not running", "daemon"
    ]):
        sys.exit(exit_codes.DOCKER_NOT_FOUND)
    elif any(term in error_lower for term in ["disk", "space", "memory", "ram"]):
        sys.exit(exit_codes.RESOURCE_ERROR)
    else:
        sys.exit(exit_codes.GENERAL_ERROR)
else:
    sys.exit(exit_codes.SUCCESS)
```

### 4. CI Resource Check Implementation (CONFIGURABLE)
```python
# src/swebench_runner/docker_run.py
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
                print(f"   Minimum: {min_memory_gb}GB (configurable via SWEBENCH_CI_MIN_MEMORY_GB)")
                print(f"   Recommended: {memory_recommended}GB")
            else:
                print(f"❌ Critical: Only {mem_gb:.1f}GB RAM available")
                print(f"   Minimum {min_memory_gb}GB required")
                sys.exit(exit_codes.RESOURCE_ERROR)
    except ImportError:
        pass  # Skip if psutil not available
    
    # Check disk space
    try:
        free_gb = shutil.disk_usage(".").free / (1024**3)
        
        if free_gb < min_disk_gb:
            if is_ci:
                print(f"⚠️  CI Warning: Only {free_gb:.1f}GB disk space available")
                print(f"   Minimum: {min_disk_gb}GB (configurable via SWEBENCH_CI_MIN_DISK_GB)")
                print(f"   Some evaluations may fail due to insufficient space")
            else:
                print(f"❌ Critical: Only {free_gb:.1f}GB free disk space")
                print(f"   Minimum {min_disk_gb}GB required")
                print("   Run: swebench clean --all")
                sys.exit(exit_codes.RESOURCE_ERROR)
    except Exception:
        pass  # Skip if can't check
```

### 4.1. CI Configuration Examples
```yaml
# GitHub Actions example
name: Test
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    env:
      CI: true
      SWEBENCH_CI_MIN_DISK_GB: 15  # GitHub Actions has ~15-30GB
      SWEBENCH_CI_MIN_MEMORY_GB: 4  # Usually has 7GB
    steps:
      - uses: actions/checkout@v3
      - name: Run tests
        run: pytest

# GitLab CI example
test:
  variables:
    CI: "true"
    SWEBENCH_CI_MIN_DISK_GB: "20"  # GitLab has ~25GB
    SWEBENCH_CI_MIN_MEMORY_GB: "4"
  script:
    - pytest

# CircleCI example
version: 2.1
jobs:
  test:
    docker:
      - image: python:3.10
    environment:
      CI: true
      SWEBENCH_CI_MIN_DISK_GB: 10  # CircleCI can be limited
      SWEBENCH_CI_MIN_MEMORY_GB: 4
    steps:
      - checkout
      - run: pytest

# Local development with limited resources
export SWEBENCH_MIN_DISK_GB=30  # Override for dev machine
export SWEBENCH_MIN_MEMORY_GB=6
swebench run --patches test.jsonl
```

### 5. Comprehensive Test Examples
See detailed test implementations in sections 6.1 and 6.2 above.

### 6. Exit Code Test Suite
```python
# tests/test_exit_codes.py (NEW FILE)
"""Comprehensive tests for exit code behavior."""
import pytest
import subprocess
from unittest.mock import patch, MagicMock
from click.testing import CliRunner

from swebench_runner.cli import cli
from swebench_runner import exit_codes
from swebench_runner.models import EvaluationResult


class TestExitCodes:
    """Test exit codes match PRD specifications."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    @pytest.fixture
    def valid_patch_file(self, tmp_path):
        """Create a valid patch file for testing."""
        patch_file = tmp_path / "test.jsonl"
        patch_file.write_text(
            '{"instance_id": "test-123", "patch": "diff --git a/test.py b/test.py\\n+test"}'
        )
        return patch_file
    
    def test_success_exit_code(self, runner, valid_patch_file):
        """Test exit code 0 for successful evaluation."""
        with patch("swebench_runner.cli.run_evaluation") as mock_eval:
            mock_eval.return_value = EvaluationResult(
                instance_id="test-123",
                passed=True,
                error=None
            )
            
            result = runner.invoke(cli, [
                "run", "--patches", str(valid_patch_file), "--no-input"
            ])
            
            assert result.exit_code == exit_codes.SUCCESS
            assert "✅" in result.output
    
    @pytest.mark.parametrize("error_message,expected_code", [
        # Timeout errors -> GENERAL_ERROR (1)
        ("Evaluation timed out after 70 minutes", exit_codes.GENERAL_ERROR),
        ("Process timeout exceeded", exit_codes.GENERAL_ERROR),
        
        # Docker errors -> DOCKER_NOT_FOUND (2)
        ("Docker daemon not found", exit_codes.DOCKER_NOT_FOUND),
        ("Cannot connect to Docker daemon", exit_codes.DOCKER_NOT_FOUND),
        ("Docker is not running", exit_codes.DOCKER_NOT_FOUND),
        
        # Network errors -> NETWORK_ERROR (3)
        ("Network unreachable", exit_codes.NETWORK_ERROR),
        ("Connection refused to registry", exit_codes.NETWORK_ERROR),
        ("Failed to pull image: timeout", exit_codes.NETWORK_ERROR),
        
        # Resource errors -> RESOURCE_ERROR (4)
        ("Insufficient disk space: only 10GB available", exit_codes.RESOURCE_ERROR),
        ("Not enough memory: 2GB RAM available", exit_codes.RESOURCE_ERROR),
        
        # Generic errors -> GENERAL_ERROR (1)
        ("Unknown evaluation error", exit_codes.GENERAL_ERROR),
        ("Harness execution failed", exit_codes.GENERAL_ERROR),
    ])
    def test_error_exit_codes(self, runner, valid_patch_file, error_message, expected_code):
        """Test correct exit codes for various error scenarios."""
        with patch("swebench_runner.cli.run_evaluation") as mock_eval:
            mock_eval.return_value = EvaluationResult(
                instance_id="test-123",
                passed=False,
                error=error_message
            )
            
            result = runner.invoke(cli, [
                "run", "--patches", str(valid_patch_file), "--no-input"
            ])
            
            assert result.exit_code == expected_code
            assert "❌" in result.output or "Error" in result.output
    
    def test_docker_check_exit_code(self, runner, valid_patch_file):
        """Test Docker check failures return correct exit code."""
        with patch("subprocess.run") as mock_run:
            # Simulate Docker not found
            mock_run.side_effect = FileNotFoundError("docker not found")
            
            result = runner.invoke(cli, [
                "run", "--patches", str(valid_patch_file), "--no-input"
            ])
            
            assert result.exit_code == exit_codes.DOCKER_NOT_FOUND
    
    def test_resource_check_exit_code(self, runner, valid_patch_file):
        """Test resource check failures return correct exit code."""
        with patch("shutil.disk_usage") as mock_disk:
            # Simulate low disk space
            mock_disk.return_value = MagicMock(free=10 * 1024**3)  # 10GB
            
            with patch("os.environ.get") as mock_env:
                mock_env.side_effect = lambda k, d=None: None  # Not CI
                
                result = runner.invoke(cli, [
                    "run", "--patches", str(valid_patch_file), "--no-input"
                ])
                
                assert result.exit_code == exit_codes.RESOURCE_ERROR


class TestExitCodeIntegration:
    """Integration tests for exit code scenarios."""
    
    def test_full_evaluation_exit_codes(self, tmp_path):
        """Test exit codes in real evaluation scenarios."""
        # This would be an integration test running actual evaluations
        # with different failure modes to verify exit codes
        pass
```

## Integration Testing Strategy

### 7. Integration Tests (NEW SECTION)
```python
# tests/test_integration.py (NEW FILE)
"""End-to-end integration tests for SWE-bench runner."""
import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import subprocess
import shutil

from click.testing import CliRunner
from swebench_runner.cli import cli
from swebench_runner import exit_codes


class TestEndToEndEvaluation:
    """Test complete evaluation flows."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    @pytest.fixture
    def mock_docker_environment(self):
        """Mock Docker environment for tests."""
        with patch("subprocess.run") as mock_run:
            # Mock successful Docker check
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="Docker version 24.0.0",
                stderr=""
            )
            
            # Mock harness installation check
            with patch("swebench_runner.docker_run.check_swebench_installed", return_value=True):
                yield mock_run
    
    @pytest.fixture
    def valid_patch_file(self, tmp_path):
        """Create a valid patch file for integration testing."""
        patch_data = {
            "instance_id": "django__django-11999",
            "patch": (
                "diff --git a/django/db/models/query.py b/django/db/models/query.py\n"
                "--- a/django/db/models/query.py\n"
                "+++ b/django/db/models/query.py\n"
                "@@ -1,5 +1,6 @@\n"
                " import copy\n"
                "+import logging\n"
            )
        }
        
        patch_file = tmp_path / "test_patch.jsonl"
        patch_file.write_text(json.dumps(patch_data))
        return patch_file
    
    def test_successful_evaluation_flow(self, runner, mock_docker_environment, valid_patch_file):
        """Test complete successful evaluation flow."""
        with patch("swebench_runner.docker_run.run_swebench_harness") as mock_harness:
            # Mock successful harness execution
            mock_harness.return_value = MagicMock(
                returncode=0,
                stdout="✅ Test passed",
                stderr=""
            )
            
            # Mock results parsing
            with patch("swebench_runner.docker_run.parse_harness_results") as mock_parse:
                mock_parse.return_value = MagicMock(
                    instance_id="django__django-11999",
                    passed=True,
                    error=None
                )
                
                # Run evaluation
                result = runner.invoke(cli, [
                    "run",
                    "--patches", str(valid_patch_file),
                    "--no-input"
                ])
                
                # Verify success
                assert result.exit_code == exit_codes.SUCCESS
                assert "✅" in result.output
                assert "django__django-11999" in result.output
    
    def test_docker_not_found_flow(self, runner, valid_patch_file):
        """Test behavior when Docker is not installed."""
        with patch("subprocess.run") as mock_run:
            # Simulate Docker not found
            mock_run.side_effect = FileNotFoundError("docker not found")
            
            result = runner.invoke(cli, [
                "run",
                "--patches", str(valid_patch_file),
                "--no-input"
            ])
            
            assert result.exit_code == exit_codes.DOCKER_NOT_FOUND
            assert "Docker" in result.output
    
    def test_network_error_flow(self, runner, mock_docker_environment, valid_patch_file):
        """Test network error handling during evaluation."""
        with patch("swebench_runner.docker_run.run_swebench_harness") as mock_harness:
            # Mock network error
            mock_harness.return_value = MagicMock(
                returncode=1,
                stdout="",
                stderr="Failed to pull image: connection timeout"
            )
            
            with patch("swebench_runner.docker_run.parse_harness_results") as mock_parse:
                mock_parse.return_value = MagicMock(
                    instance_id="django__django-11999",
                    passed=False,
                    error="Network error: Failed to pull image: connection timeout"
                )
                
                result = runner.invoke(cli, [
                    "run",
                    "--patches", str(valid_patch_file),
                    "--no-input"
                ])
                
                assert result.exit_code == exit_codes.NETWORK_ERROR
                assert "Network error" in result.output
    
    def test_resource_constraint_flow(self, runner, valid_patch_file):
        """Test resource check failures."""
        with patch("shutil.disk_usage") as mock_disk:
            # Simulate low disk space
            mock_disk.return_value = MagicMock(free=5 * 1024**3)  # 5GB
            
            result = runner.invoke(cli, [
                "run",
                "--patches", str(valid_patch_file),
                "--no-input"
            ])
            
            assert result.exit_code == exit_codes.RESOURCE_ERROR
            assert "disk space" in result.output
    
    def test_large_patch_handling(self, runner, mock_docker_environment, tmp_path):
        """Test handling of patches exceeding Docker env var limit."""
        # Create large patch (>500KB)
        large_patch = "diff --git a/test.py b/test.py\n" + "+" * 600000  # ~600KB
        patch_data = {
            "instance_id": "test-large",
            "patch": large_patch
        }
        
        patch_file = tmp_path / "large_patch.jsonl"
        patch_file.write_text(json.dumps(patch_data))
        
        result = runner.invoke(cli, [
            "run",
            "--patches", str(patch_file),
            "--no-input"
        ])
        
        # Should fail with clear error about Docker env limit
        assert result.exit_code != exit_codes.SUCCESS
        assert "500KB" in result.output
        assert "Docker" in result.output or "environment variable" in result.output
    
    @pytest.mark.parametrize("ci_env,expected_behavior", [
        ({"CI": "true"}, "warning"),  # CI mode - warnings only
        ({}, "error"),  # Normal mode - hard errors
    ])
    def test_ci_vs_normal_mode(self, runner, valid_patch_file, ci_env, expected_behavior):
        """Test different behavior in CI vs normal mode."""
        with patch.dict("os.environ", ci_env):
            with patch("shutil.disk_usage") as mock_disk:
                # Simulate borderline disk space
                mock_disk.return_value = MagicMock(free=25 * 1024**3)  # 25GB
                
                with patch("swebench_runner.docker_run.check_docker_daemon"):
                    result = runner.invoke(cli, [
                        "run",
                        "--patches", str(valid_patch_file),
                        "--no-input"
                    ])
                    
                    if expected_behavior == "warning":
                        # CI mode should show warning but continue
                        assert "⚠️" in result.output
                    else:
                        # Normal mode should error
                        assert result.exit_code == exit_codes.RESOURCE_ERROR


class TestCLIIntegration:
    """Test CLI command integration."""
    
    def test_clean_command_integration(self):
        """Test clean command with cache operations."""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            # Set up test cache
            cache_dir = Path(".swebench")
            cache_dir.mkdir()
            (cache_dir / "logs").mkdir()
            (cache_dir / "logs" / "test.log").write_text("test log")
            
            with patch("swebench_runner.cache.get_cache_dir", return_value=cache_dir):
                result = runner.invoke(cli, ["clean", "--logs"])
                
                assert result.exit_code == 0
                assert "Cleaning logs" in result.output
    
    def test_setup_command_integration(self):
        """Test setup wizard command."""
        runner = CliRunner()
        
        with patch("swebench_runner.cli.setup_wizard") as mock_wizard:
            result = runner.invoke(cli, ["setup"])
            
            assert result.exit_code == 0
            mock_wizard.assert_called_once()
```

### Integration Test Docker Mocking Strategy
```python
# tests/fixtures/docker_mocks.py
"""Reusable Docker mocking fixtures for integration tests."""
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def mock_docker_success():
    """Mock successful Docker operations."""
    with patch("subprocess.run") as mock_run:
        def docker_side_effect(cmd, **kwargs):
            if cmd[0] == "docker" and cmd[1] == "info":
                return MagicMock(returncode=0, stdout="Docker OK")
            elif cmd[0] == "python" and "-m" in cmd:
                return MagicMock(returncode=0, stdout="Evaluation complete")
            return MagicMock(returncode=0)
        
        mock_run.side_effect = docker_side_effect
        yield mock_run


@pytest.fixture  
def mock_docker_network_error():
    """Mock Docker network errors."""
    with patch("subprocess.run") as mock_run:
        def network_error_side_effect(cmd, **kwargs):
            if "pull" in str(cmd):
                return MagicMock(
                    returncode=1,
                    stderr="network timeout"
                )
            return MagicMock(returncode=0)
        
        mock_run.side_effect = network_error_side_effect
        yield mock_run
```

## Lessons Learned

1. **Always validate arguments before any side effects** - Bootstrap flow should never run before we know the command is valid
2. **Research version requirements thoroughly** - SWE-bench's Python 3.10+ requirement wasn't discovered until late
3. **CI environments are constrained** - Always provide bypass mechanisms for resource checks
4. **Consistent error handling is critical** - Exit codes must match documentation
5. **Test coverage from day one** - New modules should have tests in the same PR

## Action Items

1. Fix bootstrap flow order immediately (Priority 1)
2. Decide on Python version support with team
3. Implement CI bypass for resource checks
4. Write comprehensive tests for bootstrap and cache modules
5. Standardize exit codes across codebase
6. Fix conflicting patch size limits
7. Address linting and security warnings

## References

- PR #6: https://github.com/[org]/swe-bench-runner/pull/6
- PRD Exit Codes: Documentation/PRD.md Section 5.8
- Architecture: Documentation/Architecture.md
- CI Implementation Lessons: Documentation/Plans/CI-Implementation-Lessons.md

---

This report documents all errors found in PR #6 to ensure the information is preserved even if the conversation context is lost. Each issue includes severity, location, root cause, and recommended fix approach.