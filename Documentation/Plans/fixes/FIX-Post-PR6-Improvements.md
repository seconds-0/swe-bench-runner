# Post-PR #6 Improvements Plan

## Overview
This plan addresses feedback from PR #6 review and CI issues to improve development workflow and code quality. Focus is on high-value fixes that solve real problems.

## Phase 1: High-Value Code Fixes

### 1.1 Fix Patch Size Limit Conflict ✅ **MUST FIX**
**Issue**: Conflicting patch size limits causing user confusion
**Location**: `docker_run.py:342-352`

**Problem**: Code has hardcoded 500KB check that ignores the configurable `max_patch_size_mb` parameter

**Simple Fix**:
```python
# In docker_run.py, replace the hardcoded check with:
def load_first_patch(patch_source: str, max_size_mb: int = 5) -> Patch:
    """Load patch with proper size validation."""
    # ... existing code ...
    
    # Remove the hardcoded 500KB check entirely
    # The model validation already handles size limits
    patch.validate(max_size_mb=max_size_mb)
    
    # Add informational warning for Docker limits
    patch_size_kb = len(patch.patch.encode('utf-8')) / 1024
    if patch_size_kb > 500:  # Docker env var limit
        print(f"ℹ️  Note: {patch_size_kb:.0f}KB patch may exceed Docker environment limits")
    
    return patch
```

### 1.2 Extract Network Error Detection ✅ **GOOD VALUE**
**Issue**: Same error patterns duplicated in multiple places

**Simple Solution**: Just move the existing patterns to a shared location
```python
# In cli.py, extract the existing error mapping logic:
# Move lines 136-153 to a new file src/swebench_runner/error_utils.py

def classify_error(error_message: str) -> int:
    """Classify error message to exit code. Moved from cli.py."""
    error_lower = error_message.lower()
    
    # Existing logic from cli.py, just moved to be reusable
    if any(term in error_lower for term in [
        "network", "connection", "unreachable", "registry", "pull",
        "resolve", "dns", "connection refused", "failed to pull"
    ]):
        return exit_codes.NETWORK_ERROR
    # ... rest of existing logic ...
```

### 1.3 Basic Input Validation 🔒 **SECURITY**
**Issue**: run_id and instance_id used in commands/filenames without validation

**Minimal Fix**:
```python
# Add to models.py in Patch.__post_init__:
if not re.match(r'^[a-zA-Z0-9_\-\.]+$', self.instance_id):
    raise ValueError(f"Invalid instance_id format: {self.instance_id}")
```

### ~~1.4 Skip Bootstrap/Cache Tests~~ ❌ **NOT NEEDED**
These modules already have 100% and 99% coverage respectively.

### ~~1.5 Skip Unicode Handling~~ ❌ **LOW VALUE**
Current behavior is fine. Adding exit messages is cosmetic.

## Phase 2: Practical CI Improvements

### 2.1 Simple Pre-commit Hook ✅ **PREVENTS WHITESPACE ISSUES**
**Create**: `.pre-commit-config.yaml`

```yaml
# Just the essentials - fix the actual problems we had
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.0
    hooks:
      - id: ruff
        args: [--fix]  # Auto-fix whitespace issues
```

**Install**: `pre-commit install`

### 2.2 Minimal CI Check Script ✅ **CATCH ISSUES LOCALLY**
**Create**: `scripts/check.sh`

```bash
#!/bin/bash
# Quick CI check - run before pushing
set -e

echo "Running quick CI checks..."

# Only the checks that actually failed in PR #6
ruff check src/ tests/ || (echo "❌ Lint failed - run: ruff check --fix" && exit 1)
mypy src/ || (echo "❌ Type check failed" && exit 1)
pytest tests/test_cli_critical.py -xvs || (echo "❌ CLI tests failed" && exit 1)

echo "✅ Basic checks passed - OK to push"
```

### 2.3 Test Without Docker Script 🐳 **TEST LIKE CI**
**Create**: `scripts/test-no-docker.sh`

```bash
#!/bin/bash
# Test in CI-like environment (no Docker)

# Stop Docker to simulate CI
echo "Stopping Docker to simulate CI environment..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    osascript -e 'quit app "Docker"' 2>/dev/null || true
else
    sudo systemctl stop docker 2>/dev/null || true
fi

# Run tests
pytest tests/ -v

echo "Remember to restart Docker!"
```

### ~~2.4 Skip Complex Environment Setup~~ ❌ **OVER-ENGINEERING**
Virtual environments and elaborate setups are overkill.

### ~~2.5 Skip Coverage Config~~ ❌ **NOT THE PROBLEM**
Coverage wasn't the issue - we already lowered it to 85%.

## Phase 3: CLAUDE.md Updates

### 3.1 New Testing Rules
Add to "Critical Implementation Rules" section:

```markdown
### 11. Test Environment Parity (MANDATORY)
**Rule**: Always test in an environment that matches CI
- **No Docker testing**: Test without Docker to match CI environment
- **Mock at call site**: Mock functions where they're called, not where they're defined
- **Resource constraints**: Test with CI-level resource limits
- **Clean environment**: Test with minimal dependencies installed
- **Platform testing**: Test on both macOS and Linux when possible

### 12. Pre-Push Validation (MANDATORY)
**Rule**: Run these checks before every push
1. `./scripts/ci-local.sh` - Run all CI checks locally
2. `ruff check --fix` - Auto-fix any linting issues
3. `mypy src/` - Ensure type checking passes
4. Test without Docker running
5. Verify coverage is above threshold

### 13. Error Handling Standards (MANDATORY)
**Rule**: Use consistent error handling patterns
- **Use error_utils**: Always use `classify_error()` for exit codes
- **Clear messages**: Every error must explain how to fix it
- **Platform-specific**: Provide platform-specific fix instructions
- **Exit codes**: Use only the defined exit codes from exit_codes.py
- **Logging**: Log errors for debugging but keep user messages clean

### 14. Documentation Requirements (MANDATORY)
**Rule**: Document all non-obvious decisions
- **Magic numbers**: Every hardcoded value needs a comment explaining why
- **Environment limits**: Document why limits exist (e.g., Docker env var size)
- **Platform differences**: Document any platform-specific behavior
- **Test skips**: Document why tests are skipped with clear reasons
- **Mocking strategy**: Document why mocks are at specific levels
```

### 3.2 New Development Workflow
Add new section:

```markdown
## Development Workflow

### Before Starting Work
1. Pull latest main
2. Run `./scripts/setup-ci-env.sh` to create clean environment
3. Run `pre-commit install` to set up hooks
4. Create a workplan in `Documentation/Plans/`

### During Development  
1. Write tests first (TDD)
2. Run `./scripts/ci-local.sh` frequently
3. Test with Docker stopped
4. Test with minimal resources
5. Commit with descriptive messages

### Before Pushing
1. Run `./scripts/ci-local.sh`
2. Test in fresh virtual environment
3. Check coverage report for meaningful gaps
4. Update documentation if needed
5. Self-review the diff

### After CI Failure
1. Check exact error in CI logs
2. Reproduce locally with CI environment
3. Fix and test locally
4. Document the issue in commit message
```

## Implementation Order

1. **Week 1**: 
   - Set up pre-commit hooks
   - Create local CI scripts
   - Fix patch size limit conflict

2. **Week 2**:
   - Add missing tests
   - Extract error utilities
   - Update documentation

3. **Week 3**:
   - Update CLAUDE.md
   - Create testing guidelines
   - Team training on new workflow

## Success Metrics

- CI failures reduced by 80%
- Average PR cycle time reduced by 50%
- No more whitespace/linting failures
- Clear error messages that help users
- Tests that actually test behavior

## Notes

This plan addresses the root causes of PR #6's issues:
- Environment differences between local and CI
- Unclear requirements and limits
- Missing tooling for pre-push validation
- Lack of clear testing guidelines
- No standardized error handling

By implementing these changes, future PRs should have significantly fewer CI iterations.