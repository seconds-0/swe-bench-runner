# CI vs Local Environment Analysis

## Key Finding
**The CI is failing because all our fixes are uncommitted.** The CI runs against the code in the PR branch, not our local modifications.

## Environmental Differences

### 1. **Code State**
- **Local**: Has all the fixes we've implemented (uncommitted)
- **CI**: Running against the PR branch without any fixes

### 2. **Uncommitted Files with Fixes**
```
.github/workflows/ci.yml         - CI configuration updates
src/swebench_runner/docker_run.py - Security fixes, exit codes
src/swebench_runner/models.py     - Type annotation fixes
tests/test_bootstrap.py           - Line length fixes
tests/test_cache.py               - Line length fixes
tests/test_cli_extended.py        - Mock paths, skipped tests
tests/test_docker_run.py          - Exit code fixes
tests/test_docker_run_extended.py - Mock fixes
tests/test_models_extended.py     - Test fixes
```

### 3. **CI Failures Summary**

#### a. **Linting Failures** (5 issues)
```
docker_run.py:220 - S110 try-except-pass
docker_run.py:229 - S603 subprocess security
docker_run.py:245 - S603 subprocess security  
docker_run.py:339 - S603 subprocess security
models.py:73      - UP045 Optional[str] syntax
+ Multiple E501 line length errors in tests
```

#### b. **Test Failures** (16 failures across Python 3.9-3.12)
- Mock import paths incorrect
- Exit code mismatches
- Resource check failures
- Tests for unimplemented features

#### c. **Installation Test Failures**
- CLI fails without --no-input flag in CI
- Interactive prompts block CI execution

## Why Tests Pass Locally

1. **We fixed all issues locally** but haven't committed them
2. **Local Python 3.9.6** runs our fixed code successfully
3. **All 141 tests pass** with our fixes applied
4. **Linting passes** with our security annotations

## CI Test Matrix

The CI tests on:
- Python 3.9, 3.10, 3.11, 3.12
- Ubuntu latest, macOS latest
- With Docker not available
- With strict resource limits

## Resolution Steps

1. **Commit all fixes** to the PR branch
2. **Push to update the PR**
3. **CI will then run against the fixed code**

## Verification Done Locally

```bash
# All pass with our fixes:
python3 -m ruff check src/ tests/  # ✅ All checks passed
python3 -m mypy src/swebench_runner  # ✅ Success
python3 -m pytest tests/ -v  # ✅ 141 passed, 4 skipped
```

## Key Lesson

Always check `git status` when CI fails but local tests pass. The most common cause is uncommitted fixes that exist locally but aren't in the PR branch that CI is testing.