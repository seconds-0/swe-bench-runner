# CI and Pre-commit Hook Parity

This document ensures our pre-commit hooks remain in sync with CI checks, preventing drift and surprises.

## Mapping: CI Jobs to Pre-commit Hooks

| CI Job/Check | Pre-commit Hook | When Runs | Auto-fix? |
|--------------|-----------------|-----------|-----------|
| **Linting** (`ruff check`) | `ruff` | Every commit | ✅ Yes |
| **Type checking** (`mypy`) | `mypy` | Every commit | ❌ No |
| **Tests** (`pytest`) | `pytest` (local) | On push only | ❌ No |
| **Coverage** (85% minimum) | Part of `pytest` hook | On push only | ❌ No |
| **Commit format** | `conventional-pre-commit` | Commit message | ❌ No |
| **File size** (<1MB) | `check-added-large-files` | Every commit | ❌ No |
| **No generated files** | `no-generated-files` (local) | Every commit | ❌ No |
| **Security audit** | `pip-audit` (local) | On push only | ❌ No |
| **Python syntax** | `check-ast` | Every commit | ❌ No |
| **Whitespace** | `trailing-whitespace` | Every commit | ✅ Yes |
| **EOF newline** | `end-of-file-fixer` | Every commit | ✅ Yes |

## Maintenance Checklist

When updating CI workflows, ensure pre-commit hooks are updated:

- [ ] If adding a new check to CI, add corresponding pre-commit hook
- [ ] If changing CI command arguments, update pre-commit hook args
- [ ] If changing versions in CI, update pre-commit hook revisions
- [ ] Run `pre-commit autoupdate` monthly to keep hooks current

## Quick Commands

```bash
# Install pre-commit hooks (one-time setup)
pre-commit install --install-hooks

# Run all hooks manually
pre-commit run --all-files

# Run specific hook
pre-commit run ruff --all-files

# Update hook versions
pre-commit autoupdate

# Run CI checks without pre-commit
./scripts/check.sh

# Run FULL CI simulation before PR (includes missing checks)
make pre-pr  # or make ci-full
```

## Three Levels of Validation

1. **Pre-commit hooks** (automatic) - Fast checks on every commit/push
2. **`make check`** (manual) - Quick CI checks before pushing  
3. **`make pre-pr`** (manual) - FULL CI simulation before opening PR

The `pre-pr` script fills gaps that pre-commit doesn't cover:
- Package build and wheel validation
- Installation testing
- Multiple Python version testing (if available)
- Platform difference warnings

## Testing Without Docker

To simulate the CI environment where Docker is not available:

```bash
# Run tests as if Docker is not installed
./scripts/test-no-docker.sh

# Or manually with environment variables:
export SWEBENCH_MOCK_NO_DOCKER=true
export SWEBENCH_SKIP_DOCKER_TESTS=true
pytest tests/
```

This sets environment variables that:
- Mock Docker as unavailable for the application
- Skip Docker-dependent tests
- Use CI resource limits

## Why This Matters

In PR #6, we had multiple CI failures that could have been caught locally:
- Whitespace issues (now auto-fixed by pre-commit)
- Type checking failures (now caught before commit)
- Test failures in CI environment (now tested locally)

By maintaining **exact parity** between pre-commit and CI, we ensure:
1. No surprises when pushing code
2. Faster development cycles
3. Cleaner git history (no "fix linting" commits)
4. Consistent code quality

## Adding New Checks

When adding a new check:

1. First add to `.github/workflows/ci.yml` or `pr-checks.yml`
2. Then add equivalent hook to `.pre-commit-config.yaml`
3. Update `scripts/check.sh` to include the check
4. Document the mapping in this file

## Hook Performance

Some hooks run at different stages for performance:
- **commit stage**: Fast checks (linting, file size)
- **push stage**: Slower checks (tests, security audit)

This balance ensures commits are fast while pushes are safe.