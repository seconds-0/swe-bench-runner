# PR #6 Error Summary - Quick Reference

## Critical Errors (Blocking Merge)

1. **Bootstrap Flow Breaks All Tests**
   - File: `src/swebench_runner/cli.py:67`
   - Issue: Bootstrap runs before argument validation
   - Impact: All CLI tests fail with SystemExit(0)

2. **Python Version Incompatibility**
   - Files: `pyproject.toml`, multiple source files
   - Issue: Requires Python >=3.10 but CI tests 3.8+
   - Impact: Installation fails on Python 3.8/3.9

3. **Wrong Exit Codes**
   - Files: Multiple locations
   - Issue: Docker errors return 3 instead of 2
   - Impact: Incorrect error reporting

## High Priority Errors

4. **Conflicting Patch Size Limits**
   - File: `src/swebench_runner/docker_run.py:347`
   - Issue: 5MB parameter vs 500KB hardcoded check
   - Impact: Confusing behavior

5. **CI Resource Check Failures**
   - File: `src/swebench_runner/docker_run.py:133`
   - Issue: Requires 50GB disk but CI has 27GB
   - Impact: CI always fails

6. **Low Test Coverage (45% vs 90%)**
   - Files: `bootstrap.py` (21%), `cache.py` (33%)
   - Issue: Missing tests for new modules
   - Impact: Below merge threshold

## Medium Priority Errors

7. **Linting Violations**
   - 23 style violations (long lines, imports)
   - Missing type annotations
   - Inconsistent formatting

8. **Security Warnings**
   - Subprocess output not validated
   - No input sanitization

## Fix Order

### Immediate (Unblock CI):
1. Move bootstrap after argument validation
2. Add `SWEBENCH_SKIP_RESOURCE_CHECK` env var
3. Fix exit codes (2=Docker, 3=Network)

### Next (Core Issues):
4. Decide Python version (recommend drop 3.8/3.9)
5. Remove 500KB hardcoded check
6. Write tests for bootstrap.py and cache.py

### Finally (Polish):
7. Fix linting issues
8. Add integration tests
9. Update documentation

## Key Code Locations

- Bootstrap order: `cli.py:67`
- Patch size conflict: `docker_run.py:347`
- Resource check: `docker_run.py:133`
- Exit codes: Search for `sys.exit()`
- Python version: `pyproject.toml`

## Environment Variables Needed

- `SWEBENCH_SKIP_RESOURCE_CHECK=true` (for CI)
- `CI=true` (detect CI environment)

---

Use this summary for quick reference when fixing issues. Full details in PR6-MVP-DockerRun-Errors.md