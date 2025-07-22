# Workplan: Fix PR6 Critical Issues

## Task ID: FIX-PR6-Critical

## Problem Statement
PR #6 review identified critical blocking issues that prevent merge:
1. Bootstrap flow runs before argument validation, breaking all CLI tests
2. Docker errors return exit code 3 instead of 2 (per PRD spec)
3. Resource checks fail in CI environment (50GB required vs 27GB available)
4. Conflicting patch size limits (5MB parameter vs 500KB hardcoded)
5. Low test coverage (45% vs 90% requirement)

## Proposed Solution

### 1. Fix Bootstrap Order (CRITICAL)
- Move bootstrap logic after argument parsing in `cli.py`
- Bootstrap should only run for commands that need it (run, validate)
- Skip bootstrap for help, version, clean commands

### 2. Fix Exit Codes (CRITICAL)
- Update all Docker-related errors to use exit code 2
- Ensure consistency with `exit_codes.py` module
- Network errors should remain exit code 3

### 3. Add CI Environment Detection (CRITICAL)
- Add `SWEBENCH_SKIP_RESOURCE_CHECK` environment variable support
- Auto-detect CI environment and adjust resource requirements
- Reduce disk requirement to 20GB in CI mode

### 4. Fix Patch Size Validation
- Remove hardcoded 500KB check in `docker_run.py:347`
- Use only the configurable `max_patch_size` parameter
- Default should remain 5MB as per PRD

### 5. Improve Test Coverage
- Add tests for `bootstrap.py` module
- Add tests for `cache.py` module
- Mock all external dependencies (Docker, filesystem, network)
- Target: >90% coverage for critical paths

## Automated Test Plan
- Unit tests for bootstrap flow with proper mocking
- Unit tests for exit code scenarios
- Unit tests for CI environment detection
- Unit tests for patch size validation
- Integration test for full CLI flow (mocked Docker)

## Components Involved
- `src/swebench_runner/cli.py` - Bootstrap order
- `src/swebench_runner/docker_run.py` - Exit codes, resource checks, patch size
- `src/swebench_runner/bootstrap.py` - Test coverage needed
- `src/swebench_runner/cache.py` - Test coverage needed
- `tests/` - New test files

## Dependencies
- Existing exit_codes module
- Docker SDK for mocking
- Click testing utilities
- pytest and pytest-mock

## Implementation Checklist
- [x] Fix bootstrap order in cli.py
- [x] Add bootstrap skip for help/version/clean commands
- [x] Fix Docker exit codes (2 not 3)
- [x] Add SWEBENCH_SKIP_RESOURCE_CHECK support
- [x] Add CI environment auto-detection
- [x] Remove hardcoded 500KB patch size check (Note: 500KB is Docker env limit, not general)
- [x] Write comprehensive tests for bootstrap.py
- [x] Write comprehensive tests for cache.py
- [x] Write integration tests for CLI flow
- [ ] Run linting and fix any issues
- [ ] Verify test coverage >90%

## Verification Steps
1. Run `pytest` - all tests should pass
2. Run `pytest --cov` - coverage should be >90%
3. Run `ruff check` - no linting errors
4. Run `mypy` - no type errors
5. Test manually: `swebench --help` should not trigger bootstrap
6. Test manually: Docker errors should return exit code 2
7. Test in CI: Resource checks should be skipped with env var

## Decision Authority
- **Independent decisions**: Implementation details, test structure, mock strategies
- **User input needed**: None - requirements are clear from PR feedback

## Questions/Uncertainties
- **Blocking**: None
- **Non-blocking**: 
  - Should we auto-detect GitHub Actions, CircleCI, etc. separately? (Assumption: Use generic CI env var)
  - Should resource requirements be configurable? (Assumption: Use defaults with env override)

## Acceptable Tradeoffs
- Simplified CI detection (check common env vars) vs complex detection
- Basic test mocking vs full Docker simulation
- Focus on critical path coverage vs 100% coverage

## Status: In Progress (90% Complete)

## Notes
- Priority is unblocking CI and fixing critical errors
- New features should wait until these issues are resolved
- Following lessons from CI implementation: research first, test early

### Update 2025-07-21:
- All critical fixes implemented
- Exit codes standardized with exit_codes.py module
- Resource checks now CI-compatible with environment variables
- Test isolation implemented via conftest.py
- Bootstrap order fixed - runs after argument validation
- Patch size clarification: 500KB limit is for Docker env vars, not general limit
- Tests added for bootstrap, cache, and extended docker scenarios
- Remaining: Run linting/type checking and verify coverage