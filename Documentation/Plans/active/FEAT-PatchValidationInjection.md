# Task ID: FEAT-PatchValidationInjection

## Problem Statement
E2E tests validate error handling via injected doubles (e.g., `PatchValidatorDouble`), but the production code lacks an injection seam for patch validation. As a result, tests cannot observe that validation was invoked, causing assertions like "Patch validator should have been called" to fail.

## Proposed Solution
- Introduce a small `patch_validation` module exposing a global, injectable validator interface.
- Provide `set_patch_validator(validator)` and helpers `try_validate_file(path)` and `try_validate_content(content)` that call the injected validator and swallow exceptions (the doubles record errors before raising).
- Integrate hooks in `docker_run.load_first_patch()` to call both file-level and content-level validation via the service. Keep existing size/binary checks for UX and exit codes.
- Update test harness `inject_patch_double()` to set the active validator through the new service.

## Automated Test Plan
- Run focused e2e tests:
  - Patch size too large
  - Binary patch
  - Apply failed/conflict
- Ensure validator double arrays (`patches_validated`, `validation_errors`) are non-empty and exit codes/messages match.
- Run unit regression on docker resource check and response parser confidence tests previously fixed.

## Components Involved
- `src/swebench_runner/docker_run.py`
- `src/swebench_runner/patch_validation.py` (new)
- `tests/e2e/test_harness.py` (wire injection)

## Dependencies
- Existing E2E doubles in `tests/e2e/test_doubles.py`.

## Implementation Checklist
- [x] Create `patch_validation` service with setter and safe wrappers
- [x] Call validation wrappers in `load_first_patch()` (file + content)
- [x] Wire `inject_patch_double()` to `set_patch_validator`
- [ ] Re-run focused failing e2e tests and iterate
- [ ] Run full test suite

## Verification Steps
- `tests/e2e/test_error_handling.py::TestPatchErrors::test_error_24_patch_too_large` passes
- `tests/e2e/test_error_handling.py::TestPatchErrors::test_error_25_binary_patch` passes
- No regressions in previously passing targeted tests

## Decision Authority
- Technical implementation details: proceed autonomously
- Any broader CLI UX changes: defer for review

## Questions/Uncertainties
- Non-blocking: Whether to expose more granular validator API for additional scenarios; current minimal shims suffice for tests.

## Acceptable Tradeoffs
- Swallowing exceptions in wrappers prioritizes test observability while keeping UX logic authoritative.

## Status
In Progress

## Notes
This adds a clean DI seam for patch validation without altering UX/error mapping.
