# Phase 2.1 Test Doubles Migration - Completion Script

## Summary of Completed Work

### ✅ Infrastructure Complete (Phase 1)
1. **Extended DockerClientDouble** - Added scenarios for OOM, storage full, and container limits during container operations
2. **Created PatchValidatorDouble** - Complete class for all patch validation scenarios
3. **Extended NetworkDouble** - Added GitHub rate limit, HuggingFace rate limit, and GHCR blocked scenarios
4. **Added injection methods** - Test harness now supports all double types

### ✅ Tests Migrated (10 of 28 tests - 36%)
1. **Docker tests (6 of 6)** - All Docker error tests migrated
   - test_error_2_docker_not_running ✅
   - test_error_10_docker_permission_denied ✅
   - test_error_11_docker_desktop_stopped ✅
   - test_error_13_oom_during_test ✅
   - test_error_19_docker_storage_full ✅
   - test_error_30_docker_container_limit ✅

2. **Network tests (4 of 4)** - All network error tests migrated
   - test_error_3_network_failure ✅
   - test_error_15_ghcr_blocked ✅
   - test_error_16_git_rate_limit ✅
   - test_error_27_hf_rate_limit ✅

## Remaining Migration Tasks

### Phase 4: Patch Validation Tests (7 tests)
These tests need to be migrated to use `inject_patch_double()`:

```python
# Pattern for each test:
# 1. Replace env var with: patch_double = harness.inject_patch_double(scenario="...")
# 2. Change harness.run_cli() to harness.run_cli_direct()
# 3. Remove all SWEBENCH_MOCK_* env vars
# 4. Add verification: assert patch_double.patches_validated or similar

test_error_5_invalid_patch_schema -> inject_patch_double("invalid_schema")
test_error_6_patch_too_large_for_env -> inject_patch_double("too_large_env")
test_error_14_patch_conflict -> inject_patch_double("conflict")
test_error_23_patch_encoding_error -> inject_patch_double("encoding_error")
test_error_24_patch_too_large -> inject_patch_double("too_large")
test_error_25_binary_patch -> inject_patch_double("binary")
test_error_26_patch_apply_failed -> inject_patch_double("apply_failed")
```

### Phase 5: Remaining Tests (11 tests)

#### Resource Tests (1 test)
- test_error_4_insufficient_disk_space -> inject_filesystem_double("disk_full")

#### Architecture/System Tests (3 tests)
- test_error_7_unsupported_arch -> Mock platform detection
- test_error_20_invalid_python -> Mock Python version check
- test_error_17_corrupted_cache -> inject_filesystem_double("cache_corrupted")

#### Image/Container Tests (2 tests)
- test_error_18_stale_image -> Need to extend DockerClientDouble
- test_error_21_container_timeout -> Need timeout scenario in DockerClientDouble

#### Instance Tests (3 tests)
- test_error_22_invalid_instance_id -> Use validation double
- test_error_28_instance_timeout -> Need timeout scenario
- test_error_29_flaky_test_detected -> Need flaky test scenario

#### Meta Tests (2 tests)
- test_error_includes_suggested_action -> Already ready for doubles
- test_error_includes_documentation_link -> Already ready for doubles

## Migration Script

To complete the migration efficiently, use this script pattern for each test:

```python
# Before (using env vars):
def test_error_X_description(self):
    with SWEBenchTestHarness() as harness:
        env = {
            "SWEBENCH_MOCK_SOMETHING": "value",
            "SWEBENCH_MOCK_NO_DOCKER": "true"
        }
        returncode, stdout, stderr = harness.run_cli(
            ["run", "--patches", str(patch_file)],
            env=env
        )

# After (using test doubles):
def test_error_X_description(self):
    with SWEBenchTestHarness() as harness:
        # Inject appropriate double
        double = harness.inject_XXX_double(scenario="scenario_name")

        returncode, stdout, stderr = harness.run_cli_direct(
            ["run", "--patches", str(patch_file)]
        )

        # Add verification of double interaction
        assert double.some_action_called, "Double should have been called"
```

## Environment Variables to Remove

After migration, search and remove all occurrences of:
- SWEBENCH_MOCK_NO_DOCKER
- SWEBENCH_MOCK_PATCH_CONFLICT
- SWEBENCH_MOCK_PATCH_FAIL
- SWEBENCH_MOCK_PYTHON_VERSION
- SWEBENCH_MOCK_TIMEOUT
- SWEBENCH_MOCK_INSTANCE_TIMEOUT
- SWEBENCH_MOCK_FLAKY
- SWEBENCH_PLATFORM (when used for mocking)

## Validation Steps

1. Run each migrated test individually:
   ```bash
   python3 -m pytest tests/e2e/test_error_handling.py::TestClassName::test_name -xvs
   ```

2. Run all error handling tests:
   ```bash
   python3 -m pytest tests/e2e/test_error_handling.py -xvs
   ```

3. Check for any remaining mock env vars:
   ```bash
   grep -r "SWEBENCH_MOCK" tests/e2e/test_error_handling.py
   ```

4. Verify test execution time improvement:
   ```bash
   time python3 -m pytest tests/e2e/test_error_handling.py
   ```

## Expected Benefits After Completion

1. **Cleaner tests** - No environment variable pollution
2. **Better debugging** - Direct execution with clear stack traces
3. **Type safety** - IDE understands test doubles
4. **Faster execution** - ~20-30% speed improvement
5. **Maintainability** - Easier to add new test scenarios

## Next Steps

1. Complete remaining 18 test migrations (Phase 4-5)
2. Remove all SWEBENCH_MOCK_* environment variables (Phase 6)
3. Run full test suite to validate (Phase 6)
4. Update documentation with migration guide
5. Consider adding more scenarios to test doubles as needed
