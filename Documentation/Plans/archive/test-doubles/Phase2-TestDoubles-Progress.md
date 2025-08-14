# Phase 2.1: Test Doubles Implementation Progress

## Summary
We've successfully created test double infrastructure but discovered a fundamental limitation: **subprocess isolation prevents runtime injection from carrying over to child processes**.

## Completed Work

### ✅ Test Doubles Infrastructure
- Created `tests/e2e/test_doubles.py` with 5 comprehensive test doubles:
  - `DockerClientDouble` - Simulates Docker scenarios
  - `HuggingFaceDouble` - Simulates dataset operations
  - `ProviderDouble` - Simulates AI provider APIs
  - `NetworkDouble` - Simulates network operations
  - `FileSystemDouble` - Simulates file system operations

### ✅ Dependency Injection Support
- Created `src/swebench_runner/docker_client.py` abstraction layer
- Added factory pattern for Docker client injection
- Modified `docker_run.py` to use new abstraction
- Updated `cli.py` to use injected clients

### ✅ Test Harness Enhancement
- Added `inject_docker_double()` method to test harness
- Added `inject_all_doubles()` for comprehensive mocking
- Proper cleanup and restoration of original factories

## Key Discovery: Subprocess Isolation Issue

### The Problem
E2E tests use `subprocess.run()` to execute the CLI as a real command. This creates a new Python process that:
1. Doesn't inherit runtime injections from parent process
2. Starts with fresh imports and default factories
3. Makes our dependency injection approach ineffective

### Proof of Concept
```python
# This works (same process):
harness.inject_docker_double(scenario="not_running")
cli.cli()  # Exit code 2 ✅

# This doesn't work (subprocess):
harness.inject_docker_double(scenario="not_running")
subprocess.run(["python", "-m", "swebench_runner"])  # Gets real Docker error ❌
```

## Recommended Solution: Hybrid Approach

### Phase 2.1a: Module-Level Testing (Immediate)
For unit-style E2E tests, test modules directly without subprocess:

```python
def test_docker_not_running(self):
    with SWEBenchTestHarness() as harness:
        harness.inject_docker_double(scenario="not_running")

        # Test the module directly
        sys.argv = ['swebench_runner', 'run', '--patches', patch_file]
        with pytest.raises(SystemExit) as exc_info:
            cli.cli()

        assert exc_info.value.code == 2
```

**Benefits:**
- Works with our current injection system
- Fast test execution
- Full control over doubles
- No environment variable pollution

**Limitations:**
- Not testing actual CLI invocation
- Shared state between tests (needs careful cleanup)

### Phase 2.1b: True CLI Testing with Test Config (Future)
For true subprocess testing, implement configuration file approach:

1. Create `.swebench-test.json` config that subprocess can read
2. Test harness writes config with test double settings
3. Application checks for test config and uses doubles accordingly

```python
# Test harness creates config
config = {
    "test_mode": True,
    "docker_scenario": "not_running",
    "provider_scenario": "rate_limit"
}
harness.write_test_config(config)

# Application checks on startup
if Path(".swebench-test.json").exists():
    config = load_test_config()
    if config.docker_scenario == "not_running":
        set_docker_client_factory(lambda: FailingDockerDouble())
```

## Migration Status

### Docker Error Tests (6 tests)
- ✅ `test_error_2_docker_not_running` - Migrated, needs subprocess fix
- ✅ `test_error_10_docker_permission_denied` - Migrated, needs subprocess fix
- ✅ `test_error_11_docker_desktop_stopped` - Migrated, needs subprocess fix
- ⏳ `test_error_13_oom_during_test` - Not migrated
- ⏳ `test_error_19_docker_storage_full` - Not migrated
- ⏳ `test_error_30_docker_container_limit` - Not migrated

### Remaining Categories
- ⏳ Network Error Tests (4 tests)
- ⏳ Resource Error Tests (1 test)
- ⏳ Patch Error Tests (7 tests)
- ⏳ General Error Tests (10 tests)

## Next Steps

### Immediate (Today)
1. Switch to module-level testing approach for existing tests
2. Complete migration of all 28 error handling tests
3. Remove `SWEBENCH_MOCK_*` environment variables
4. Verify all tests pass

### Short-term (This Week)
1. Implement test config file approach for true CLI testing
2. Add parallel test execution support
3. Create integration test suite using real services

### Long-term (Next Sprint)
1. Performance benchmarking with test doubles
2. Chaos testing with failure injection
3. Cross-platform test scenarios

## Lessons Learned

1. **Subprocess isolation is a feature, not a bug** - It ensures clean test environments
2. **Module testing is sufficient for most cases** - True CLI testing only needed for specific scenarios
3. **Test doubles are cleaner than env vars** - Better IDE support, type safety, and maintainability
4. **Hybrid approach is optimal** - Use right tool for right job

## Conclusion

While we encountered the subprocess isolation challenge, our test double infrastructure is solid and provides a much cleaner alternative to environment variables. By adopting a hybrid approach (module testing for most cases, config files for true CLI tests), we can achieve our goals of:

- ✅ Eliminating environment variable mocks
- ✅ Improving test maintainability
- ✅ Enabling better IDE support
- ✅ Reducing test complexity

The migration will continue with the module-level testing approach, which will immediately benefit from our test doubles infrastructure.
