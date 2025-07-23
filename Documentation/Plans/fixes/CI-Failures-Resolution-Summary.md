# CI Failures Resolution Summary

## Initial Issues Found

### 1. Environmental Differences
- **Root Cause**: CI was testing committed code in PR, local tests were running with uncommitted fixes
- **Solution**: Committed all local fixes to PR branch

### 2. After First Commit - New CI Failures

#### a. Mypy Type Checking Errors (6 errors)
```
docker_run.py:13: error: Library stubs not installed for "docker"
docker_run.py:14: error: Library stubs not installed for "docker.errors"
docker_run.py:175: error: Library stubs not installed for "psutil"
cli.py:253: error: Library stubs not installed for "docker"
```

**Root Cause**: CI environment doesn't have type stubs for `docker` and `psutil` packages

**Solution**: Created `mypy.ini` configuration file to ignore missing imports:
```ini
[mypy-docker.*]
ignore_missing_imports = True

[mypy-psutil.*]
ignore_missing_imports = True
```

#### b. Test Failures (2 tests)
- `test_ci_mode_custom_memory_requirement`
- `test_psutil_not_available`

**Root Cause**: Tests were trying to mock `psutil` module, but `psutil` is not installed in CI (it's an optional dependency used in try/except blocks)

**Solution**: Fixed mocking strategy to mock at sys.modules level:
```python
# Instead of @patch("psutil.virtual_memory")
# Use:
mock_psutil = Mock()
mock_psutil.virtual_memory.return_value = Mock(available=1.5 * 1024**3)
with patch.dict('sys.modules', {'psutil': mock_psutil}):
    docker_run.check_resources()
```

## Key Lessons Learned

1. **Always check git status** when CI fails but local tests pass
2. **Optional dependencies** need special handling in tests - mock at import level
3. **Type stubs** for third-party packages may not be available in CI
4. **CI environment** is minimal - only required dependencies are installed

## Final Status

All fixes have been committed and pushed:
- ✅ Mypy configuration added
- ✅ Test mocking fixed for optional dependencies
- ✅ Unnecessary type: ignore comments removed
- ✅ All tests pass locally
- ✅ Mypy passes locally

The CI should now pass all checks.
