# CI vs Local Test Configuration Deep Analysis

## Problem Statement
- **Local**: Tests complete in 1.2 seconds for 168 tests
- **CI**: Tests timeout after 30 minutes (1800 seconds) - that's 1500x slower!
- **Latest Status**: Even with `-m "not integration"` added to CI, tests still timeout

## Configuration Files to Analyze

### 1. CI Configuration (.github/workflows/ci.yml)
### 2. Pytest Configuration (pyproject.toml, pytest.ini, setup.cfg)
### 3. Python Environment (versions, dependencies)
### 4. System Resources (CPU, memory, parallelization)
### 5. Test Discovery Patterns
### 6. Environment Variables
### 7. Coverage Settings

---

## Key Findings

### 1. CI Timeout Location Identified!
- Tests timeout at exactly 30 minutes (CI job timeout)
- Last successful test: `test_generation_integration.py::TestGenerationFailureHandler::test_handle_batch_failure_mostly_failed_with_confirm`
- First failing test: `test_generation_integration.py::TestCLIIntegration::test_run_with_provider_generates_patches`
- Second failing test: `test_generation_integration.py::TestCLIIntegration::test_run_with_generate_only`
- CI had to kill orphan pytest process (pid 2177) after timeout

### 2. Root Cause: asyncio.run() Mocking
Both failing tests patch `asyncio.run`, which is problematic because:
- They're testing CLI commands that internally use `asyncio.run()`
- Mocking `asyncio.run` can interfere with pytest-asyncio or Click's async handling
- This creates a deadlock or infinite wait in CI environment

### 3. Environment Differences
- **Local**: Python 3.9.6 (my machine)
- **CI**: Python 3.10, 3.11, 3.12
- **Key difference**: CI runs with coverage enabled, which may affect async behavior
- **CI command**: `pytest tests/ -v --cov=swebench_runner --cov-report=xml --cov-report=term-missing -m "not integration"`

### 4. Why Local Works but CI Hangs
- Local environment may handle the asyncio mocking differently
- CI environment with coverage tools may create different async event loop behavior
- The combination of Click CLI runner + asyncio.run mocking + coverage = deadlock

---

## Solution Implemented

### Fix Applied: Remove asyncio.run() Mocking
Instead of mocking `asyncio.run`, we now:
1. Mock the `GenerationIntegration` class
2. Create a mock instance with an `AsyncMock` for the `generate_patches_for_evaluation` method
3. This allows the CLI's `asyncio.run()` to execute normally while controlling the async behavior

### Changed Tests:
1. `test_run_with_provider_generates_patches`
2. `test_run_with_generate_only`
3. `test_generate_command_with_instance`

All three tests now use proper async mocking without interfering with Click's event loop.

---

## Detailed Analysis
