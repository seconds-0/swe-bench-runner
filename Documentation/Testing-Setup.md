# Local Testing Setup for SWE-Bench Runner

## Problem Statement

The codebase requires Python 3.10+ due to usage of modern Python features (e.g., union type syntax `str | None`), but local development environments may have older Python versions. This creates a barrier for running tests locally.

## Current Testing Issues

### 1. Python Version Mismatch
- **Issue**: Code uses Python 3.10+ syntax (`str | None`)
- **Local Environment**: Often has Python 3.9 or older
- **Impact**: Tests cannot run locally without proper Python version

### 2. Test Hanging in CI
- **Issue**: `test_cli_generation_simple.py::test_generate_command_basic` hangs in CI
- **Root Cause**: Complex interaction between async code, Click's CliRunner, and mocking
- **Temporary Fix**: Test is skipped with `pytest.skip()` to unblock CI

## Solutions Implemented

### 1. Docker-Based Test Runners

Three test scripts have been created in the `scripts/` directory:

#### a. `test-docker.sh` - Full test runner with image building
```bash
./scripts/test-docker.sh                    # Run all tests
./scripts/test-docker.sh tests/test_cli.py  # Run specific test file
PYTHON_VERSION=3.10 ./scripts/test-docker.sh # Use specific Python version
```

#### b. `test-quick.sh` - Fast test runner without image building
```bash
./scripts/test-quick.sh                     # Run all tests quickly
./scripts/test-quick.sh tests/              # Run all tests in directory
```

#### c. `test-single.sh` - Single file test runner
```bash
./scripts/test-single.sh tests/test_cli.py  # Test single file
```

### 2. Environment Setup Options

#### Option 1: Docker (Recommended)
- No local Python installation required
- Consistent with CI environment
- Use the scripts above

#### Option 2: pyenv (For persistent local development)
```bash
# Install pyenv (macOS)
brew install pyenv

# Install Python 3.11
pyenv install 3.11.0
pyenv local 3.11.0

# Create virtual environment
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

#### Option 3: Use system Python 3.10+
```bash
# Check if you have Python 3.10+ installed
python3.10 --version || python3.11 --version

# Use it directly
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Running Tests Locally

### Quick Test Run
```bash
# Test everything with Docker (no setup required)
./scripts/test-quick.sh

# Test specific file
./scripts/test-single.sh tests/test_cli.py

# Test with specific Python version
PYTHON_VERSION=3.12 ./scripts/test-quick.sh
```

### With Local Python Environment
```bash
# Activate virtual environment
source .venv/bin/activate

# Run all tests
pytest

# Run specific test
pytest tests/test_cli.py -v

# Run with coverage
pytest --cov=swebench_runner --cov-report=html
```

## Known Issues

### 1. Hanging Test
- **File**: `tests/test_cli_generation_simple.py`
- **Test**: `test_generate_command_basic`
- **Status**: Currently skipped
- **Issue**: Complex async/sync mocking interaction with Click's CliRunner
- **TODO**: Investigate proper mocking strategy for async CLI commands

### 2. Slow Docker First Run
- **Issue**: First run downloads Python Docker image (~50MB)
- **Solution**: Use `test-quick.sh` after first run, image will be cached

## CI/Local Parity

To ensure your local tests match CI:
1. Use Python 3.11 (same as CI lint/primary tests)
2. Set CI environment variables:
   ```bash
   export CI=true
   export SWEBENCH_CI_MIN_DISK_GB=15
   export SWEBENCH_CI_MIN_MEMORY_GB=4
   ```
3. Run with same pytest options as CI:
   ```bash
   pytest tests/ -v --cov=swebench_runner --cov-report=xml
   ```

## Recommendations

1. **For Quick Fixes**: Use Docker test scripts - no setup required
2. **For Active Development**: Install Python 3.11 locally with pyenv
3. **For CI Debugging**: Use `test-docker.sh` to replicate CI environment exactly
4. **Before Pushing**: Always run `./scripts/test-quick.sh` to catch issues early

## Next Steps

1. Fix the hanging test in `test_cli_generation_simple.py`
2. Add pre-commit hook to run quick tests automatically
3. Consider adding a devcontainer configuration for VS Code users
4. Add make targets for common test operations