# End-to-End Testing Guide

This directory contains comprehensive end-to-end tests that validate the complete SWE-bench Runner workflow with real API calls and Docker evaluation.

## ⚠️ Important: These Tests Cost Money!

These tests make real API calls to OpenAI, Anthropic, or other providers and will incur charges on your account. Expected costs:
- **Per test run**: $0.01 - $0.05
- **Full suite**: $0.05 - $0.10

## Prerequisites

1. **Python 3.11+** (recommended, though tests may work with 3.9+)
2. **Docker** installed and running
3. **API Keys** for at least one provider:
   - OpenAI: `OPENAI_API_KEY`
   - Anthropic: `ANTHROPIC_API_KEY`
4. **10GB+ free disk space** for Docker operations
5. **SWE-bench CLI installed**: `pip install -e .`

## Quick Start

### 1. Set Up Environment

```bash
# Copy the example config
cp .env.e2e.example .env.e2e

# Edit .env.e2e and add your API keys
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...

# Load environment
source .env.e2e
```

### 2. Run Tests

```bash
# Dry run (check setup without spending money)
./scripts/run_e2e_test.sh --dry-run

# Run minimal test (cheapest option ~$0.01)
./scripts/run_e2e_test.sh --minimal

# Run specific scenario
./scripts/run_e2e_test.sh --scenario 1  # Happy path test

# Run full suite (costs ~$0.05-0.10)
./scripts/run_e2e_test.sh --no-confirm

# Run with pytest directly (more control)
SWEBENCH_E2E_ENABLED=true pytest tests/e2e/test_real_api_workflow.py -v -s
```

## Test Scenarios

### Scenario 1: Happy Path (Minimal Cost)
- Uses cheapest model (gpt-3.5-turbo or claude-3-haiku)
- Generates patch for 1 instance
- Runs evaluation
- Verifies results
- **Cost**: ~$0.002

### Scenario 2: Multi-Provider Comparison
- Tests multiple providers if available
- Generates patches with each
- Compares results
- **Cost**: ~$0.004

### Scenario 3: Error Recovery
- Tests invalid API key handling
- Tests missing file handling
- Tests invalid dataset handling
- **Cost**: ~$0.001 (mostly error paths)

### Scenario 4: Full Pipeline
- Complete workflow from setup to results
- Lists providers
- Generates patches for 2 instances
- Runs full evaluation
- Generates HTML reports
- **Cost**: ~$0.01

## Configuration

### Environment Variables

```bash
# Required
SWEBENCH_E2E_ENABLED=true      # Safety switch to enable tests
OPENAI_API_KEY=sk-...          # At least one API key required
ANTHROPIC_API_KEY=sk-ant-...

# Optional
SWEBENCH_E2E_MAX_COST=0.10     # Maximum cost limit (default: $0.10)
SWEBENCH_E2E_LOG_DIR=./e2e_logs # Log directory
OPENAI_TEST_MODEL=gpt-3.5-turbo # Model to use for tests
ANTHROPIC_TEST_MODEL=claude-3-haiku-20240307
```

## Test Output

Tests generate several artifacts:

1. **Console Output**: Real-time progress with color-coded status
2. **Log Files**: `e2e_logs/e2e_test_YYYYMMDD_HHMMSS.log`
3. **JSON Reports**: `e2e_test_YYYYMMDD_HHMMSS.json` with detailed metrics
4. **HTML Reports**: Generated evaluation reports in temp directory

### Example Output

```
[2025-08-18 10:45:23] → Starting E2E test run
[2025-08-18 10:45:24] ✓ Docker running (version 28.2.2)
[2025-08-18 10:45:25] ✓ API key found for OpenAI
[2025-08-18 10:45:26] → Running: swebench generate --dataset lite --count 1
[2025-08-18 10:45:35] ✓ Patch generated (cost: $0.002)
[2025-08-18 10:45:36] → Running: swebench run --patches test_patches.jsonl
[2025-08-18 10:46:45] ✓ Evaluation completed
[2025-08-18 10:46:46] → Results: 1 passed, 0 failed
[2025-08-18 10:46:47] → Total cost: $0.002
[2025-08-18 10:46:48] ✓ Test PASSED
```

## CI/CD Integration

### GitHub Actions

The repository includes a GitHub Actions workflow (`.github/workflows/e2e-tests.yml`) that:
- Runs on manual trigger (workflow_dispatch)
- Accepts cost limit and scenario inputs
- Uses repository secrets for API keys
- Uploads test artifacts
- Comments results on PRs

To enable:
1. Add secrets to repository: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`
2. Trigger manually from Actions tab
3. Or uncomment schedule for weekly runs

### Running in CI

```yaml
# Example GitHub Actions step
- name: Run E2E Tests
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    SWEBENCH_E2E_ENABLED: true
    SWEBENCH_E2E_MAX_COST: 0.10
  run: |
    ./scripts/run_e2e_test.sh --no-confirm --scenario 1
```

## Cost Management

### Budget Controls

1. **Per-test limit**: Each test respects `SWEBENCH_E2E_MAX_COST`
2. **Model selection**: Tests use cheapest models by default
3. **Early termination**: Tests stop if budget exceeded
4. **Cost tracking**: Every API call is logged with estimated cost

### Cost Optimization Tips

- Use `--scenario 1` for quick validation (~$0.002)
- Set `OPENAI_TEST_MODEL=gpt-3.5-turbo` for cheaper OpenAI tests
- Use `--minimal` flag for basic connectivity test
- Run `--dry-run` first to verify setup without cost

## Troubleshooting

### Common Issues

#### "No API keys found"
```bash
export OPENAI_API_KEY=sk-...
# or
export ANTHROPIC_API_KEY=sk-ant-...
```

#### "Docker daemon is not running"
```bash
# macOS
open -a Docker

# Linux
sudo systemctl start docker
```

#### "E2E tests disabled"
```bash
export SWEBENCH_E2E_ENABLED=true
```

#### "Command not found: swebench"
```bash
# Install the CLI
pip install -e .
```

### Debug Mode

For detailed debugging:
```bash
# Run with pytest debugging
SWEBENCH_E2E_ENABLED=true pytest tests/e2e/test_real_api_workflow.py -vvs --tb=long

# Check logs
tail -f e2e_logs/e2e_test_*.log

# Run single test function
pytest tests/e2e/test_real_api_workflow.py::TestRealAPIWorkflow::test_scenario_1_happy_path -v
```

## Writing New E2E Tests

To add new E2E tests:

1. Add test method to `TestRealAPIWorkflow` class
2. Use the logger for all output: `self.logger.log("message", "INFO", cost=0.001)`
3. Track costs for API calls
4. Use `self.run_command()` for CLI calls
5. Add appropriate assertions
6. Document expected cost in docstring

Example:
```python
def test_my_scenario(self):
    """Test description. Cost: ~$0.01"""
    self.logger.log("Starting my test", "INFO")

    # Run command
    code, stdout, stderr = self.run_command(
        ["swebench", "command", "--flag"],
        timeout=60
    )

    # Track cost
    self.logger.log("API call made", "SUCCESS", cost=0.005)

    # Assertions
    assert code == 0
    assert "expected" in stdout
```

## Safety Features

1. **Confirmation prompt**: Tests ask for confirmation before running
2. **Cost limits**: Hard limits prevent runaway costs
3. **Timeouts**: All operations have timeouts
4. **Dry run mode**: Test setup without making API calls
5. **Detailed logging**: Every action is logged with timestamp
6. **Environment safety**: `SWEBENCH_E2E_ENABLED` must be explicitly set

## Contact

For issues or questions about E2E tests:
- Create an issue on GitHub
- Check test logs in `e2e_logs/`
- Review the JSON report for detailed step information
