# Integration Test Execution Guide

This guide provides step-by-step instructions for safely running integration tests with real APIs.

## 🎯 Quick Start

```bash
# 1. Copy and configure credentials
cp .env.integration.template .env.integration
# Edit .env.integration with your API keys

# 2. Run tests with cost controls
python scripts/run_integration_tests.py

# 3. Or test specific provider
python scripts/run_integration_tests.py --provider ollama
```

## 📋 Pre-Execution Checklist

### 1. Environment Setup

- [ ] Copy `.env.integration.template` to `.env.integration`
- [ ] Add OpenAI API key (get from https://platform.openai.com/api-keys)
- [ ] Add Anthropic API key (get from https://console.anthropic.com/settings/keys)
- [ ] Verify both keys have active credits/subscriptions

### 2. Cost Preparation

- [ ] Review estimated costs:
  - OpenAI: ~$0.01 for full test suite
  - Anthropic: ~$0.008 for full test suite
  - Ollama: Free (local execution)
  - **Total: < $0.02**
- [ ] Set spending alerts in provider dashboards
- [ ] Have provider dashboards open during testing

### 3. Ollama Setup (for local testing)

```bash
# Run the setup script
./scripts/setup_ollama.sh

# Or manually:
curl -fsSL https://ollama.com/install.sh | sh
ollama serve  # In separate terminal
ollama pull llama3.2:1b
```

## 🚀 Execution Steps

### Phase 1: Start with Ollama (Free)

```bash
# Test Ollama first since it's free
python scripts/run_integration_tests.py --provider ollama

# Or run directly with pytest
pytest tests/integration/test_ollama_integration.py -m integration -v
```

Expected outcome:
- All 16 tests should pass
- No costs incurred
- ~30 seconds execution time

### Phase 2: Test Single API Call

Before running full suites, verify credentials work:

```bash
# Test one OpenAI call
python scripts/run_integration_tests.py \
  --provider openai \
  --single-test test_basic_generation

# Test one Anthropic call
python scripts/run_integration_tests.py \
  --provider anthropic \
  --single-test test_basic_generation
```

Cost: < $0.0001 per test

### Phase 3: Run Full Test Suites

```bash
# Run all tests with cost tracking
python scripts/run_integration_tests.py --provider all

# Or run individually:
python scripts/run_integration_tests.py --provider openai
python scripts/run_integration_tests.py --provider anthropic
```

### Phase 4: Review Results

Check the generated report:
```bash
cat integration_test_costs.json
```

## 🛡️ Safety Features

### Cost Controls

The test runner enforces:
- Maximum $0.001 per test
- Maximum $0.05 total budget
- Automatic stop on budget exceed

### Credential Security

- Never commit `.env.integration`
- File is git-ignored
- Keys only loaded at runtime
- No keys in logs or output

### Error Handling

Tests handle:
- Missing credentials gracefully
- API failures with clear errors
- Network issues with retries
- Rate limits with backoff

## 📊 Monitoring During Execution

### Real-Time Monitoring

1. **Terminal Output**: Watch for color-coded results
   - 🟢 Green: Test passed
   - 🟡 Yellow: Warning/skip
   - 🔴 Red: Test failed

2. **Provider Dashboards**:
   - OpenAI: https://platform.openai.com/usage
   - Anthropic: https://console.anthropic.com/usage

3. **Cost Tracking**:
   - Live updates in terminal
   - Final report in JSON

### What to Watch For

- Unexpected rate limit errors
- Higher than expected costs
- Authentication failures
- Model availability issues

## 🔧 Troubleshooting

### OpenAI Issues

```bash
# Verify API key
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY" | jq .

# Common issues:
# - Expired key: Generate new at platform.openai.com
# - No credits: Add payment method
# - Rate limit: Wait and retry
```

### Anthropic Issues

```bash
# Verify API key format
echo $ANTHROPIC_API_KEY | grep -q "^sk-ant-" && echo "Valid format" || echo "Invalid format"

# Common issues:
# - Wrong key format: Should start with sk-ant-
# - Region restrictions: Check account settings
# - Model access: Verify model availability
```

### Ollama Issues

```bash
# Check if running
curl http://localhost:11434/api/tags

# Check models
ollama list

# Common issues:
# - Service not running: ollama serve
# - Model not pulled: ollama pull llama3.2:1b
# - Port conflict: Check port 11434
```

## 📈 Post-Execution Analysis

### Review Test Results

1. **Check Summary**:
   ```bash
   # View results summary
   cat integration_test_costs.json | jq '.results[] | {provider, success, cost}'
   ```

2. **Analyze Failures**:
   ```bash
   # Extract failed tests
   cat integration_test_costs.json | jq '.results[] | select(.success==false)'
   ```

3. **Calculate Total Costs**:
   ```bash
   # Sum all costs
   cat integration_test_costs.json | jq '.total_cost'
   ```

### Document Issues

If tests fail, create an issue report:

```markdown
## Integration Test Failure Report

**Date**: [timestamp]
**Provider**: [provider name]
**Test**: [test name]
**Error**: [error message]

### Steps to Reproduce
1. [step 1]
2. [step 2]

### Expected vs Actual
- Expected: [what should happen]
- Actual: [what happened]

### Potential Fixes
- [ ] Update API client version
- [ ] Adjust error handling
- [ ] Update test expectations
```

## 🔄 CI/CD Integration

### GitHub Actions Setup

1. **Add Secrets**:
   ```
   OPENAI_API_KEY: Your OpenAI key
   ANTHROPIC_API_KEY: Your Anthropic key
   ```

2. **Test Workflow**:
   ```yaml
   - name: Run Integration Tests
     env:
       OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
       ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
     run: |
       python scripts/run_integration_tests.py --provider all
   ```

3. **Schedule**:
   - Weekly: Mondays at 3 AM UTC
   - On release branches
   - Manual trigger available

## ✅ Success Criteria

Integration tests are successful when:

1. **Ollama**: All 16 tests pass
2. **OpenAI**: All 14 tests pass
3. **Anthropic**: All 15 tests pass
4. **Costs**: Total < $0.02
5. **No Security Issues**: No credentials exposed
6. **Documentation**: All results recorded

## 🚨 Emergency Procedures

### If Costs Spike

1. **Immediately**:
   ```bash
   # Kill all Python processes
   pkill -f pytest
   pkill -f python
   ```

2. **Check dashboards** for unusual activity
3. **Rotate API keys** if compromise suspected
4. **Review logs** for root cause

### If Tests Hang

1. **Check timeouts** are working (60s default)
2. **Force stop**: Ctrl+C twice
3. **Clean up**:
   ```bash
   ps aux | grep pytest | grep -v grep | awk '{print $2}' | xargs kill -9
   ```

## 📚 Additional Resources

- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Anthropic API Reference](https://docs.anthropic.com/claude/reference/getting-started-with-the-api)
- [Ollama Documentation](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [pytest Documentation](https://docs.pytest.org/en/stable/)

---

Remember: Start small, monitor closely, and document everything!
