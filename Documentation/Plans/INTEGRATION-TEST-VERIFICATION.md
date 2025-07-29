# Integration Test Verification Plan

**Task ID**: TEST-IntegrationVerification
**Status**: In Progress

## Problem Statement

We have comprehensive integration tests for OpenAI, Anthropic, and Ollama providers, but we haven't verified they actually work with real APIs. Without running these tests with actual credentials, we cannot claim our provider integrations are production-ready. We need to safely execute these tests while managing costs and security.

## Proposed Solution

Create a systematic approach to verify all integration tests work with real APIs:

1. **Phased Execution**: Start with cheapest/simplest tests first
2. **Cost Controls**: Use minimal models and strict token limits
3. **Security**: Secure credential handling without exposure
4. **Documentation**: Record all results and issues found
5. **Remediation**: Fix any failing tests discovered

### Execution Strategy

#### Phase 1: Environment Setup & Cost Estimation
- Set up secure credential management
- Calculate exact costs for each test
- Create cost monitoring dashboard
- Set up local Ollama for free testing

#### Phase 2: Ollama Testing (Free)
- Start with Ollama since it's free and local
- Verify all Ollama tests pass
- Document any issues found

#### Phase 3: Minimal API Tests
- Run single basic generation test for each cloud provider
- Verify credentials work and basic flow succeeds
- Estimate actual vs predicted costs

#### Phase 4: Full Test Suite Execution
- Run complete test suites for each provider
- Monitor costs in real-time
- Document all failures

#### Phase 5: Remediation
- Fix any failing tests
- Update documentation
- Create CI secrets for automated testing

## Implementation Checklist

### Phase 1: Environment Setup
- [x] Create `.env.integration` file for test credentials (git-ignored)
- [x] Set up cost tracking spreadsheet
- [x] Install and configure Ollama locally
- [x] Create test execution script with cost guards
- [x] Set up provider dashboards for monitoring

### Phase 2: Ollama Verification (Free, Local)
- [ ] Start Ollama service
- [ ] Pull required model: `llama3.2:1b`
- [ ] Run Ollama integration tests
- [ ] Document any failures
- [ ] Fix Ollama-specific issues

### Phase 3: Minimal Cloud Provider Tests
- [ ] Run single OpenAI test: `test_basic_generation`
- [ ] Run single Anthropic test: `test_basic_generation`
- [ ] Verify costs match estimates
- [ ] Check error handling with invalid keys

### Phase 4: Full Test Execution
- [ ] Run all OpenAI tests (~$0.01 estimated)
- [ ] Run all Anthropic tests (~$0.008 estimated)
- [ ] Monitor real-time costs
- [ ] Document test results

### Phase 5: Issue Remediation
- [ ] Fix any API contract mismatches
- [ ] Update error handling for current API responses
- [ ] Adjust token counting if needed
- [ ] Update cost calculations if pricing changed

### Phase 6: Documentation & CI Setup
- [ ] Update test results in documentation
- [ ] Create GitHub secrets setup guide
- [ ] Test CI integration workflow
- [ ] Create cost monitoring automation

## Dependencies

- **Tools**: pytest, python-dotenv, Ollama
- **Credentials**: OpenAI API key, Anthropic API key
- **Local Setup**: Ollama installed and running
- **Cost Budget**: ~$0.05 for complete verification

## Verification Steps

1. All tests pass with real APIs
2. Costs stay within budget (<$0.05)
3. No credentials exposed in logs
4. CI secrets configured correctly
5. Documentation updated with results

## Decision Authority

**Can decide independently:**
- Test execution order
- Debugging approaches for failures
- Minor test adjustments for API changes
- Documentation updates

**Need user input:**
- Spending more than $0.05 on tests
- Major architectural changes to providers
- Removing/skipping tests permanently
- CI workflow modifications

## Questions/Uncertainties

**Blocking:**
- Need actual API credentials to proceed
- Need approval for ~$0.05 test budget

**Non-blocking:**
- Exact error messages from current APIs (will discover during testing)
- Whether all test models are still available
- Current rate limits for test accounts

## Acceptable Tradeoffs

- Skip expensive models even if documented
- Use minimal prompts to reduce costs
- Test core functionality over edge cases if costs escalate
- Document but don't fix minor API inconsistencies

## Cost Safety Measures

### Per-Test Cost Limits
```python
# Maximum cost per test in dollars
MAX_COST_PER_TEST = 0.001  # $0.001

# Maximum total session cost
MAX_TOTAL_COST = 0.05  # $0.05

# Token limits per test
MAX_TOKENS_PER_TEST = 100
```

### Cost Tracking Script
```bash
#!/bin/bash
# track_costs.sh - Monitor costs during test execution

echo "Starting cost tracking..."
echo "Provider,Test,Tokens,Cost" > test_costs.csv

# Run tests with cost tracking
pytest tests/integration -m integration -v \
  --cost-report=test_costs.csv \
  --max-cost=0.05
```

## Secure Credential Management

### Local Testing Setup
```bash
# .env.integration (git-ignored)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_TEST_MODEL=gpt-3.5-turbo
ANTHROPIC_TEST_MODEL=claude-3-haiku-20240307
OLLAMA_TEST_MODEL=llama3.2:1b

# Test with rate limits
OPENAI_MAX_REQUESTS_PER_MINUTE=3
ANTHROPIC_MAX_REQUESTS_PER_MINUTE=3
```

### Execution Script
```python
# run_integration_tests.py
import os
from dotenv import load_dotenv

# Load test credentials
load_dotenv('.env.integration')

# Add cost tracking
os.environ['TRACK_API_COSTS'] = 'true'
os.environ['MAX_TEST_COST'] = '0.05'

# Run tests
os.system('pytest tests/integration -m integration -v')
```

## Expected Results

### Ollama Tests
- All 16 tests should pass
- No costs incurred (local execution)
- ~30 seconds execution time

### OpenAI Tests
- All 14 tests should pass
- ~$0.01 total cost
- ~45 seconds execution time

### Anthropic Tests
- All 15 tests should pass
- ~$0.008 total cost
- ~40 seconds execution time

## Risk Mitigation

1. **Cost Overrun**: Hard limits in test code
2. **Credential Leak**: Use .env files, never commit
3. **Rate Limiting**: Add delays between tests
4. **API Changes**: Document and adapt, don't fail
5. **Network Issues**: Add retries for transient failures

## Notes

- Start testing at low-traffic times to avoid rate limits
- Keep provider dashboards open during testing
- Save all test outputs for analysis
- Create before/after diffs of any code changes
