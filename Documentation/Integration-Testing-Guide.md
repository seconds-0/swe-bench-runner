# Integration Testing Guide

This guide explains how to run true integration tests that validate our provider implementations against real APIs.

## Overview

Our integration tests make **real API calls** to external services to validate:
- API contract adherence
- Real-world error handling
- Streaming behavior
- Cost calculation accuracy
- Token counting validation
- Performance characteristics

## Test Structure

Integration tests are located in `tests/integration/` and are marked with `@pytest.mark.integration`. They are **skipped by default** to avoid unexpected API costs during regular development.

### Available Integration Tests

1. **OpenAI Integration** (`test_openai_integration.py`)
   - Basic text generation
   - Streaming responses
   - Error handling (auth, model not found, token limits)
   - Cost calculation validation
   - Model availability queries
   - **NEW:** JSON mode generation
   - **NEW:** Concurrent request handling
   - **NEW:** Max tokens enforcement
   - **NEW:** System message handling

2. **Anthropic Integration** (`test_anthropic_integration.py`)
   - Basic text generation
   - Streaming with metadata chunks
   - System message handling
   - Token counting API validation
   - Long context handling
   - **NEW:** Concurrent streaming requests
   - **NEW:** Assistant message conversations
   - **NEW:** Max tokens enforcement
   - **NEW:** Multiple system messages
   - **NEW:** Empty message edge cases

3. **Ollama Integration** (`test_ollama_integration.py`)
   - Local model execution
   - Connection error handling
   - Model availability checks
   - Performance characteristics
   - Free tier validation (costs = $0)
   - **NEW:** Model pull functionality
   - **NEW:** Concurrent local requests
   - **NEW:** Hardware resource information
   - **NEW:** Request timeout handling
   - **NEW:** Model switching
   - **NEW:** Malformed request handling

## Running Integration Tests

### Prerequisites

1. **API Keys** (for cloud providers):
   ```bash
   export OPENAI_API_KEY="sk-..."
   export ANTHROPIC_API_KEY="sk-ant-..."
   ```

2. **Ollama** (for local testing):
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Start Ollama service
   ollama serve
   
   # Pull test model
   ollama pull llama3.2:1b
   ```

### Running Tests

```bash
# Run ALL integration tests (requires all providers)
pytest tests/integration -m integration -v

# Run specific provider tests
pytest tests/integration/test_openai_integration.py -m integration -v
pytest tests/integration/test_anthropic_integration.py -m integration -v
pytest tests/integration/test_ollama_integration.py -m integration -v

# Run with detailed output
pytest tests/integration -m integration -v -s

# Run a specific test
pytest tests/integration/test_openai_integration.py::TestOpenAIIntegration::test_basic_generation -m integration -v
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required for OpenAI tests |
| `OPENAI_TEST_MODEL` | Model to test | `gpt-3.5-turbo` |
| `ANTHROPIC_API_KEY` | Anthropic API key | Required for Anthropic tests |
| `ANTHROPIC_TEST_MODEL` | Model to test | `claude-3-haiku-20240307` |
| `OLLAMA_TEST_MODEL` | Model to test | `llama3.2:1b` |

## Cost Management

Integration tests use minimal models and prompts to reduce costs:

- **OpenAI**: Uses `gpt-3.5-turbo` (~$0.001 per test run)
- **Anthropic**: Uses `claude-3-haiku` (~$0.0008 per test run)
- **Ollama**: Free (local execution)

Estimated cost for full test suite: < $0.02

## CI/CD Integration

Integration tests run automatically in CI:

1. **Weekly Schedule**: Every Monday at 3 AM UTC
2. **Release Branches**: On any `release/*` branch
3. **Manual Trigger**: Add `[integration]` to commit message
4. **Workflow Dispatch**: Manual trigger via GitHub Actions UI

API keys are stored as GitHub secrets and injected securely during CI runs.

## Writing New Integration Tests

When adding new integration tests:

1. **Use Minimal Prompts**: Keep API costs low
   ```python
   @pytest.fixture
   def minimal_test_prompt() -> str:
       return "Say 'test' and nothing else."
   ```

2. **Add Skip Conditions**: Handle missing credentials gracefully
   ```python
   @pytest.fixture
   def skip_without_api_key():
       if not os.environ.get("PROVIDER_API_KEY"):
           pytest.skip("PROVIDER_API_KEY not set")
   ```

3. **Test Real Behaviors**: Focus on API-specific behaviors
   ```python
   async def test_streaming_generation(self, provider):
       # Test that streaming actually streams incrementally
       chunks = []
       async for chunk in provider.stream_unified(request):
           chunks.append(chunk)
       assert len(chunks) > 1  # Multiple chunks received
   ```

4. **Validate Costs**: Ensure pricing calculations are accurate
   ```python
   async def test_cost_calculation_accuracy(self, provider):
       response = await provider.generate_unified(request)
       expected_cost = (tokens / 1000) * price_per_1k
       assert abs(response.usage.total_cost - expected_cost) < 0.000001
   ```

## Security Considerations

1. **Never commit API keys** - Use environment variables only
2. **Sanitize error messages** - Don't log sensitive data
3. **Use test-specific models** - Avoid expensive models
4. **Set spending limits** - Configure limits in provider dashboards

## Troubleshooting

### OpenAI Tests Failing

1. Check API key is valid: `curl https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY"`
2. Verify account has credits
3. Check for rate limiting

### Anthropic Tests Failing

1. Verify API key format: Should start with `sk-ant-`
2. Check model availability in your region
3. Ensure account is active

### Ollama Tests Failing

1. Verify Ollama is running: `curl http://localhost:11434/api/tags`
2. Check model is pulled: `ollama list`
3. Ensure sufficient disk space for models

## Best Practices

1. **Run Before Major Changes**: Always run integration tests before releasing
2. **Monitor Costs**: Check provider dashboards regularly
3. **Update Models**: Keep test models current with provider offerings
4. **Document Failures**: When tests fail due to API changes, document the changes

## Recently Added Test Coverage

The following critical test scenarios have been added to improve production reliability:

### Concurrent Request Handling
- Tests multiple simultaneous requests to verify thread safety
- Validates rate limiting under concurrent load
- Ensures proper resource management

### Advanced Features
- **JSON Mode (OpenAI)**: Structured output generation
- **Max Tokens Enforcement**: Validates token limits are respected
- **Message Role Handling**: Tests various conversation patterns

### Edge Cases
- Empty message handling
- Invalid role types
- Multiple system messages
- Model switching during session

### Local Provider Features (Ollama)
- Model pull functionality
- Hardware resource monitoring
- Timeout handling for slow hardware

## Future Enhancements

- [ ] Add retry mechanism testing with mocked failures
- [ ] Add rate limit testing with controlled triggers
- [ ] Add circuit breaker state transition tests
- [ ] Add function calling/tool use tests
- [ ] Add performance benchmarking
- [ ] Add multi-provider comparison tests
- [ ] Implement cost tracking and reporting
- [ ] Add network failure simulation tests