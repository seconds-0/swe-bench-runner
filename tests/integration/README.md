# Integration Tests

This directory contains integration tests that validate our provider implementations against real APIs.

## Running Integration Tests

Integration tests are marked with `@pytest.mark.integration` and are skipped by default. To run them:

```bash
# Run all integration tests (requires API keys)
pytest tests/integration -m integration

# Run specific provider tests
pytest tests/integration/test_openai_integration.py -m integration
pytest tests/integration/test_anthropic_integration.py -m integration
pytest tests/integration/test_openrouter_integration.py -m integration
pytest tests/integration/test_ollama_integration.py -m integration

# Run with verbose output
pytest tests/integration -m integration -v -s
```

## Required Environment Variables

### OpenAI
- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_TEST_MODEL` (optional): Model to test with (default: gpt-3.5-turbo)

### Anthropic
- `ANTHROPIC_API_KEY`: Your Anthropic API key
- `ANTHROPIC_TEST_MODEL` (optional): Model to test with (default: claude-haiku-3-5-20241022)

### OpenRouter
- `OPENROUTER_API_KEY`: Your OpenRouter API key (universal provider access)
- `OPENROUTER_TEST_MODEL` (optional): Model to test with (default: anthropic/claude-3-haiku)

### Ollama
- No API key required
- Ollama service must be running on localhost:11434
- Test model must be pulled: `ollama pull llama3.2:1b`

## Test Coverage

Integration tests validate:
1. **Basic Generation**: Simple prompt → response flow
2. **Streaming**: Real-time token streaming
3. **Error Handling**: Invalid models, rate limits, auth failures
4. **Cost Calculation**: Verify actual costs match estimates
5. **Token Counting**: Validate token counts from real responses
6. **Model Availability**: Check supported models are accessible

## CI/CD Integration

In CI, integration tests run:
- Weekly on schedule
- On release branches
- When `[integration]` is in commit message

API keys are stored as GitHub secrets and injected securely.

## Cost Management

Tests use minimal models and short prompts to minimize API costs:
- OpenAI: gpt-3.5-turbo (~$0.001 per test)
- Anthropic: claude-haiku-3.5 (~$0.0008 per test)
- OpenRouter: anthropic/claude-3-haiku (~$0.0008 per test)
- Ollama: Free (local)

Estimated monthly cost for weekly runs: < $2

## Adding New Integration Tests

1. Create test file: `test_<provider>_integration.py`
2. Mark all tests with `@pytest.mark.integration`
3. Use minimal prompts to reduce costs
4. Add skip conditions for missing credentials
5. Document any special setup requirements

## Security

- Never commit API keys
- Use environment variables only
- Tests should not log sensitive data
- Sanitize any error messages that might contain keys