# Task 1: Fix OpenAI Integration Tests - Remediation Plan

## Task ID: INTEG-FIX-OPENAI-001

## Problem Statement
The OpenAI integration tests have several critical issues:
1. Missing test coverage for essential scenarios (timeouts, retries, rate limits)
2. Weak assertions that don't properly validate functionality
3. Test isolation issues with environment variable manipulation
4. Missing provider-specific feature tests (tools, logprobs, functions)
5. No performance benchmarks or cost optimization

## Proposed Solution

### 1. Add Missing Test Coverage
Create comprehensive tests for all critical scenarios:
- Network timeout handling
- Retry mechanism validation with mocked failures
- Rate limit recovery timing
- Connection pool exhaustion
- Unicode/emoji content handling
- Malformed response handling
- Large response handling near token limits

### 2. Strengthen Assertions
Replace weak assertions with comprehensive validation:
- JSON mode: Validate exact structure and content
- Cost calculation: Validate both components and total
- Concurrent requests: Track and validate specific failures
- System messages: Verify message influence on output

### 3. Fix Test Isolation
Use pytest fixtures for environment manipulation:
- Replace direct os.environ modification with monkeypatch
- Add proper setup/teardown for each test
- Ensure tests can run in parallel safely

### 4. Add Provider-Specific Tests
Test OpenAI-specific features:
- Function calling with tools
- Logprobs parameter
- Seed for reproducibility
- N parameter for multiple completions
- Response format enforcement

### 5. Optimize Costs
Implement cost-saving measures:
- Cache model capabilities for session
- Reduce concurrent test count to 2
- Use minimal prompts where possible
- Add test result caching for expensive operations

## Implementation Checklist

### Phase 1: Fix Existing Tests (Priority: URGENT)
- [ ] Fix environment variable isolation in test_authentication_error
- [ ] Strengthen JSON mode assertions (lines 251-252)
- [ ] Improve concurrent request validation
- [ ] Add proper cost component validation
- [ ] Enhance system message verification

### Phase 2: Add Critical Missing Tests (Priority: HIGH)
- [ ] test_network_timeout_handling
- [ ] test_retry_mechanism_with_exponential_backoff
- [ ] test_rate_limit_recovery_timing
- [ ] test_unicode_emoji_content
- [ ] test_malformed_response_handling
- [ ] test_connection_pool_exhaustion

### Phase 3: Add Provider-Specific Tests (Priority: MEDIUM)
- [ ] test_function_calling_with_tools
- [ ] test_logprobs_generation
- [ ] test_seed_reproducibility
- [ ] test_multiple_completions_n_parameter
- [ ] test_response_format_json_schema

### Phase 4: Optimize Performance (Priority: LOW)
- [ ] Implement model capability caching
- [ ] Reduce concurrent test requests
- [ ] Add performance benchmarks
- [ ] Optimize prompt lengths

## Specific Code Changes

### 1. Fix Authentication Test (lines 117-144)
```python
@pytest.mark.asyncio
async def test_authentication_error(self, monkeypatch):
    """Test handling of authentication errors with invalid key."""
    # Use monkeypatch instead of direct os.environ modification
    monkeypatch.setenv("OPENAI_API_KEY", "sk-invalid-test-key-12345")

    provider = OpenAIProvider()
    await provider.initialize()

    request = UnifiedRequest(
        model="gpt-3.5-turbo",
        prompt="test",
        max_tokens=10,
    )

    with pytest.raises(ProviderAuthenticationError) as exc_info:
        await provider.generate_unified(request)

    assert exc_info.value.provider == "openai"
    assert "authentication" in str(exc_info.value).lower() or "api key" in str(exc_info.value).lower()
```

### 2. Fix JSON Mode Test (lines 230-254)
```python
@pytest.mark.asyncio
async def test_json_mode_generation(self, provider: OpenAIProvider, openai_test_model: str):
    """Test JSON mode for structured output generation."""
    request = UnifiedRequest(
        model=openai_test_model,
        prompt="Generate a JSON object with name='Alice' and age=30.",
        temperature=0.0,
        max_tokens=50,
        response_format={"type": "json_object"},
    )

    response = await provider.generate_unified(request)

    # Validate response
    assert response.content is not None
    content = response.content.strip()

    # Should be valid JSON with expected structure
    import json
    parsed = json.loads(content)
    assert isinstance(parsed, dict)
    assert "name" in parsed
    assert parsed["name"] == "Alice"
    assert "age" in parsed
    assert parsed["age"] == 30
```

### 3. Add Timeout Test
```python
@pytest.mark.asyncio
async def test_network_timeout_handling(self, provider: OpenAIProvider, openai_test_model: str):
    """Test handling of network timeouts."""
    # Temporarily set very low timeout
    original_timeout = provider._client.timeout
    provider._client.timeout = 0.001  # 1ms timeout

    try:
        request = UnifiedRequest(
            model=openai_test_model,
            prompt="test",
            max_tokens=10,
        )

        with pytest.raises(ProviderNetworkError) as exc_info:
            await provider.generate_unified(request)

        assert "timeout" in str(exc_info.value).lower()
        assert exc_info.value.provider == "openai"
    finally:
        provider._client.timeout = original_timeout
```

### 4. Add Retry Mechanism Test
```python
@pytest.mark.asyncio
async def test_retry_mechanism(self, provider: OpenAIProvider, openai_test_model: str, mocker):
    """Test exponential backoff retry mechanism."""
    # Mock the underlying API call to fail twice then succeed
    call_count = 0
    original_create = provider._client.chat.completions.create

    async def mock_create(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ProviderNetworkError("openai", "Temporary network error")
        return await original_create(*args, **kwargs)

    mocker.patch.object(provider._client.chat.completions, 'create', side_effect=mock_create)

    request = UnifiedRequest(
        model=openai_test_model,
        prompt="test",
        max_tokens=10,
    )

    response = await provider.generate_unified(request)

    # Should succeed after retries
    assert response.content is not None
    assert call_count == 3  # Failed twice, succeeded on third
```

## Verification Steps
1. Run all tests with pytest-xdist for parallel execution
2. Verify no race conditions with environment variables
3. Check that all new tests pass consistently
4. Validate cost tracking accuracy
5. Ensure error messages are helpful and actionable

## Decision Authority
- Can modify test structure and organization
- Can add new test fixtures and utilities
- Must maintain backward compatibility with existing test runners
- Cannot change provider implementation, only tests

## Questions/Uncertainties

### Blocking
- None - all requirements are clear

### Non-blocking
- Should we add performance regression tests? (Assumption: Yes, but low priority)
- Should we test with multiple model versions? (Assumption: Use parametrize for key models)
- How detailed should error message validation be? (Assumption: Check key terms and length)

## Acceptable Tradeoffs
- Performance tests can use shorter prompts to reduce costs
- Some edge cases can be tested with mocks instead of real API calls
- Provider-specific features can be added incrementally

## Status: Not Started

## Notes
- Priority is fixing broken tests first, then adding missing coverage
- Focus on test isolation to enable parallel execution
- Ensure all tests have clear documentation of what they validate
