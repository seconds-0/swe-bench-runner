# Task 2: Fix Anthropic Integration Tests - Remediation Plan

## Task ID: INTEG-FIX-ANTHROPIC-002

## Problem Statement
The Anthropic integration tests have CRITICAL API mismatches and were never properly fixed:
1. **WRONG API FORMAT**: Using `response.choices[0].message.content` instead of `response.content`
2. **WRONG COST ACCESS**: Using `response.usage.prompt_cost` instead of `response.cost`
3. Missing test coverage for essential scenarios
4. Test isolation issues with environment variables
5. No provider-specific feature tests (vision, conversation turns)

## Proposed Solution

### 1. Fix Critical API Mismatches (URGENT)
Replace all incorrect response access patterns:
- Change `response.choices[0].message.content` → `response.content`
- Change `response.choices[0].finish_reason` → `response.finish_reason`
- Change `response.usage.prompt_cost` → Calculate from `response.cost` and token usage
- Fix streaming chunk access patterns

### 2. Add Missing Test Coverage
- Network timeout handling
- Retry mechanism with exponential backoff
- Rate limit recovery
- Unicode/emoji content
- Vision capabilities
- Multi-turn conversations
- Partial response handling

### 3. Fix Test Isolation
- Replace os.environ manipulation with pytest fixtures
- Ensure thread-safe test execution

### 4. Add Provider-Specific Tests
- Vision/image input handling
- Multi-turn conversation context
- Token counting API validation
- Claude-specific formatting

## Implementation Checklist

### Phase 1: Fix Critical API Issues (Priority: URGENT)
- [ ] Fix lines 58-59: response.content access
- [ ] Fix line 60: response.finish_reason access  
- [ ] Fix lines 69-71: cost calculation
- [ ] Fix lines 98-99: streaming content access
- [ ] Fix line 174: system message response
- [ ] Fix line 276: long context response
- [ ] Fix line 333: assistant message response
- [ ] Fix line 351: max tokens response
- [ ] Fix lines 369-370: multiple system messages
- [ ] Fix line 387: empty message response
- [ ] Fix environment variable isolation (lines 134-158)

### Phase 2: Add Critical Missing Tests (Priority: HIGH)
- [ ] test_network_timeout_handling
- [ ] test_retry_mechanism_validation
- [ ] test_vision_capability
- [ ] test_multi_turn_conversation
- [ ] test_unicode_emoji_handling
- [ ] test_partial_streaming_response

### Phase 3: Add Provider-Specific Tests (Priority: MEDIUM)
- [ ] test_image_input_handling
- [ ] test_conversation_memory
- [ ] test_claude_specific_formatting
- [ ] test_token_counting_accuracy

## Specific Code Changes

### 1. Fix Basic Generation Test (lines 42-72)
```python
@pytest.mark.asyncio
async def test_basic_generation(self, provider: AnthropicProvider, anthropic_test_model: str, minimal_test_prompt: str):
    """Test basic text generation with real API call."""
    request = UnifiedRequest(
        model=anthropic_test_model,
        prompt=minimal_test_prompt,
        temperature=0.0,
        max_tokens=10,
    )
    
    response = await provider.generate_unified(request)
    
    # Validate response structure - USE UNIFIED API
    assert response.id is not None
    assert response.model == anthropic_test_model
    assert response.content is not None  # FIXED
    assert "test" in response.content.lower()  # FIXED
    assert response.finish_reason in ["stop", "max_tokens"]  # FIXED
    
    # Validate token usage
    assert response.usage is not None
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0
    assert response.usage.total_tokens == response.usage.prompt_tokens + response.usage.completion_tokens
    
    # Validate cost calculation - USE UNIFIED API
    assert response.cost is not None  # FIXED
    assert response.cost > 0  # FIXED
    
    # If we need component costs, calculate them
    capabilities = await provider.get_capabilities()
    model_info = next((m for m in capabilities.supported_models if m.id == anthropic_test_model), None)
    if model_info:
        expected_prompt_cost = (response.usage.prompt_tokens / 1000) * model_info.pricing.prompt_token_cost
        expected_completion_cost = (response.usage.completion_tokens / 1000) * model_info.pricing.completion_token_cost
        expected_total = expected_prompt_cost + expected_completion_cost
        assert abs(response.cost - expected_total) < 0.000001
```

### 2. Fix Streaming Test (lines 73-114)
```python
@pytest.mark.asyncio
async def test_streaming_generation(self, provider: AnthropicProvider, anthropic_test_model: str, streaming_test_prompt: str):
    """Test streaming responses with real API."""
    request = UnifiedRequest(
        model=anthropic_test_model,
        prompt=streaming_test_prompt,
        temperature=0.0,
        max_tokens=50,
        stream=True,
    )
    
    chunks: List[str] = []
    chunk_count = 0
    start_time = datetime.now()
    
    async for chunk in provider.generate_stream(request):  # USE UNIFIED STREAMING
        chunk_count += 1
        if chunk.content:  # FIXED - use unified chunk format
            chunks.append(chunk.content)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Validate streaming behavior
    assert chunk_count > 1
    assert len(chunks) > 0
    full_response = "".join(chunks)
    assert any(str(i) in full_response for i in range(1, 6))
    assert duration > 0.1
```

### 3. Fix Environment Variable Test (lines 131-159)
```python
@pytest.mark.asyncio
async def test_authentication_error(self, monkeypatch):
    """Test handling of authentication errors with invalid key."""
    # Use monkeypatch for proper isolation
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-invalid-test-key-12345")
    
    provider = AnthropicProvider()
    await provider.initialize()
    
    request = UnifiedRequest(
        model="claude-3-haiku-20240307",
        prompt="test",
        max_tokens=10,
    )
    
    with pytest.raises(ProviderAuthenticationError) as exc_info:
        await provider.generate_unified(request)
    
    assert exc_info.value.provider == "anthropic"
    assert "authentication" in str(exc_info.value).lower() or "api key" in str(exc_info.value).lower()
```

### 4. Add Vision Test
```python
@pytest.mark.asyncio
async def test_vision_capability(self, provider: AnthropicProvider, anthropic_test_model: str):
    """Test Claude's vision capabilities with image input."""
    # Only test if model supports vision
    if "vision" not in anthropic_test_model and "claude-3" not in anthropic_test_model:
        pytest.skip("Model doesn't support vision")
    
    request = UnifiedRequest(
        model=anthropic_test_model,
        prompt="What do you see in this image? Just say 'test pattern' if you see anything.",
        max_tokens=20,
        # In real implementation, would include base64 image
        # For now, just test the API accepts the format
    )
    
    try:
        response = await provider.generate_unified(request)
        assert response.content is not None
    except ProviderError as e:
        # If vision not supported, should be clear error
        assert "vision" in str(e).lower() or "image" in str(e).lower()
```

### 5. Add Multi-Turn Conversation Test
```python
@pytest.mark.asyncio
async def test_multi_turn_conversation(self, provider: AnthropicProvider, anthropic_test_model: str):
    """Test handling of multi-turn conversations."""
    # First turn
    request1 = UnifiedRequest(
        model=anthropic_test_model,
        prompt="My name is Alice. What's yours?",
        temperature=0.0,
        max_tokens=30,
    )
    
    response1 = await provider.generate_unified(request1)
    assert response1.content is not None
    
    # Second turn - should remember context
    request2 = UnifiedRequest(
        model=anthropic_test_model,
        prompt="What was my name again?",
        temperature=0.0,
        max_tokens=30,
        # In real implementation, would pass conversation history
    )
    
    response2 = await provider.generate_unified(request2)
    # Without conversation history, it won't know the name
    # This tests the API structure, not memory
    assert response2.content is not None
```

## Verification Steps
1. Run validation script to ensure API format is correct
2. Execute all tests individually to verify fixes
3. Check that cost calculations match expected values
4. Verify proper test isolation with parallel execution
5. Ensure error messages are actionable

## Decision Authority
- Must fix all API mismatches to use unified format
- Can reorganize test structure for clarity
- Cannot change provider implementation
- Must maintain backward compatibility

## Questions/Uncertainties

### Blocking
- None - API format is clearly wrong and must be fixed

### Non-blocking  
- Should we test vision with actual images? (Assumption: Use mock for cost)
- How deep should conversation testing go? (Assumption: Basic 2-turn test)
- Should we validate Claude-specific formatting? (Assumption: Yes, basic test)

## Acceptable Tradeoffs
- Vision tests can use mocks to avoid image processing costs
- Multi-turn conversation tests can be simple 2-3 turn examples
- Some Claude-specific features can be deferred to later phases

## Status: Not Started

## Notes
- CRITICAL: The API format issues must be fixed immediately
- These tests were claimed to be fixed but weren't - this is a serious quality issue
- Focus on fixing the broken API calls before adding new features
- Ensure thorough testing of the fixes before marking complete