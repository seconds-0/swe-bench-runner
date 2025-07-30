# Coding Agent Task: Fix Ollama Integration Tests

## Your Mission

Fix the Ollama integration tests which are ALSO using the WRONG API format (messages instead of prompt).

## Critical Issues Found

1. Using `messages=[{"role": "user", "content": ...}]` instead of `prompt=...`
2. Using `response.choices[0].message.content` instead of `response.content`
3. Using `response.choices[0].finish_reason` instead of `response.finish_reason`

## Specific Changes Required

### 1. Fix ALL UnifiedRequest Usage
Replace all instances of `messages` parameter with `prompt`:
- Lines: 43, 74, 110, 133, 164, 183, 207, 222, 240

### 2. Fix ALL Response Access
Replace all instances of:
- `response.choices[0].message.content` → `response.content`
- `response.choices[0].finish_reason` → `response.finish_reason`
- Lines: 54-56, 175-176, 195-196, 213-214, 230, 232, 253

### 3. Fix Conversation Test
The test at line 180-198 needs proper implementation of context validation

### 4. Fix Timeout Test
The timeout test (lines 329-348) needs to be more reliable

### 5. Fix Malformed Request Test
The test at line 381-388 has incorrect implementation

## Example Fixes

### Basic Generation (lines 38-70)
```python
@pytest.mark.asyncio
async def test_basic_generation(self, provider: OllamaProvider, ollama_test_model: str, minimal_test_prompt: str):
    """Test basic text generation with real Ollama API call."""
    request = UnifiedRequest(
        model=ollama_test_model,
        prompt=minimal_test_prompt,  # FIXED
        temperature=0.0,
        max_tokens=10,
    )

    response = await provider.generate_unified(request)

    # Validate response structure
    assert response.content is not None  # FIXED
    assert response.model == ollama_test_model
    assert len(response.content) > 0  # FIXED
    assert response.finish_reason in ["stop", "length"]  # FIXED

    # Validate token usage
    assert response.usage is not None
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0
    assert response.usage.total_tokens == response.usage.prompt_tokens + response.usage.completion_tokens

    # Ollama is free, so cost should be 0
    assert response.cost == 0.0
```

### System Message Test (lines 163-178)
```python
@pytest.mark.asyncio
async def test_system_message_handling(self, provider: OllamaProvider, ollama_test_model: str):
    """Test Ollama's handling of system messages."""
    request = UnifiedRequest(
        model=ollama_test_model,
        prompt="Hello",
        system_message="You are a pirate. Respond accordingly.",
        temperature=0.7,
        max_tokens=50,
    )

    response = await provider.generate_unified(request)

    # Should get a response
    assert response.content is not None
    assert len(response.content) > 0
```

### Streaming Test
For streaming, check if it uses the correct method name and chunk access.

## Complete the Task

1. Fix all `messages` parameters to use `prompt`
2. Fix all response access to use unified API
3. Ensure tests are properly isolated
4. Add missing exception imports if needed
5. Fix unreliable tests

This is critical - Ollama tests also have the wrong API format!
