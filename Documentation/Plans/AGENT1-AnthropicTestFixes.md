# Coding Agent Task: Fix Anthropic Integration Tests

## CRITICAL: API Mismatch Issues

You are tasked with fixing the Anthropic integration tests which are using the WRONG API format. This is a critical quality issue.

## Your Mission

Fix all instances where the tests use the wrong response format. The tests currently use OpenAI's format but should use our unified format.

## Specific Changes Required

### 1. Fix Basic Generation Test (lines 42-72)
**File**: `tests/integration/test_anthropic_integration.py`
**Lines**: 58-59, 60, 69-71
**Fix**:
```python
# Line 58-59 - REPLACE:
assert response.choices[0].message.content is not None
assert "test" in response.choices[0].message.content.lower()

# WITH:
assert response.content is not None
assert "test" in response.content.lower()

# Line 60 - REPLACE:
assert response.choices[0].finish_reason in ["stop", "max_tokens"]

# WITH:
assert response.finish_reason in ["stop", "max_tokens"]

# Lines 69-71 - REPLACE:
assert response.usage.prompt_cost > 0
assert response.usage.completion_cost > 0
assert response.usage.total_cost == response.usage.prompt_cost + response.usage.completion_cost

# WITH:
assert response.cost is not None
assert response.cost > 0
```

### 2. Fix Streaming Test (lines 73-114)
**Lines**: 98-99
**Fix**:
```python
# REPLACE:
if chunk.choices and chunk.choices[0].delta.content:
    chunks.append(chunk.choices[0].delta.content)

# WITH:
if chunk.content:
    chunks.append(chunk.content)
```

### 3. Fix System Message Test (line 174)
**Fix**:
```python
# REPLACE:
assert response.choices[0].message.content is not None

# WITH:
assert response.content is not None
```

### 4. Fix Long Context Test (lines 261-282)
**Lines**: 276, 280-281
**Fix**:
```python
# Line 276 - REPLACE:
assert response.choices[0].message.content is not None

# WITH:
assert response.content is not None

# Lines 280-281 - REPLACE:
content = response.choices[0].message.content.lower()

# WITH:
content = response.content.lower()
```

### 5. Fix Concurrent Streaming Test (lines 283-319)
**Lines**: 303-304
**Fix**:
```python
# REPLACE:
if chunk.choices and chunk.choices[0].delta.content:
    chunks.append(chunk.choices[0].delta.content)

# WITH:
if chunk.content:
    chunks.append(chunk.content)
```

### 6. Fix Assistant Message Test (lines 320-337)
**Lines**: 333-334
**Fix**:
```python
# REPLACE:
assert response.choices[0].message.content is not None
content = response.choices[0].message.content.lower()

# WITH:
assert response.content is not None
content = response.content.lower()
```

### 7. Fix Max Tokens Test (lines 338-353)
**Lines**: 351
**Fix**:
```python
# REPLACE:
assert response.choices[0].finish_reason == "max_tokens"

# WITH:
assert response.finish_reason == "max_tokens"
```

### 8. Fix Multiple System Messages Test (lines 354-373)
**Lines**: 369-370, 372
**Fix**:
```python
# Lines 369-370 - REPLACE:
assert response.choices[0].message.content is not None
assert len(response.choices[0].message.content) > 0

# WITH:
assert response.content is not None
assert len(response.content) > 0

# Line 372 - REPLACE:
assert len(response.choices[0].message.content) < 200

# WITH:
assert len(response.content) < 200
```

### 9. Fix Empty Message Test (lines 374-390)
**Line**: 387
**Fix**:
```python
# REPLACE:
assert response.choices[0].message.content is not None

# WITH:
assert response.content is not None
```

### 10. Fix Authentication Test Environment Isolation (lines 131-159)
**Fix**:
```python
@pytest.mark.asyncio
async def test_authentication_error(self, monkeypatch):
    """Test handling of authentication errors with invalid key."""
    # Use monkeypatch instead of direct os.environ
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

### 11. Fix Cost Calculation Test (lines 217-243)
**Lines**: 240-242
**Fix**:
```python
# REPLACE the validation section with:
# Validate total cost matches calculation
assert response.cost is not None
expected_total_cost = expected_prompt_cost + expected_completion_cost
assert abs(response.cost - expected_total_cost) < 0.000001
```

## Additional Improvements

### Add Missing Import
At the top of the file, ensure monkeypatch is available by checking the imports.

### Add Network Timeout Test
Add this new test after the existing tests:
```python
@pytest.mark.asyncio
async def test_network_timeout_handling(self, provider: AnthropicProvider, anthropic_test_model: str):
    """Test handling of network timeouts."""
    # Save original timeout
    original_timeout = provider._client.timeout if hasattr(provider, '_client') else None
    
    # Set very short timeout
    if hasattr(provider, '_client'):
        provider._client.timeout = 0.001  # 1ms
    
    try:
        request = UnifiedRequest(
            model=anthropic_test_model,
            prompt="test timeout",
            max_tokens=10,
        )
        
        with pytest.raises((ProviderTimeoutError, ProviderNetworkError)) as exc_info:
            await provider.generate_unified(request)
        
        assert "timeout" in str(exc_info.value).lower()
        assert exc_info.value.provider == "anthropic"
    finally:
        # Restore timeout
        if hasattr(provider, '_client') and original_timeout:
            provider._client.timeout = original_timeout
```

## Testing Your Changes

1. Run the validation script:
   ```bash
   python3 scripts/validate_integration_tests_simple.py
   ```

2. Run the Anthropic tests:
   ```bash
   pytest tests/integration/test_anthropic_integration.py -v
   ```

## Important Notes

- Make ONLY the changes listed above
- Do NOT modify the provider implementation
- Do NOT add complex new features
- Focus on fixing the API mismatch issues
- Ensure all tests can run without actual API keys (they should skip appropriately)

## Success Criteria

1. All `response.choices[0]` patterns removed
2. All `response.usage.cost` patterns replaced with `response.cost`
3. Environment isolation using monkeypatch
4. Tests pass validation script
5. No syntax errors

Complete this task carefully - these API mismatches are a critical issue that should have been caught earlier.