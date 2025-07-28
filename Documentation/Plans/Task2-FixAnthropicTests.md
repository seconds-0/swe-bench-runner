# Task 2: Fix Anthropic Integration Tests

**Branch**: fix/anthropic-integration-tests
**File**: `tests/integration/test_anthropic_integration.py`
**Priority**: CRITICAL

## Objective
Fix all Anthropic integration tests to use the correct UnifiedRequest API. Currently tests are using `messages` array which doesn't exist in UnifiedRequest.

## Correct API Reference
```python
# CORRECT - from working Ollama tests:
request = UnifiedRequest(
    prompt="user message here",
    system_message="optional system message",  # Only when needed
    model=model_name,
    temperature=0.0,
    max_tokens=10,
)

# WRONG - current broken pattern:
request = UnifiedRequest(
    messages=[{"role": "user", "content": "text"}],  # This field doesn't exist!
    ...
)
```

## Specific Changes Required

### 1. Basic Generation Test (line 47)
**Current (WRONG)**:
```python
messages=[{"role": "user", "content": minimal_test_prompt}],
```
**Fix to**:
```python
prompt=minimal_test_prompt,
```

### 2. Streaming Test (line 78)
**Current (WRONG)**:
```python
messages=[{"role": "user", "content": streaming_test_prompt}],
```
**Fix to**:
```python
prompt=streaming_test_prompt,
```

### 3. Invalid API Key Test (line 120)
**Current (WRONG)**:
```python
messages=[{"role": "user", "content": "test"}],
```
**Fix to**:
```python
prompt="test",
```

### 4. Rate Limit Test (line 143)
**Current (WRONG)**:
```python
messages=[{"role": "user", "content": "test"}],
```
**Fix to**:
```python
prompt="test",
```

### 5. System Message Test (lines 165-168)
**Current (WRONG)**:
```python
messages=[
    {"role": "system", "content": "You are a helpful assistant that always says 'Hello!' first."},
    {"role": "user", "content": "What is 2+2?"}
],
```
**Fix to**:
```python
prompt="What is 2+2?",
system_message="You are a helpful assistant that always says 'Hello!' first.",
```

### 6. Temperature Tests (lines 189, 199, 213)
Look for patterns like:
```python
messages=test_messages
```
These likely need to be changed based on what `test_messages` contains. If it's a list with user content, extract the user message as `prompt`.

### 7. Cost Calculation Test (line 224)
**Current (WRONG)**:
```python
messages=[{"role": "user", "content": cost_test_prompt}],
```
**Fix to**:
```python
prompt=cost_test_prompt,
```

### 8. Token Counting Test (line 270)
**Current (WRONG)**:
```python
messages=[
    {"role": "user", "content": token_test_prompt}
],
```
**Fix to**:
```python
prompt=token_test_prompt,
```

### 9. Parallel Requests Test (line 298)
**Current (WRONG)**:
```python
messages=[{"role": "user", "content": f"Count to 3 (request {i})"}],
```
**Fix to**:
```python
prompt=f"Count to 3 (request {i})",
```

### 10. Model Not Found Test (line 331)
**Current (WRONG)**:
```python
messages=[
    {"role": "user", "content": "test"}
],
```
**Fix to**:
```python
prompt="test",
```

### 11. Max Tokens Test (line 353)
**Current (WRONG)**:
```python
messages=[{"role": "user", "content": "Write a very long story about a dragon."}],
```
**Fix to**:
```python
prompt="Write a very long story about a dragon.",
```

### 12. Multiple System Messages Test (lines 370-374)
**Current (WRONG)**:
```python
messages=[
    {"role": "system", "content": "You are helpful."},
    {"role": "system", "content": "Always be concise."},
    {"role": "user", "content": "Tell me about dragons."}
],
```
**Fix to**:
```python
prompt="Tell me about dragons.",
system_message="You are helpful. Always be concise.",  # Combine multiple system messages
```

### 13. Claude-Specific Features Test (line 392)
Look for similar `messages=[` pattern and fix accordingly.

## Special Considerations

For the temperature variation tests that use `test_messages` variable:
1. Check what `test_messages` contains
2. If it's defined as a list of messages, extract the user content for `prompt`
3. If there's a system message, extract it for `system_message`

## Testing Instructions

1. After making changes, run a quick syntax check:
   ```bash
   python -m py_compile tests/integration/test_anthropic_integration.py
   ```

2. Run a single test to verify the fix works (requires Anthropic API key):
   ```bash
   pytest tests/integration/test_anthropic_integration.py::TestAnthropicIntegration::test_basic_generation -xvs
   ```

3. If no API key available, at least verify imports work:
   ```python
   from tests.integration.test_anthropic_integration import TestAnthropicIntegration
   ```

## Success Criteria

- All `messages=` usages replaced with `prompt=`
- System messages use `system_message=` parameter
- Multiple system messages combined into one
- No syntax errors
- Tests use only fields that exist in UnifiedRequest class
- Pattern matches working Ollama tests

## Additional Notes

- Do NOT change test logic or assertions
- Do NOT add new features
- ONLY fix the API usage to match UnifiedRequest
- Keep all other test code exactly the same
- When combining multiple system messages, join with a space or period as appropriate