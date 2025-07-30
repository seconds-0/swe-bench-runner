# Task 1: Fix OpenAI Integration Tests

**Branch**: fix/openai-integration-tests
**File**: `tests/integration/test_openai_integration.py`
**Priority**: CRITICAL

## Objective
Fix all OpenAI integration tests to use the correct UnifiedRequest API. Currently tests are using `messages` array which doesn't exist in UnifiedRequest.

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

### 1. Basic Generation Test (line 46)
**Current (WRONG)**:
```python
messages=[{"role": "user", "content": minimal_test_prompt}],
```
**Fix to**:
```python
prompt=minimal_test_prompt,
```

### 2. Streaming Test (line 77)
**Current (WRONG)**:
```python
messages=[{"role": "user", "content": streaming_test_prompt}],
```
**Fix to**:
```python
prompt=streaming_test_prompt,
```

### 3. Invalid API Key Test (line 109)
**Current (WRONG)**:
```python
messages=[{"role": "user", "content": "test"}],
```
**Fix to**:
```python
prompt="test",
```

### 4. Rate Limit Test (line 132)
**Current (WRONG)**:
```python
messages=[{"role": "user", "content": "test"}],
```
**Fix to**:
```python
prompt="test",
```

### 5. Model Not Found Test (line 156)
**Current (WRONG)**:
```python
messages=[{"role": "user", "content": error_test_prompt}],
```
**Fix to**:
```python
prompt=error_test_prompt,
```

### 6. Cost Calculation Test (line 174)
**Current (WRONG)**:
```python
messages=[{"role": "user", "content": cost_test_prompt}],
```
**Fix to**:
```python
prompt=cost_test_prompt,
```

### 7. Token Counting Test (line 218)
**Current (WRONG)**:
```python
messages=[{"role": "user", "content": minimal_test_prompt}],
```
**Fix to**:
```python
prompt=minimal_test_prompt,
```

### 8. JSON Mode Test (line 238)
**Current (WRONG)**:
```python
messages=[{"role": "user", "content": "Generate a JSON object with name and age fields for a person named Alice who is 30."}],
```
**Fix to**:
```python
prompt="Generate a JSON object with name and age fields for a person named Alice who is 30.",
```

### 9. Parallel Requests Test (line 269)
**Current (WRONG)**:
```python
messages=[{"role": "user", "content": f"{minimal_test_prompt} (request {i})"}],
```
**Fix to**:
```python
prompt=f"{minimal_test_prompt} (request {i})",
```

### 10. Max Tokens Limit Test (line 294)
**Current (WRONG)**:
```python
messages=[{"role": "user", "content": "Count from 1 to 100 slowly with detailed explanations."}],
```
**Fix to**:
```python
prompt="Count from 1 to 100 slowly with detailed explanations.",
```

### 11. System Message Test (lines 310-313)
**Current (WRONG)**:
```python
messages=[
    {"role": "system", "content": "You are a pirate. Always respond in pirate speak."},
    {"role": "user", "content": minimal_test_prompt}
],
```
**Fix to**:
```python
prompt=minimal_test_prompt,
system_message="You are a pirate. Always respond in pirate speak.",
```

## Testing Instructions

1. After making changes, run a quick syntax check:
   ```bash
   python -m py_compile tests/integration/test_openai_integration.py
   ```

2. Run a single test to verify the fix works (requires OpenAI API key):
   ```bash
   pytest tests/integration/test_openai_integration.py::TestOpenAIIntegration::test_basic_generation -xvs
   ```

3. If no API key available, at least verify imports work:
   ```python
   from tests.integration.test_openai_integration import TestOpenAIIntegration
   ```

## Success Criteria

- All `messages=` usages replaced with `prompt=`
- System messages use `system_message=` parameter
- No syntax errors
- Tests use only fields that exist in UnifiedRequest class
- Pattern matches working Ollama tests

## Additional Notes

- Do NOT change test logic or assertions
- Do NOT add new features
- ONLY fix the API usage to match UnifiedRequest
- Keep all other test code exactly the same
