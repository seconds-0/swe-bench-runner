# Coding Agent Task: Fix OpenAI Integration Tests

## Your Mission

Fix the OpenAI integration tests which are using the WRONG API format. They're using `messages` instead of `prompt` and the wrong response access patterns.

## Specific Changes Required

### 1. Fix ALL UnifiedRequest Usage
**Throughout the file**: Change `messages=[{"role": "user", "content": ...}]` to `prompt=...`

These occur at lines: 46, 77, 109, 132, 156, 174, 238, 262, 309, 313, 351

### 2. Fix Response Access Patterns
Change all instances of:
- `response.choices[0].message.content` → `response.content`
- `response.choices[0].finish_reason` → `response.finish_reason`
- `response.usage.prompt_cost` → Calculate from `response.cost`
- `response.usage.completion_cost` → Calculate from `response.cost`
- `response.usage.total_cost` → Use `response.cost`

### 3. Fix Streaming Access
Line 89-90:
```python
# REPLACE:
if chunk.choices and chunk.choices[0].delta.content:
    chunks.append(chunk.choices[0].delta.content)

# WITH:
if chunk.content:
    chunks.append(chunk.content)
```

### 4. Fix Environment Variable Test
Lines 117-144: Use monkeypatch instead of direct os.environ manipulation

### 5. Fix JSON Mode Test
Lines 251-252: Strengthen assertions to validate exact structure

### 6. Fix System Message Handling
For requests that need system messages, use the `system_message` parameter:
```python
request = UnifiedRequest(
    model=model,
    prompt="User message",
    system_message="System instructions",
    ...
)
```

## Example Fixes

### Basic Generation Test (lines 42-71)
```python
@pytest.mark.asyncio
async def test_basic_generation(self, provider: OpenAIProvider, openai_test_model: str, minimal_test_prompt: str):
    """Test basic text generation with real API call."""
    request = UnifiedRequest(
        model=openai_test_model,
        prompt=minimal_test_prompt,  # FIXED: Use prompt, not messages
        temperature=0.0,
        max_tokens=10,
    )

    response = await provider.generate_unified(request)

    # Validate response structure
    assert response.content is not None  # FIXED: Use unified API
    assert response.model == openai_test_model
    assert "test" in response.content.lower()  # FIXED
    assert response.finish_reason in ["stop", "length"]  # FIXED

    # Validate token usage
    assert response.usage is not None
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0
    assert response.usage.total_tokens == response.usage.prompt_tokens + response.usage.completion_tokens

    # Validate cost calculation
    assert response.cost is not None  # FIXED: Use unified cost
    assert response.cost > 0
```

### System Message Test (lines 302-316)
```python
@pytest.mark.asyncio
async def test_system_message_handling(self, provider: OpenAIProvider, openai_test_model: str):
    """Test proper handling of system messages."""
    request = UnifiedRequest(
        model=openai_test_model,
        prompt="Hello there!",  # User message
        system_message="You are a pirate. Always respond in pirate speak.",  # System message
        temperature=0.7,
        max_tokens=30,
    )

    response = await provider.generate_unified(request)

    # Should get a response
    assert response.content is not None
    assert len(response.content) > 0
```

## Complete List of Line Changes

1. Line 46: `messages=[{"role": "user", "content": minimal_test_prompt}]` → `prompt=minimal_test_prompt`
2. Line 57-58: Fix response access
3. Line 59: Fix finish_reason access
4. Line 68-70: Fix cost access
5. Line 77: Fix messages parameter
6. Line 89-90: Fix streaming chunk access
7. Line 109: Fix messages parameter
8. Line 117-144: Use monkeypatch
9. Line 132: Fix messages parameter
10. Line 156: Fix messages parameter
11. Line 160: Fix finish_reason access
12. Line 174: Fix messages parameter
13. Line 189-191: Fix cost validation
14. Line 219, 223: Fix response access
15. Line 238: Fix messages parameter
16. Line 251-252: Strengthen JSON validation
17. Line 262: Fix messages parameter
18. Line 298: Fix finish_reason
19. Line 309: Fix messages with system_message
20. Line 313: Fix messages parameter
21. Line 351: Fix messages parameter

## Testing Your Changes

1. Run validation:
   ```bash
   python3 scripts/validate_integration_tests_simple.py
   ```

2. Test syntax:
   ```bash
   python3 -m py_compile tests/integration/test_openai_integration.py
   ```

## Important Notes

- Use `prompt` for user messages, NOT `messages`
- Use `system_message` for system instructions
- Access response fields directly (response.content, not response.choices[0].message.content)
- Use monkeypatch for test isolation
- Keep changes focused on API fixes

Complete this task carefully to ensure consistency with our unified API design.
