# Provider API Verification Report

## Executive Summary

**CRITICAL FINDING**: ALL integration tests (OpenAI, Anthropic, and Ollama) are using the WRONG API. They are testing an API that doesn't exist in the codebase. None of these integration tests would work if actually run.

## API Mismatch Details

### 1. UnifiedRequest API (What Actually Exists)

The `UnifiedRequest` model in `src/swebench_runner/providers/unified_models.py` expects:
```python
@dataclass
class UnifiedRequest:
    prompt: str                          # Required user prompt
    system_message: str | None = None    # Optional system message
    max_tokens: int | None = None
    temperature: float = 0.7
    stream: bool = False
    model: str | None = None
    stop_sequences: list[str] | None = None
```

### 2. What the Tests Are Using (WRONG)

All three integration test files are trying to use:
```python
request = UnifiedRequest(
    model="gpt-4o",
    messages=[{"role": "user", "content": "test"}],  # THIS FIELD DOESN'T EXIST!
    temperature=0.0,
    max_tokens=10
)
```

### 3. Specific Test File Issues

#### test_openai_integration.py
- **Lines with messages array**: 46, 77, 109, 133, 143, 156, 165, 174, 183, 199, 213, 224, 269, 310-312, 332-335, 353, 370-374, 392-396
- **Response access issues**: Expects `response.choices[0].message.content` but should be `response.content`
- **Other API mismatches**: Expects `response.id`, `response.choices` structure

#### test_anthropic_integration.py
- **Lines with messages array**: 47, 78, 120, 143, 165-168, 183, 199, 213, 224, 270-274, 298, 331-335, 353, 369-374, 392-396
- **Response access issues**: Same as OpenAI tests
- **Other API mismatches**: Same structural issues

#### test_ollama_integration.py
- **Already identified**: Using messages array throughout
- **Same fundamental issues**: Wrong request format, wrong response expectations

## How the API Actually Works

### Request Transformation Flow

1. **User creates UnifiedRequest** with `prompt` and optional `system_message`
2. **Provider's transformer** converts this to provider-specific format:
   - **OpenAI**: Creates messages array `[{"role": "system", ...}, {"role": "user", ...}]`
   - **Anthropic**: Creates separate `system` field and `messages` array
   - **Ollama**: Uses `prompt` and `system` fields directly

### Response Transformation Flow

1. **Provider returns** native format (OpenAI's choices, Anthropic's content, etc.)
2. **Provider's parser** converts to `UnifiedResponse`:
   - Direct `content` field (not `choices[0].message.content`)
   - Direct `model`, `usage`, `finish_reason` fields
   - No `choices` array in unified response

## Test Execution Impact

### Would Any Tests Pass?
**NO**. Every single test would fail immediately because:

1. **Request Creation Fails**: `TypeError: __init__() got an unexpected keyword argument 'messages'`
2. **Even if fixed**: Response access would fail with `AttributeError: 'UnifiedResponse' object has no attribute 'choices'`

### False Coverage Claims
- We claim to have integration tests for all three providers
- In reality, we have ZERO working integration tests
- The tests are testing an imaginary API that doesn't exist

## Root Cause Analysis

### Likely Scenario
1. Tests were written based on OpenAI's native API format
2. The unified abstraction layer was designed differently
3. Tests were never updated to match the actual unified API
4. Tests were marked as "integration" but never actually run with real providers

### Evidence
- All tests use identical wrong patterns (copy-paste)
- Tests expect OpenAI-specific response structure even for Anthropic/Ollama
- No evidence these tests were ever run successfully

## Recommendations

### 1. Immediate Actions
- **Do NOT claim integration test coverage** - we have none
- **Document the issue** in test coverage reports
- **Add warning comments** to the test files

### 2. Fix Strategy
Two options:

**Option A: Fix the Tests (Recommended)**
- Rewrite all tests to use correct `prompt`/`system_message` API
- Update response assertions to use `response.content` directly
- Test with actual providers to ensure they work

**Option B: Change the API**
- Modify `UnifiedRequest` to accept `messages` array
- Add compatibility layer to support both formats
- More complex, affects existing code

### 3. Testing Strategy
- Create a single working test first
- Verify it runs with a real provider
- Then fix all other tests following the pattern
- Add CI job that actually runs integration tests (with API keys)

## Example of Correct Test

```python
async def test_basic_generation(provider, model, prompt):
    # CORRECT API usage
    request = UnifiedRequest(
        prompt=prompt,
        system_message="You are a helpful assistant",
        model=model,
        temperature=0.0,
        max_tokens=10
    )

    response = await provider.generate_unified(request)

    # CORRECT response access
    assert response.content is not None
    assert response.model == model
    assert response.usage.prompt_tokens > 0
    assert response.finish_reason in ["stop", "length"]
```

## Conclusion

This is a critical testing infrastructure failure. We have been operating under the false assumption that we have integration test coverage for our providers. In reality:

1. **Zero integration tests work**
2. **All tests use wrong API**
3. **Tests have never been run successfully**
4. **We have no actual provider integration verification**

This explains why the Ollama test issues weren't caught earlier - none of the integration tests for ANY provider have ever worked. They're all testing an API that doesn't exist in our codebase.
