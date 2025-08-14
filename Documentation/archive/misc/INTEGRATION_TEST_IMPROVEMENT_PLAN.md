# Integration Test Enhancement Plan

## Executive Summary

Our thorough review of the integration tests revealed several critical issues that need immediate attention:

### Critical Issues Found:
1. **Anthropic tests are STILL BROKEN** - Using `response.choices[0].message.content` instead of `response.content`
2. **Missing timeout and retry tests** - No validation of resilience mechanisms
3. **Weak assertions** - Tests verify basic functionality but miss edge cases
4. **No performance benchmarks** - Cannot detect performance regressions
5. **Test isolation problems** - Environment variable manipulation could cause race conditions

## Immediate Action Items

### 1. Fix Broken Anthropic Tests (CRITICAL)

The Anthropic tests still use the wrong API. They need these changes:

```python
# BROKEN (current):
assert response.choices[0].message.content is not None
assert "test" in response.choices[0].message.content.lower()
assert response.usage.prompt_cost > 0

# FIXED (should be):
assert response.content is not None
assert "test" in response.content.lower()
assert response.cost > 0
```

All response access in `test_anthropic_integration.py` needs fixing.

### 2. Add Critical Missing Tests

#### Timeout Handling Test
```python
@pytest.mark.asyncio
async def test_request_timeout_handling(provider, test_model):
    """Verify requests don't hang indefinitely."""
    import asyncio

    request = UnifiedRequest(
        model=test_model,
        prompt="Generate an extremely detailed 50,000 word essay",
        max_tokens=50000,  # Unreasonably large
    )

    start = datetime.now()
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            provider.generate_unified(request),
            timeout=30.0
        )
    duration = (datetime.now() - start).total_seconds()

    assert duration < 35  # Should timeout close to 30s
```

#### Retry Mechanism Test
```python
@pytest.mark.asyncio
async def test_retry_with_backoff(provider, monkeypatch):
    """Verify exponential backoff retry logic."""
    attempt_times = []

    async def track_retries(*args, **kwargs):
        attempt_times.append(datetime.now())
        if len(attempt_times) < 3:
            raise aiohttp.ClientError("Network error")
        return {"choices": [{"message": {"content": "success"}}]}

    monkeypatch.setattr(provider, '_make_request', track_retries)

    request = UnifiedRequest(model=test_model, prompt="test", max_tokens=5)
    response = await provider.generate_unified(request)

    # Verify retry happened with backoff
    assert len(attempt_times) == 3

    # Check exponential backoff timing
    gap1 = (attempt_times[1] - attempt_times[0]).total_seconds()
    gap2 = (attempt_times[2] - attempt_times[1]).total_seconds()
    assert gap2 > gap1 * 1.5  # Exponential increase
```

### 3. Strengthen Existing Tests

#### Improve JSON Mode Test
```python
@pytest.mark.asyncio
async def test_json_mode_structured_output(provider, openai_test_model):
    """Test JSON mode with schema validation."""
    request = UnifiedRequest(
        model=openai_test_model,
        prompt="""Generate a JSON object representing a person with:
        - name (string): "Alice"
        - age (number): 30
        - hobbies (array of strings): at least 2 hobbies
        - address (object): with city and country fields""",
        temperature=0.0,
        max_tokens=150,
        response_format={"type": "json_object"},
    )

    response = await provider.generate_unified(request)

    import json
    from jsonschema import validate

    # Define expected schema
    schema = {
        "type": "object",
        "required": ["name", "age", "hobbies", "address"],
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "number"},
            "hobbies": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2
            },
            "address": {
                "type": "object",
                "required": ["city", "country"],
                "properties": {
                    "city": {"type": "string"},
                    "country": {"type": "string"}
                }
            }
        }
    }

    parsed = json.loads(response.content)
    validate(instance=parsed, schema=schema)  # Raises if invalid

    # Additional specific checks
    assert parsed["name"].lower() == "alice"
    assert parsed["age"] == 30
```

#### Fix Test Isolation
```python
# Instead of modifying os.environ directly:
@pytest.fixture
def invalid_api_key(monkeypatch):
    """Safely set invalid API key for testing."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-invalid-test-key")
    yield
    # Automatically restored after test

@pytest.mark.asyncio
async def test_authentication_error(invalid_api_key):
    """Test auth error with proper isolation."""
    provider = OpenAIProvider()
    await provider.initialize()
    # ... rest of test
```

### 4. Add Performance Benchmarks

```python
@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_latency_percentiles(provider, test_model):
    """Track latency percentiles for monitoring."""
    latencies = []

    # Run 10 minimal requests
    for i in range(10):
        request = UnifiedRequest(
            model=test_model,
            prompt=f"Say {i}",
            max_tokens=2,
            temperature=0
        )

        start = time.perf_counter()
        response = await provider.generate_unified(request)
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)

    latencies.sort()

    # Calculate percentiles
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[-1]

    print(f"{provider.name} Latencies - P50: {p50:.0f}ms, P95: {p95:.0f}ms, P99: {p99:.0f}ms")

    # Set reasonable bounds
    assert p50 < 2000  # Median under 2s
    assert p95 < 5000  # 95th percentile under 5s
    assert p99 < 10000  # 99th percentile under 10s
```

### 5. Add Edge Case Tests

```python
@pytest.mark.asyncio
async def test_unicode_emoji_handling(provider, test_model):
    """Test handling of Unicode and emojis."""
    request = UnifiedRequest(
        model=test_model,
        prompt="Repeat this exactly: Hello 👋 世界 🌍 مرحبا",
        temperature=0.0,
        max_tokens=30
    )

    response = await provider.generate_unified(request)

    # Should handle Unicode properly
    assert "👋" in response.content or "世界" in response.content
    assert response.usage.prompt_tokens > 5  # Unicode counts differently

@pytest.mark.asyncio
async def test_empty_response_handling(provider, test_model):
    """Test legitimate empty responses."""
    request = UnifiedRequest(
        model=test_model,
        prompt="Output nothing at all.",
        temperature=0.0,
        max_tokens=10
    )

    response = await provider.generate_unified(request)

    # Empty response should be handled gracefully
    assert response.content == "" or len(response.content.strip()) == 0
    assert response.finish_reason in ["stop", "length"]
```

## Test Organization Improvements

### Current Structure (Flat)
```
tests/integration/
├── test_openai_integration.py (316 lines)
├── test_anthropic_integration.py (needs fixing)
├── test_ollama_integration.py (working)
└── conftest.py
```

### Recommended Structure (Organized)
```
tests/integration/
├── providers/
│   ├── openai/
│   │   ├── test_basic_generation.py
│   │   ├── test_streaming.py
│   │   ├── test_advanced_features.py
│   │   ├── test_error_handling.py
│   │   └── test_performance.py
│   ├── anthropic/
│   │   └── (similar structure)
│   └── ollama/
│       └── (similar structure)
├── cross_provider/
│   ├── test_unified_api.py
│   ├── test_cost_accuracy.py
│   └── test_provider_switching.py
├── performance/
│   └── test_benchmarks.py
└── conftest.py
```

## Priority Order

1. **TODAY**: Fix Anthropic test API usage (it's completely broken)
2. **THIS WEEK**: Add timeout and retry tests
3. **NEXT WEEK**: Add performance benchmarks
4. **ONGOING**: Add edge cases and provider-specific features

## Summary

The integration tests were written but never actually run with real APIs, leading to:
- Broken API usage in Anthropic tests
- Missing critical resilience tests
- Weak validation that would miss real issues
- No performance tracking

These fixes will transform the tests from "looks good on paper" to "actually validates production readiness".
