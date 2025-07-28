"""True integration tests for OpenAI provider.

These tests make real API calls to OpenAI's service and validate:
- Basic generation functionality
- Streaming responses
- Error handling with real API errors
- Cost calculation accuracy
- Token counting validation
- Model availability checks

WARNING: These tests incur real API costs. Use OPENAI_TEST_MODEL env var
to control which model is used (default: gpt-3.5-turbo for cost efficiency).
"""

import os
import pytest
from typing import List
from datetime import datetime

from swebench_runner.providers.openai import OpenAIProvider
from swebench_runner.providers.unified_models import UnifiedRequest
from swebench_runner.providers.exceptions import (
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderTokenLimitError,
    ProviderResponseError,
    ProviderConnectionError,
    ProviderTimeoutError,
)


@pytest.mark.integration
class TestOpenAIIntegration:
    """Integration tests for OpenAI provider with real API calls."""
    
    @pytest.fixture
    async def provider(self, skip_without_openai_key) -> OpenAIProvider:
        """Create an OpenAI provider with real credentials."""
        provider = OpenAIProvider()
        await provider.initialize()
        return provider
    
    @pytest.mark.asyncio
    async def test_basic_generation(self, provider: OpenAIProvider, openai_test_model: str, minimal_test_prompt: str):
        """Test basic text generation with real API call."""
        request = UnifiedRequest(
            model=openai_test_model,
            prompt=minimal_test_prompt,
            temperature=0.0,  # Deterministic for testing
            max_tokens=10,
        )
        
        response = await provider.generate_unified(request)
        
        # Validate response structure
        assert response.model == openai_test_model
        assert response.content is not None
        assert "test" in response.content.lower()
        assert response.finish_reason in ["stop", "length"]
        
        # Validate token usage
        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens == response.usage.prompt_tokens + response.usage.completion_tokens
        
        # Validate cost calculation
        assert response.cost is not None
        assert response.cost > 0
    
    @pytest.mark.asyncio
    async def test_streaming_generation(self, provider: OpenAIProvider, openai_test_model: str, streaming_test_prompt: str):
        """Test streaming responses with real API."""
        request = UnifiedRequest(
            model=openai_test_model,
            prompt=streaming_test_prompt,
            temperature=0.0,
            max_tokens=50,
            stream=True,
        )
        
        chunks: List[str] = []
        chunk_count = 0
        start_time = datetime.now()
        
        async for chunk in provider.generate_stream(request):
            chunk_count += 1
            if chunk.content:
                chunks.append(chunk.content)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Validate streaming behavior
        assert chunk_count > 1  # Should receive multiple chunks
        assert len(chunks) > 0  # Should have content
        full_response = "".join(chunks)
        assert any(str(i) in full_response for i in range(1, 6))  # Should contain numbers 1-5
        
        # Streaming should show incremental delivery (not just one chunk)
        assert duration > 0.1  # Should take some time to stream
    
    @pytest.mark.asyncio
    async def test_model_not_found_error(self, provider: OpenAIProvider):
        """Test handling of invalid model with real API."""
        request = UnifiedRequest(
            model="gpt-999-nonexistent",
            prompt="test",
            max_tokens=10,
        )
        
        with pytest.raises(ProviderResponseError) as exc_info:
            await provider.generate_unified(request)
        
        assert "gpt-999-nonexistent" in str(exc_info.value)
        assert exc_info.value.provider == "openai"
    
    @pytest.mark.asyncio
    async def test_authentication_error(self, monkeypatch):
        """Test handling of authentication errors with invalid key."""
        # Use monkeypatch for proper test isolation
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
    
    @pytest.mark.asyncio
    async def test_token_limit_handling(self, provider: OpenAIProvider, openai_test_model: str, error_test_prompt: str):
        """Test handling of token limit errors with real API."""
        # This test might not trigger a token limit error with newer models
        # but tests the error path if it does occur
        request = UnifiedRequest(
            model=openai_test_model,
            prompt=error_test_prompt,
            max_tokens=1,  # Very low to potentially trigger issues
        )
        
        try:
            response = await provider.generate_unified(request)
            # If no error, validate the response handled the large input
            assert response.finish_reason in ["stop", "length"]
        except ProviderTokenLimitError as e:
            # If we hit token limit, validate error handling
            assert e.provider == "openai"
            assert "token" in str(e).lower()
    
    @pytest.mark.asyncio
    async def test_cost_calculation_accuracy(self, provider: OpenAIProvider, openai_test_model: str, cost_test_prompt: str):
        """Test that cost calculations match expected pricing."""
        request = UnifiedRequest(
            model=openai_test_model,
            prompt=cost_test_prompt,
            temperature=0.0,
            max_tokens=20,
        )
        
        response = await provider.generate_unified(request)
        
        # Get model pricing
        capabilities = await provider.get_capabilities()
        model_info = next((m for m in capabilities.supported_models if m.id == openai_test_model), None)
        assert model_info is not None
        
        # Calculate expected costs
        expected_prompt_cost = (response.usage.prompt_tokens / 1000) * model_info.pricing.prompt_token_cost
        expected_completion_cost = (response.usage.completion_tokens / 1000) * model_info.pricing.completion_token_cost
        expected_total_cost = expected_prompt_cost + expected_completion_cost
        
        # Validate total cost matches calculation
        assert response.cost is not None
        assert abs(response.cost - expected_total_cost) < 0.000001
    
    @pytest.mark.asyncio
    async def test_model_availability(self, provider: OpenAIProvider):
        """Test that we can query available models from real API."""
        capabilities = await provider.get_capabilities()
        
        # Should have standard models available
        model_ids = [m.id for m in capabilities.supported_models]
        assert "gpt-3.5-turbo" in model_ids or "gpt-3.5-turbo-0125" in model_ids
        assert any("gpt-4" in m for m in model_ids)  # Some GPT-4 variant
        
        # All models should have valid pricing
        for model in capabilities.supported_models:
            assert model.pricing.prompt_token_cost > 0
            assert model.pricing.completion_token_cost > 0
            assert model.context_window > 0
    
    @pytest.mark.asyncio 
    async def test_circuit_breaker_recovery(self, provider: OpenAIProvider, openai_test_model: str, minimal_test_prompt: str):
        """Test that circuit breaker recovers after transient failures."""
        # First, make a successful request to ensure circuit is closed
        request = UnifiedRequest(
            model=openai_test_model,
            prompt=minimal_test_prompt,
            max_tokens=10,
        )
        
        response1 = await provider.generate_unified(request)
        assert response1.content is not None
        
        # Circuit breaker should still be closed, allowing more requests
        response2 = await provider.generate_unified(request)
        assert response2.content is not None
        
        # Note: Testing actual circuit breaker opening would require forcing
        # multiple failures, which we can't reliably do with a real API
        # without potentially getting rate limited or banned
    
    @pytest.mark.asyncio
    async def test_network_timeout_handling(self, provider: OpenAIProvider, openai_test_model: str):
        """Test handling of network timeouts."""
        # Save original timeout
        original_timeout = provider._client.timeout if hasattr(provider, '_client') else None
        
        # Set very short timeout
        if hasattr(provider, '_client'):
            provider._client.timeout = 0.001  # 1ms
        
        try:
            request = UnifiedRequest(
                model=openai_test_model,
                prompt="test timeout",
                max_tokens=10,
            )
            
            with pytest.raises((ProviderTimeoutError, ProviderConnectionError)) as exc_info:
                await provider.generate_unified(request)
            
            assert "timeout" in str(exc_info.value).lower()
            assert exc_info.value.provider == "openai"
        finally:
            # Restore timeout
            if hasattr(provider, '_client') and original_timeout:
                provider._client.timeout = original_timeout
    
    @pytest.mark.asyncio
    async def test_system_message_handling(self, provider: OpenAIProvider, openai_test_model: str):
        """Test proper handling of system messages."""
        request = UnifiedRequest(
            model=openai_test_model,
            prompt="Hello there!",
            system_message="You are a pirate. Always respond in pirate speak.",
            temperature=0.7,
            max_tokens=30,
        )
        
        response = await provider.generate_unified(request)
        
        # Should get a response (checking for pirate speak is unreliable)
        assert response.content is not None
        assert len(response.content) > 0
    
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
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, provider: OpenAIProvider, openai_test_model: str, minimal_test_prompt: str):
        """Test handling of concurrent requests."""
        import asyncio
        
        # Create multiple requests
        requests = [
            UnifiedRequest(
                model=openai_test_model,
                prompt=f"{minimal_test_prompt} (request {i})",
                temperature=0.0,
                max_tokens=10,
            )
            for i in range(3)
        ]
        
        # Execute concurrently
        tasks = [provider.generate_unified(req) for req in requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should succeed or fail gracefully
        successful_responses = [r for r in responses if not isinstance(r, Exception)]
        assert len(successful_responses) >= 2  # At least 2 should succeed
        
        # Check responses are valid
        for response in successful_responses:
            if not isinstance(response, Exception):
                assert response.content is not None
    
    @pytest.mark.asyncio
    async def test_max_tokens_enforcement(self, provider: OpenAIProvider, openai_test_model: str):
        """Test that max_tokens is properly enforced."""
        request = UnifiedRequest(
            model=openai_test_model,
            prompt="Count from 1 to 100 slowly with detailed explanations.",
            temperature=0.0,
            max_tokens=5,  # Very low limit
        )
        
        response = await provider.generate_unified(request)
        
        # Should hit the token limit
        assert response.finish_reason == "length"
        assert response.usage.completion_tokens <= 5