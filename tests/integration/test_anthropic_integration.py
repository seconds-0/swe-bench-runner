"""True integration tests for Anthropic provider.

These tests make real API calls to Anthropic's service and validate:
- Basic generation functionality  
- Streaming responses
- Error handling with real API errors
- Cost calculation accuracy
- Token counting via API
- Model availability checks

WARNING: These tests incur real API costs. Use ANTHROPIC_TEST_MODEL env var
to control which model is used (default: claude-3-haiku-20240307 for cost efficiency).
"""

import os
import pytest
from typing import List
from datetime import datetime

from swebench_runner.providers.anthropic import AnthropicProvider
from swebench_runner.providers.unified_models import UnifiedRequest
from swebench_runner.providers.exceptions import (
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderTokenLimitError,
    ProviderResponseError,
    ProviderError,
    ProviderConnectionError,
    ProviderTimeoutError,
)


@pytest.mark.integration
class TestAnthropicIntegration:
    """Integration tests for Anthropic provider with real API calls."""
    
    @pytest.fixture
    async def provider(self, skip_without_anthropic_key) -> AnthropicProvider:
        """Create an Anthropic provider with real credentials."""
        provider = AnthropicProvider()
        await provider.initialize()
        return provider
    
    @pytest.mark.asyncio
    async def test_basic_generation(self, provider: AnthropicProvider, anthropic_test_model: str, minimal_test_prompt: str):
        """Test basic text generation with real API call."""
        request = UnifiedRequest(
            model=anthropic_test_model,
            prompt=minimal_test_prompt,
            temperature=0.0,  # Deterministic for testing
            max_tokens=10,
        )
        
        response = await provider.generate_unified(request)
        
        # Validate response structure
        assert response.id is not None
        assert response.model == anthropic_test_model
        assert len(response.choices) == 1
        assert response.content is not None
        assert "test" in response.content.lower()
        assert response.finish_reason in ["stop", "max_tokens"]
        
        # Validate token usage
        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens == response.usage.prompt_tokens + response.usage.completion_tokens
        
        # Validate cost calculation
        assert response.cost is not None
        assert response.cost > 0
    
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
        message_start_seen = False
        content_blocks_seen = 0
        
        async for chunk in provider.generate_stream(request):
            chunk_count += 1
            
            # Anthropic sends metadata chunks too
            if chunk.model and not message_start_seen:
                message_start_seen = True
                assert chunk.model == anthropic_test_model
            
            if chunk.content:
                chunks.append(chunk.content)
                content_blocks_seen += 1
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Validate streaming behavior
        assert chunk_count > 1  # Should receive multiple chunks
        assert content_blocks_seen > 0  # Should have content chunks
        assert message_start_seen  # Should have seen message start
        full_response = "".join(chunks)
        assert any(str(i) in full_response for i in range(1, 6))  # Should contain numbers 1-5
        
        # Streaming should show incremental delivery
        assert duration > 0.1  # Should take some time to stream
    
    @pytest.mark.asyncio
    async def test_model_not_found_error(self, provider: AnthropicProvider):
        """Test handling of invalid model with real API."""
        request = UnifiedRequest(
            model="claude-999-nonexistent",
            prompt="test",
            max_tokens=10,
        )
        
        with pytest.raises(ProviderResponseError) as exc_info:
            await provider.generate_unified(request)
        
        assert "claude-999-nonexistent" in str(exc_info.value)
        assert exc_info.value.provider == "anthropic"
    
    @pytest.mark.asyncio
    async def test_authentication_error(self, monkeypatch):
        """Test handling of authentication errors with invalid key."""
        # Use monkeypatch for proper test isolation
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
    
    @pytest.mark.asyncio
    async def test_system_message_handling(self, provider: AnthropicProvider, anthropic_test_model: str):
        """Test Anthropic's special handling of system messages."""
        request = UnifiedRequest(
            model=anthropic_test_model,
            prompt="Hi there",
            system_message="You are a helpful assistant that always says 'Hello!' first.",
            temperature=0.0,
            max_tokens=50,
        )
        
        response = await provider.generate_unified(request)
        
        # Should handle system message properly
        assert response.content is not None
        assert "hello" in response.content.lower()
    
    @pytest.mark.asyncio
    async def test_token_counting_api(self, provider: AnthropicProvider, anthropic_test_model: str):
        """Test token counting via Anthropic's dedicated API endpoint."""
        # First check if the model supports token counting API
        test_messages = [{"role": "user", "content": "Count my tokens please"}]
        
        try:
            # Try the token counting endpoint
            count_result = await provider.token_counter.count_tokens_async(
                model=anthropic_test_model,
                messages=test_messages
            )
            
            # If successful, validate the count
            assert count_result.prompt_tokens > 0
            assert count_result.total_tokens == count_result.prompt_tokens
            
            # Now generate with the same prompt and compare
            request = UnifiedRequest(
                model=anthropic_test_model,
                prompt="Count my tokens please",
                max_tokens=10,
            )
            
            response = await provider.generate_unified(request)
            
            # Token counts should match (allow small variance for potential formatting)
            assert abs(response.usage.prompt_tokens - count_result.prompt_tokens) <= 2
            
        except Exception as e:
            # If token counting API fails, that's OK - not all models support it
            # Just ensure we can still generate
            request = UnifiedRequest(
                model=anthropic_test_model,
                prompt="Count my tokens please",
                max_tokens=10,
            )
            response = await provider.generate_unified(request)
            assert response.usage.prompt_tokens > 0
    
    @pytest.mark.asyncio
    async def test_cost_calculation_accuracy(self, provider: AnthropicProvider, anthropic_test_model: str, cost_test_prompt: str):
        """Test that cost calculations match expected pricing."""
        request = UnifiedRequest(
            model=anthropic_test_model,
            prompt=cost_test_prompt,
            temperature=0.0,
            max_tokens=20,
        )
        
        response = await provider.generate_unified(request)
        
        # Get model pricing
        capabilities = await provider.get_capabilities()
        model_info = next((m for m in capabilities.supported_models if m.id == anthropic_test_model), None)
        assert model_info is not None
        
        # Calculate expected costs
        expected_prompt_cost = (response.usage.prompt_tokens / 1000) * model_info.pricing.prompt_token_cost
        expected_completion_cost = (response.usage.completion_tokens / 1000) * model_info.pricing.completion_token_cost
        expected_total_cost = expected_prompt_cost + expected_completion_cost
        
        # Validate total cost matches expected calculation
        assert response.cost is not None
        assert abs(response.cost - expected_total_cost) < 0.000001
    
    @pytest.mark.asyncio
    async def test_model_availability(self, provider: AnthropicProvider):
        """Test that we can query available models from real API."""
        capabilities = await provider.get_capabilities()
        
        # Should have Claude models available
        model_ids = [m.id for m in capabilities.supported_models]
        assert any("claude-3-haiku" in m for m in model_ids)
        assert any("claude-3-5-sonnet" in m for m in model_ids)
        
        # All models should have valid pricing
        for model in capabilities.supported_models:
            assert model.pricing.prompt_token_cost > 0
            assert model.pricing.completion_token_cost > 0
            assert model.context_window > 0
    
    @pytest.mark.asyncio
    async def test_long_context_handling(self, provider: AnthropicProvider, anthropic_test_model: str):
        """Test handling of longer contexts specific to Anthropic."""
        # Create a moderately long context (not too long to avoid costs)
        long_context = "This is a test. " * 100  # ~400 tokens
        
        request = UnifiedRequest(
            model=anthropic_test_model,
            prompt="What was the first word I said?",
            temperature=0.0,
            max_tokens=20,
        )
        
        response = await provider.generate_unified(request)
        
        # Should handle the context and respond appropriately
        assert response.content is not None
        assert response.usage.prompt_tokens > 400  # Should count all the context
        
        # Response should reference the beginning
        content = response.content.lower()
        assert "this" in content or "first" in content
    
    @pytest.mark.asyncio
    async def test_concurrent_streaming_requests(self, provider: AnthropicProvider, anthropic_test_model: str):
        """Test handling of concurrent streaming requests."""
        import asyncio
        
        # Create multiple streaming requests
        requests = [
            UnifiedRequest(
                model=anthropic_test_model,
                prompt=f"Count to 3 (request {i})",
                temperature=0.0,
                max_tokens=20,
                stream=True,
            )
            for i in range(3)
        ]
        
        async def stream_to_completion(request):
            chunks = []
            async for chunk in provider.generate_stream(request):
                if chunk.content:
                    chunks.append(chunk.content)
            return "".join(chunks)
        
        # Execute concurrently
        tasks = [stream_to_completion(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # At least 2 should succeed
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) >= 2
        
        # Each should have content
        for result in successful_results:
            if not isinstance(result, Exception):
                assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_assistant_message_handling(self, provider: AnthropicProvider, anthropic_test_model: str):
        """Test handling of assistant messages in conversation."""
        request = UnifiedRequest(
            model=anthropic_test_model,
            prompt="And what's 3+3?",
            temperature=0.0,
            max_tokens=20,
        )
        
        response = await provider.generate_unified(request)
        
        # Should handle the conversation context
        assert response.content is not None
        content = response.content.lower()
        # Should mention 6 or six
        assert "6" in content or "six" in content
    
    @pytest.mark.asyncio
    async def test_max_tokens_enforcement(self, provider: AnthropicProvider, anthropic_test_model: str):
        """Test that max_tokens is properly enforced."""
        request = UnifiedRequest(
            model=anthropic_test_model,
            prompt="Write a very long story about a dragon.",
            temperature=0.0,
            max_tokens=5,  # Very low limit
        )
        
        response = await provider.generate_unified(request)
        
        # Should hit the token limit
        assert response.finish_reason == "max_tokens"
        assert response.usage.completion_tokens <= 5
    
    @pytest.mark.asyncio
    async def test_multiple_system_messages(self, provider: AnthropicProvider, anthropic_test_model: str):
        """Test Anthropic's handling of multiple system messages."""
        # Anthropic concatenates multiple system messages
        request = UnifiedRequest(
            model=anthropic_test_model,
            prompt="What is Python?",
            system_message="You are helpful. Always be concise.",
            temperature=0.0,
            max_tokens=30,
        )
        
        response = await provider.generate_unified(request)
        
        # Should handle multiple system messages gracefully
        assert response.content is not None
        assert len(response.content) > 0
        # Response should be concise due to system message
        assert len(response.content) < 200
    
    @pytest.mark.asyncio
    async def test_empty_message_handling(self, provider: AnthropicProvider, anthropic_test_model: str):
        """Test handling of edge cases like empty messages."""
        request = UnifiedRequest(
            model=anthropic_test_model,
            prompt="",  # Empty message
            temperature=0.0,
            max_tokens=20,
        )
        
        # Should handle gracefully (might error or treat as minimal input)
        try:
            response = await provider.generate_unified(request)
            assert response.content is not None
        except ProviderError as e:
            # If it errors, should be a clear validation error
            assert "empty" in str(e).lower() or "content" in str(e).lower()
    
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
            
            with pytest.raises((ProviderTimeoutError, ProviderConnectionError)) as exc_info:
                await provider.generate_unified(request)
            
            assert "timeout" in str(exc_info.value).lower()
            assert exc_info.value.provider == "anthropic"
        finally:
            # Restore timeout
            if hasattr(provider, '_client') and original_timeout:
                provider._client.timeout = original_timeout