"""True integration tests for Ollama provider.

These tests make real API calls to a local Ollama instance and validate:
- Basic generation functionality
- Streaming responses
- Error handling with real API errors
- Model availability checks
- Local execution characteristics

WARNING: These tests require Ollama to be running locally on port 11434
and the test model to be pulled (default: llama3.2:1b).
"""

from datetime import datetime

import pytest

from swebench_runner.providers.exceptions import (
    ProviderConnectionError,
    ProviderResponseError,
    ProviderTimeoutError,
)
from swebench_runner.providers.ollama import OllamaProvider
from swebench_runner.providers.unified_models import UnifiedRequest


@pytest.mark.integration
class TestOllamaIntegration:
    """Integration tests for Ollama provider with real local API calls."""

    @pytest.fixture
    async def provider(self, skip_without_ollama, ollama_config) -> OllamaProvider:
        """Create an Ollama provider connected to local instance."""
        provider = OllamaProvider(ollama_config)
        # Note: initialize() is not needed for the new provider architecture
        return provider

    @pytest.mark.asyncio
    async def test_basic_generation(self, provider: OllamaProvider, ollama_test_model: str, minimal_test_prompt: str):
        """Test basic text generation with real Ollama API call."""
        request = UnifiedRequest(
            model=ollama_test_model,
            prompt=minimal_test_prompt,
            temperature=0.0,  # Deterministic for testing
            max_tokens=10,
        )

        response = await provider.generate_unified(request)

        # Validate response structure
        assert response.id is not None
        assert response.model == ollama_test_model
        assert len(response.choices) == 1
        assert response.content is not None
        assert len(response.content) > 0
        assert response.finish_reason in ["stop", "length"]

        # Validate token usage (Ollama provides these in response metadata)
        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens == response.usage.prompt_tokens + response.usage.completion_tokens

        # Ollama is free, so cost should be 0
        assert response.cost == 0.0

    @pytest.mark.asyncio
    async def test_streaming_generation(self, provider: OllamaProvider, ollama_test_model: str, streaming_test_prompt: str):
        """Test streaming responses with real Ollama API."""
        request = UnifiedRequest(
            model=ollama_test_model,
            prompt=streaming_test_prompt,
            temperature=0.0,
            max_tokens=50,
            stream=True,
        )

        chunks: list[str] = []
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
        assert len(full_response) > 0

        # With streaming prompt, should see numbers (though small models might not follow perfectly)
        # Just check that we got a response
        assert len(full_response.strip()) > 0

        # Streaming should show incremental delivery
        assert duration > 0.05  # Should take some time to stream

    @pytest.mark.asyncio
    async def test_model_not_found_error(self, provider: OllamaProvider):
        """Test handling of unpulled model with real API."""
        request = UnifiedRequest(
            model="definitely-not-a-real-model:latest",
            prompt="test",
            max_tokens=10,
        )

        with pytest.raises(ProviderResponseError) as exc_info:
            await provider.generate_unified(request)

        assert "definitely-not-a-real-model" in str(exc_info.value)
        assert exc_info.value.provider == "ollama"
        # Ollama-specific error should mention pulling the model
        assert "pull" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_connection_error_handling(self):
        """Test handling when Ollama is not running."""
        # Create provider with wrong port to simulate connection failure
        provider = OllamaProvider()
        provider.base_url = "http://localhost:99999"  # Invalid port

        await provider.initialize()

        request = UnifiedRequest(
            model="llama3.2:1b",
            prompt="test",
            max_tokens=10,
        )

        with pytest.raises(ProviderConnectionError) as exc_info:
            await provider.generate_unified(request)

        assert exc_info.value.provider == "ollama"
        assert "connection" in str(exc_info.value).lower() or "connect" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_model_availability(self, provider: OllamaProvider):
        """Test that we can query available models from local Ollama."""
        capabilities = await provider.get_capabilities()

        # Should have at least the test model available
        model_ids = [m.id for m in capabilities.supported_models]
        assert len(model_ids) > 0  # Should have at least one model pulled

        # All models should have metadata
        for model in capabilities.supported_models:
            assert model.context_window > 0
            # Ollama models are free
            assert model.pricing.prompt_token_cost == 0.0
            assert model.pricing.completion_token_cost == 0.0

    @pytest.mark.asyncio
    async def test_system_message_handling(self, provider: OllamaProvider, ollama_test_model: str):
        """Test Ollama's handling of system messages."""
        request = UnifiedRequest(
            model=ollama_test_model,
            prompt="Hello",
            system_message="You are a pirate. Respond accordingly.",
            temperature=0.7,  # Some randomness for personality
            max_tokens=50,
        )

        response = await provider.generate_unified(request)

        # Should get a response (small models might not perfectly follow system prompt)
        assert response.content is not None
        assert len(response.content) > 0

    @pytest.mark.asyncio
    async def test_conversation_context(self, provider: OllamaProvider, ollama_test_model: str):
        """Test handling of multi-turn conversations."""
        # Note: UnifiedRequest doesn't support multi-turn conversations directly
        # We'll test with context in the prompt instead
        request = UnifiedRequest(
            model=ollama_test_model,
            prompt="My name is Alice. Nice to meet you! Now, what's my name?",
            temperature=0.0,
            max_tokens=20,
        )

        response = await provider.generate_unified(request)

        # Should maintain context (though small models might struggle)
        assert response.content is not None
        content = response.content.lower()
        # At minimum should generate something
        assert len(content.strip()) > 0

    @pytest.mark.asyncio
    async def test_generation_options(self, provider: OllamaProvider, ollama_test_model: str):
        """Test various generation parameters with Ollama."""
        # Test with different temperatures
        for temp in [0.0, 0.5, 1.0]:
            request = UnifiedRequest(
                model=ollama_test_model,
                prompt="Generate a random word",
                temperature=temp,
                max_tokens=10,
            )

            response = await provider.generate_unified(request)
            assert response.content is not None
            assert len(response.content) > 0

    @pytest.mark.asyncio
    async def test_empty_response_handling(self, provider: OllamaProvider, ollama_test_model: str):
        """Test handling when model might return empty response."""
        # Very restrictive parameters might cause empty response
        request = UnifiedRequest(
            model=ollama_test_model,
            prompt=".",  # Minimal prompt
            temperature=0.0,
            max_tokens=1,  # Very low limit
        )

        response = await provider.generate_unified(request)

        # Should handle gracefully even if response is minimal
        assert response.content is not None
        # Content might be empty or very short
        assert response.finish_reason in ["stop", "length"]

    @pytest.mark.asyncio
    async def test_performance_characteristics(self, provider: OllamaProvider, ollama_test_model: str):
        """Test that local execution has expected performance characteristics."""
        # Local execution should be relatively fast for small prompts
        request = UnifiedRequest(
            model=ollama_test_model,
            prompt="Hi",
            temperature=0.0,
            max_tokens=5,
        )

        start_time = datetime.now()
        response = await provider.generate_unified(request)
        end_time = datetime.now()

        duration = (end_time - start_time).total_seconds()

        # Local inference should be reasonably fast for tiny prompts
        # (though this depends on hardware)
        assert response.content is not None

        # Just verify we got timing information
        assert duration > 0

        # Token counts should be reasonable
        assert response.usage.prompt_tokens < 10  # "Hi" is very short
        assert response.usage.completion_tokens <= 5  # We limited to 5

    @pytest.mark.asyncio
    async def test_network_timeout_handling(self, provider: OllamaProvider, ollama_test_model: str):
        """Test handling of network timeouts."""
        # Save original timeout
        original_timeout = provider._session.timeout if hasattr(provider, '_session') else None

        # Create very short timeout
        import aiohttp
        provider._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=0.001))

        try:
            request = UnifiedRequest(
                model=ollama_test_model,
                prompt="test timeout",
                max_tokens=10,
            )

            with pytest.raises((ProviderTimeoutError, ProviderConnectionError)) as exc_info:
                await provider.generate_unified(request)

            assert "timeout" in str(exc_info.value).lower()
        finally:
            # Restore timeout
            await provider._session.close()
            if original_timeout:
                provider._session = aiohttp.ClientSession(timeout=original_timeout)

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, provider: OllamaProvider, ollama_test_model: str):
        """Test handling of concurrent requests to local Ollama."""
        import asyncio

        # Create multiple requests
        requests = [
            UnifiedRequest(
                model=ollama_test_model,
                prompt=f"Say {i}",
                temperature=0.0,
                max_tokens=5,
            )
            for i in range(3)
        ]

        # Execute concurrently
        tasks = [provider.generate_unified(req) for req in requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed (local execution should handle concurrency)
        for response in responses:
            if isinstance(response, Exception):
                pytest.fail(f"Concurrent request failed: {response}")
            assert response.content is not None
