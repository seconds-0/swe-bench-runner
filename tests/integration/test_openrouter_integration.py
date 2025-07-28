"""True integration tests for OpenRouter provider.

These tests make real API calls to OpenRouter's service and validate:
- Basic generation functionality
- Multi-provider model access
- Streaming responses
- Error handling with real API errors
- Cost calculation accuracy
- Model availability checks

WARNING: These tests incur real API costs. Use OPENROUTER_TEST_MODEL env var
to control which model is used (default: anthropic/claude-3-haiku for cost efficiency).
"""

import os
from datetime import datetime

import pytest

from swebench_runner.providers.exceptions import (
    ProviderAuthenticationError,
    ProviderResponseError,
)
from swebench_runner.providers.openrouter import OpenRouterProvider
from swebench_runner.providers.unified_models import UnifiedRequest


@pytest.mark.integration
class TestOpenRouterIntegration:
    """Integration tests for OpenRouter provider with real API calls."""

    @pytest.fixture
    def skip_without_openrouter_key(self):
        """Skip test if OpenRouter API key is not available."""
        if not os.environ.get("OPENROUTER_API_KEY"):
            pytest.skip("OPENROUTER_API_KEY not set - skipping OpenRouter integration test")

    @pytest.fixture
    def openrouter_test_model(self) -> str:
        """Get the OpenRouter model to use for testing."""
        # Default to Claude Haiku for cost efficiency
        return os.environ.get("OPENROUTER_TEST_MODEL", "anthropic/claude-3-haiku")

    @pytest.fixture
    async def provider(self, skip_without_openrouter_key) -> OpenRouterProvider:
        """Create an OpenRouter provider with real credentials."""
        provider = OpenRouterProvider()
        await provider.initialize()
        return provider

    @pytest.mark.asyncio
    async def test_basic_generation(self, provider: OpenRouterProvider, openrouter_test_model: str):
        """Test basic text generation with real API call."""
        request = UnifiedRequest(
            model=openrouter_test_model,
            prompt="Say 'test' and nothing else.",
            temperature=0.0,
            max_tokens=10,
        )

        response = await provider.generate_unified(request)

        # Validate response structure
        assert response.content is not None
        assert "test" in response.content.lower()
        assert response.model == openrouter_test_model
        assert response.finish_reason in ["stop", "max_tokens", "length"]

        # Validate token usage
        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens == response.usage.prompt_tokens + response.usage.completion_tokens

        # OpenRouter always provides costs
        assert response.cost is not None
        assert response.cost > 0

    @pytest.mark.asyncio
    async def test_streaming_generation(self, provider: OpenRouterProvider, openrouter_test_model: str):
        """Test streaming responses with real API."""
        request = UnifiedRequest(
            model=openrouter_test_model,
            prompt="Count from 1 to 5, one number per line.",
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

        # Should contain at least some numbers
        assert any(str(i) in full_response for i in range(1, 6))

        # Streaming should show incremental delivery
        assert duration > 0.1  # Should take some time to stream

    @pytest.mark.asyncio
    async def test_model_not_found_error(self, provider: OpenRouterProvider):
        """Test handling of invalid model with real API."""
        request = UnifiedRequest(
            model="invalid/model-that-does-not-exist",
            prompt="test",
            max_tokens=10,
        )

        with pytest.raises(ProviderResponseError) as exc_info:
            await provider.generate_unified(request)

        assert "invalid/model-that-does-not-exist" in str(exc_info.value)
        assert exc_info.value.provider == "openrouter"

    @pytest.mark.asyncio
    async def test_authentication_error(self):
        """Test handling of authentication errors with invalid key."""
        # Temporarily set invalid key
        original_key = os.environ.get("OPENROUTER_API_KEY")
        os.environ["OPENROUTER_API_KEY"] = "sk-or-invalid-test-key-12345"

        try:
            provider = OpenRouterProvider()
            await provider.initialize()

            request = UnifiedRequest(
                model="anthropic/claude-3-haiku",
                prompt="test",
                max_tokens=10,
            )

            with pytest.raises(ProviderAuthenticationError) as exc_info:
                await provider.generate_unified(request)

            assert exc_info.value.provider == "openrouter"
            assert "authentication" in str(exc_info.value).lower() or "unauthorized" in str(exc_info.value).lower()

        finally:
            # Restore original key
            if original_key:
                os.environ["OPENROUTER_API_KEY"] = original_key
            else:
                del os.environ["OPENROUTER_API_KEY"]

    @pytest.mark.asyncio
    async def test_multi_provider_models(self, provider: OpenRouterProvider):
        """Test accessing models from different providers through OpenRouter."""
        # Test with different provider models (if available in account)
        test_models = [
            ("anthropic/claude-3-haiku", "anthropic"),
            ("meta-llama/llama-3-8b-instruct", "meta"),
            ("mistralai/mistral-7b-instruct", "mistral"),
        ]

        successful_providers = []

        for model, provider_name in test_models:
            try:
                request = UnifiedRequest(
                    model=model,
                    prompt="Say 'hello'",
                    temperature=0.0,
                    max_tokens=10,
                )

                response = await provider.generate_unified(request)

                if response.content:
                    successful_providers.append(provider_name)
                    assert response.model == model
                    assert response.cost > 0

            except ProviderResponseError:
                # Model might not be available in the account
                pass
            except Exception as e:
                # Log but don't fail - model availability varies
                print(f"Model {model} failed: {e}")

        # Should be able to access at least one provider's model
        assert len(successful_providers) >= 1

    @pytest.mark.asyncio
    async def test_cost_tracking(self, provider: OpenRouterProvider, openrouter_test_model: str):
        """Test that OpenRouter provides accurate cost information."""
        request = UnifiedRequest(
            model=openrouter_test_model,
            prompt="Write exactly 10 words about the weather.",
            temperature=0.0,
            max_tokens=30,
        )

        response = await provider.generate_unified(request)

        # OpenRouter always provides detailed costs
        assert response.cost is not None
        assert response.cost > 0

        # Cost should be reasonable (not astronomical)
        assert response.cost < 0.01  # Less than 1 cent for this small request

        # Usage should be tracked
        assert response.usage.prompt_tokens > 5
        assert response.usage.completion_tokens > 5

    @pytest.mark.asyncio
    async def test_model_capabilities(self, provider: OpenRouterProvider):
        """Test querying available models and their capabilities."""
        capabilities = await provider.get_capabilities()

        # Should have models available
        assert len(capabilities.supported_models) > 0

        # Check model information
        model_names = [m.id for m in capabilities.supported_models]

        # Should include some popular models
        assert any("claude" in m for m in model_names)  # Anthropic models
        assert any("gpt" in m for m in model_names) or any("openai" in m for m in model_names)  # OpenAI models

        # All models should have pricing
        for model in capabilities.supported_models:
            assert model.pricing.prompt_token_cost >= 0
            assert model.pricing.completion_token_cost >= 0
            assert model.context_window > 0

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, provider: OpenRouterProvider, openrouter_test_model: str):
        """Test handling of concurrent requests."""
        import asyncio

        # Create multiple requests
        requests = [
            UnifiedRequest(
                model=openrouter_test_model,
                prompt=f"Say 'test {i}' and nothing else.",
                temperature=0.0,
                max_tokens=10,
            )
            for i in range(3)
        ]

        # Execute concurrently
        tasks = [provider.generate_unified(req) for req in requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed
        successful_responses = [r for r in responses if not isinstance(r, Exception)]
        assert len(successful_responses) == 3

        # Each response should be valid
        for i, response in enumerate(successful_responses):
            assert response.content is not None
            assert response.cost > 0

    @pytest.mark.asyncio
    async def test_system_message_handling(self, provider: OpenRouterProvider, openrouter_test_model: str):
        """Test proper handling of system messages."""
        request = UnifiedRequest(
            model=openrouter_test_model,
            prompt="What's 2+2?",
            system_message="You are a calculator that always responds with just the number, nothing else.",
            temperature=0.0,
            max_tokens=10,
        )

        response = await provider.generate_unified(request)

        # Should get a response
        assert response.content is not None
        # Should contain the number 4
        assert "4" in response.content

    @pytest.mark.asyncio
    async def test_max_tokens_enforcement(self, provider: OpenRouterProvider, openrouter_test_model: str):
        """Test that max_tokens is properly enforced."""
        request = UnifiedRequest(
            model=openrouter_test_model,
            prompt="Count from 1 to 100 with detailed explanations for each number.",
            temperature=0.0,
            max_tokens=5,  # Very low limit
        )

        response = await provider.generate_unified(request)

        # Should hit the token limit
        assert response.finish_reason in ["max_tokens", "length"]
        # Should have used approximately the max tokens (some variance allowed)
        assert response.usage.completion_tokens <= 10  # Allow some overhead
