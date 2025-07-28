"""Tests for unified data models."""

import pytest

from swebench_runner.providers.unified_models import (
    FinishReason,
    TokenUsage,
    UnifiedRequest,
    UnifiedResponse,
)


class TestTokenUsage:
    """Test TokenUsage data model."""

    def test_token_usage_basic(self):
        """Test basic TokenUsage creation."""
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )

        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150

    def test_token_usage_auto_calculate_total(self):
        """Test automatic total_tokens calculation when zero."""
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=0  # Will be auto-calculated
        )

        assert usage.total_tokens == 150

    def test_token_usage_preserve_explicit_total(self):
        """Test that explicit total_tokens is preserved when non-zero."""
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=200  # Explicit value should be kept
        )

        assert usage.total_tokens == 200


class TestUnifiedRequest:
    """Test UnifiedRequest data model."""

    def test_unified_request_basic(self):
        """Test basic UnifiedRequest creation."""
        request = UnifiedRequest(prompt="Test prompt")

        assert request.prompt == "Test prompt"
        assert request.system_message is None
        assert request.max_tokens is None
        assert request.temperature == 0.7
        assert request.stream is False
        assert request.model is None
        assert request.stop_sequences is None

    def test_unified_request_full(self):
        """Test UnifiedRequest with all parameters."""
        request = UnifiedRequest(
            prompt="Test prompt",
            system_message="You are a helpful assistant",
            max_tokens=1000,
            temperature=0.8,
            stream=True,
            model="gpt-4",
            stop_sequences=["###", "DONE"]
        )

        assert request.prompt == "Test prompt"
        assert request.system_message == "You are a helpful assistant"
        assert request.max_tokens == 1000
        assert request.temperature == 0.8
        assert request.stream is True
        assert request.model == "gpt-4"
        assert request.stop_sequences == ["###", "DONE"]

    def test_unified_request_empty_prompt_validation(self):
        """Test that empty prompt raises ValueError."""
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            UnifiedRequest(prompt="")

    def test_unified_request_temperature_validation(self):
        """Test temperature validation."""
        # Temperature too low
        with pytest.raises(ValueError, match="Temperature must be between 0 and 2"):
            UnifiedRequest(prompt="test", temperature=-0.1)

        # Temperature too high
        with pytest.raises(ValueError, match="Temperature must be between 0 and 2"):
            UnifiedRequest(prompt="test", temperature=2.1)

        # Valid temperatures should work
        request1 = UnifiedRequest(prompt="test", temperature=0.0)
        assert request1.temperature == 0.0

        request2 = UnifiedRequest(prompt="test", temperature=2.0)
        assert request2.temperature == 2.0

    def test_unified_request_max_tokens_validation(self):
        """Test max_tokens validation."""
        # Negative max_tokens should fail
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            UnifiedRequest(prompt="test", max_tokens=-1)

        # Zero max_tokens should fail
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            UnifiedRequest(prompt="test", max_tokens=0)

        # Positive max_tokens should work
        request = UnifiedRequest(prompt="test", max_tokens=100)
        assert request.max_tokens == 100


class TestUnifiedResponse:
    """Test UnifiedResponse data model."""

    def test_unified_response_basic(self):
        """Test basic UnifiedResponse creation."""
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        response = UnifiedResponse(
            content="Generated response",
            model="gpt-4",
            usage=usage,
            latency_ms=1500,
            finish_reason="stop",
            provider="openai"
        )

        assert response.content == "Generated response"
        assert response.model == "gpt-4"
        assert response.usage == usage
        assert response.latency_ms == 1500
        assert response.finish_reason == "stop"
        assert response.provider == "openai"
        assert response.cost is None
        assert response.raw_response == {}

    def test_unified_response_full(self):
        """Test UnifiedResponse with all parameters."""
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        raw_response = {"choices": [{"message": {"content": "test"}}]}

        response = UnifiedResponse(
            content="Generated response",
            model="gpt-4",
            usage=usage,
            latency_ms=1500,
            finish_reason="stop",
            provider="openai",
            cost=0.03,
            raw_response=raw_response
        )

        assert response.cost == 0.03
        assert response.raw_response == raw_response

    def test_unified_response_negative_latency_fix(self):
        """Test that negative latency is fixed to 0."""
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        response = UnifiedResponse(
            content="test",
            model="gpt-4",
            usage=usage,
            latency_ms=-100,  # Invalid negative latency
            finish_reason="stop",
            provider="openai"
        )

        assert response.latency_ms == 0

    def test_unified_response_empty_content_allowed(self):
        """Test that empty content is allowed in some cases."""
        usage = TokenUsage(prompt_tokens=100, completion_tokens=0, total_tokens=100)

        # Empty content with "length" finish reason should be allowed
        response = UnifiedResponse(
            content="",
            model="gpt-4",
            usage=usage,
            latency_ms=1000,
            finish_reason="length",
            provider="openai"
        )

        assert response.content == ""

        # Empty content with other finish reasons should also be allowed
        # (the validation allows empty content for edge cases)
        response2 = UnifiedResponse(
            content="",
            model="gpt-4",
            usage=usage,
            latency_ms=1000,
            finish_reason="stop",
            provider="openai"
        )

        assert response2.content == ""


class TestFinishReason:
    """Test FinishReason enum and normalization."""

    def test_finish_reason_values(self):
        """Test that FinishReason enum has expected values."""
        assert FinishReason.STOP.value == "stop"
        assert FinishReason.LENGTH.value == "length"
        assert FinishReason.TOOL_CALLS.value == "tool_calls"
        assert FinishReason.CONTENT_FILTER.value == "content_filter"
        assert FinishReason.ERROR.value == "error"

    def test_normalize_openai_reasons(self):
        """Test normalization of OpenAI finish reasons."""
        assert FinishReason.normalize("stop", "openai") == "stop"
        assert FinishReason.normalize("length", "openai") == "length"
        assert FinishReason.normalize("tool_calls", "openai") == "tool_calls"
        assert FinishReason.normalize("content_filter", "openai") == "content_filter"

        # Unknown OpenAI reason should default to "stop"
        assert FinishReason.normalize("unknown_reason", "openai") == "stop"

    def test_normalize_anthropic_reasons(self):
        """Test normalization of Anthropic finish reasons."""
        assert FinishReason.normalize("end_turn", "anthropic") == "stop"
        assert FinishReason.normalize("max_tokens", "anthropic") == "length"
        assert FinishReason.normalize("tool_use", "anthropic") == "tool_calls"

        # Unknown Anthropic reason should default to "stop"
        assert FinishReason.normalize("unknown_reason", "anthropic") == "stop"

    def test_normalize_ollama_reasons(self):
        """Test normalization of Ollama finish reasons."""
        assert FinishReason.normalize("stop", "ollama") == "stop"
        assert FinishReason.normalize("length", "ollama") == "length"

        # Unknown Ollama reason should default to "stop"
        assert FinishReason.normalize("unknown_reason", "ollama") == "stop"

    def test_normalize_unknown_provider(self):
        """Test normalization for unknown provider."""
        # Unknown provider should default to "stop"
        assert FinishReason.normalize("any_reason", "unknown_provider") == "stop"

    def test_normalize_comprehensive_mapping(self):
        """Test comprehensive mapping across all providers."""
        # Test all expected mappings
        test_cases = [
            # OpenAI
            ("stop", "openai", "stop"),
            ("length", "openai", "length"),
            ("tool_calls", "openai", "tool_calls"),
            ("content_filter", "openai", "content_filter"),

            # Anthropic
            ("end_turn", "anthropic", "stop"),
            ("max_tokens", "anthropic", "length"),
            ("tool_use", "anthropic", "tool_calls"),

            # Ollama
            ("stop", "ollama", "stop"),
            ("length", "ollama", "length"),

            # Unknown cases
            ("unknown", "openai", "stop"),
            ("unknown", "anthropic", "stop"),
            ("unknown", "ollama", "stop"),
            ("anything", "unknown_provider", "stop"),
        ]

        for provider_reason, provider, expected in test_cases:
            result = FinishReason.normalize(provider_reason, provider)
            assert result == expected, f"Failed for {provider_reason}/{provider}: expected {expected}, got {result}"


class TestIntegration:
    """Integration tests for unified models."""

    def test_complete_workflow(self):
        """Test a complete workflow using all unified models."""
        # Create a request
        request = UnifiedRequest(
            prompt="Write a Python function to calculate fibonacci",
            system_message="You are a Python expert",
            max_tokens=500,
            temperature=0.2,
            model="gpt-4"
        )

        # Simulate processing and create response
        usage = TokenUsage(
            prompt_tokens=25,
            completion_tokens=75,
            total_tokens=0  # Will be auto-calculated
        )

        response = UnifiedResponse(
            content="def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            model="gpt-4",
            usage=usage,
            latency_ms=2500,
            finish_reason="stop",
            provider="openai",
            cost=0.015
        )

        # Verify everything works together
        assert request.prompt in request.prompt
        assert usage.total_tokens == 100  # Auto-calculated
        assert response.usage.total_tokens == 100
        assert FinishReason.normalize(response.finish_reason, response.provider) == "stop"

    def test_model_validation_edge_cases(self):
        """Test edge cases and validation across models."""
        # Test minimal valid request
        request = UnifiedRequest(prompt="hi")
        assert request.prompt == "hi"

        # Test minimal valid response
        usage = TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        response = UnifiedResponse(
            content="hi",
            model="test",
            usage=usage,
            latency_ms=100,
            finish_reason="stop",
            provider="test"
        )
        assert response.content == "hi"
