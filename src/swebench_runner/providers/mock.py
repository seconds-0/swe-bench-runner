"""Mock provider for testing."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .base import ModelProvider, ModelResponse, ProviderCapabilities, ProviderConfig
from .exceptions import (
    ProviderAuthenticationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderTokenLimitError,
)
from .registry import register_provider

logger = logging.getLogger(__name__)


@register_provider
class MockProvider(ModelProvider):
    """Mock provider for testing purposes.

    Supports configurable responses and error injection for testing
    provider integration without making real API calls.
    """

    name = "mock"
    description = "Mock provider for testing"
    api_version = "1.0"
    requires_api_key = False  # Mock provider doesn't need real API keys
    supported_models = ["mock-small", "mock-large", "mock-streaming"]
    supports_streaming = True
    default_model = "mock-small"

    def __init__(
        self,
        config: ProviderConfig,
        mock_responses: dict[str, str] | None = None,
        mock_errors: dict[str, Exception] | None = None,
        response_delay: float = 0.1,
    ):
        """Initialize mock provider.

        Args:
            config: Provider configuration
            mock_responses: Dictionary mapping prompts to responses
            mock_errors: Dictionary mapping prompts to exceptions
            response_delay: Simulated response delay in seconds
        """
        super().__init__(config)
        self.mock_responses = mock_responses or {}
        self.mock_errors = mock_errors or {}
        self.response_delay = response_delay
        self._call_count = 0
        self._total_tokens = 0

    def _init_capabilities(self) -> ProviderCapabilities:
        """Initialize mock provider capabilities."""
        model = self.config.model if self.config else self.default_model
        return ProviderCapabilities(
            max_context_length=8192 if "large" in (model or "") else 4096,
            supports_streaming=True,
            supports_json_mode=True,
            supports_function_calling=False,
            rate_limits={"requests_per_minute": 60, "tokens_per_minute": 10000},
            supported_models=self.supported_models,
            cost_per_1k_prompt_tokens=0.001,
            cost_per_1k_completion_tokens=0.002,
        )

    async def generate(self, prompt: str, **kwargs: Any) -> ModelResponse:
        """Generate a mock response.

        Args:
            prompt: The input prompt
            **kwargs: Additional parameters (max_tokens, temperature, etc.)

        Returns:
            ModelResponse with mock data

        Raises:
            Various provider exceptions based on mock_errors configuration
        """
        self._call_count += 1

        # Check for configured errors first
        if prompt in self.mock_errors:
            # Simulate processing time even for errors
            await asyncio.sleep(self.response_delay)
            raise self.mock_errors[prompt]

        # Check for specific error patterns in prompt
        if "auth_error" in prompt.lower():
            raise ProviderAuthenticationError(
                "Mock authentication failed", provider="mock"
            )
        elif "rate_limit" in prompt.lower():
            raise ProviderRateLimitError(
                "Mock rate limit exceeded", retry_after=60, provider="mock"
            )
        elif "timeout" in prompt.lower():
            raise ProviderTimeoutError("Mock request timed out", provider="mock")
        elif "token_limit" in prompt.lower():
            raise ProviderTokenLimitError(
                "Mock token limit exceeded",
                token_count=10000,
                limit=self.capabilities.max_context_length,
                provider="mock"
            )
        elif "generic_error" in prompt.lower():
            raise ProviderError("Mock generic error", provider="mock")

        # Simulate response delay
        await asyncio.sleep(self.response_delay)

        # Get response
        if prompt in self.mock_responses:
            content = self.mock_responses[prompt]
        else:
            # Generate a default response
            content = f"Mock response to: {prompt[:50]}..."
            if kwargs.get("stream", False):
                content = f"[STREAMING] {content}"

        # Calculate mock token counts
        prompt_tokens = len(prompt.split()) * 2  # Rough estimation
        completion_tokens = len(content.split()) * 2
        total_tokens = prompt_tokens + completion_tokens
        self._total_tokens += total_tokens

        # Calculate cost
        prompt_cost_per_1k = self.capabilities.cost_per_1k_prompt_tokens or 0.0
        completion_cost_per_1k = self.capabilities.cost_per_1k_completion_tokens or 0.0
        cost = (
            (prompt_tokens / 1000) * prompt_cost_per_1k +
            (completion_tokens / 1000) * completion_cost_per_1k
        )

        # Build response
        return ModelResponse(
            content=content,
            model=self.config.model or self.default_model,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            cost=cost,
            latency_ms=int(self.response_delay * 1000),
            provider=self.name,
            finish_reason="stop",
            raw_response={
                "mock": True,
                "call_count": self._call_count,
                "kwargs": kwargs,
            }
        )

    async def validate_connection(self) -> bool:
        """Validate mock connection (always succeeds unless configured otherwise)."""
        # Check if we're configured to fail validation
        if "fail_validation" in self.mock_errors:
            self._health_status = "unhealthy"
            return False

        self._health_status = "healthy"
        return True

    def estimate_cost(self, prompt_tokens: int, max_tokens: int) -> float:
        """Estimate the cost of a generation.

        Args:
            prompt_tokens: Number of tokens in the prompt
            max_tokens: Maximum number of completion tokens

        Returns:
            Estimated cost in USD
        """
        prompt_cost_per_1k = self.capabilities.cost_per_1k_prompt_tokens or 0.0
        completion_cost_per_1k = self.capabilities.cost_per_1k_completion_tokens or 0.0
        prompt_cost = (prompt_tokens / 1000) * prompt_cost_per_1k
        completion_cost = (max_tokens / 1000) * completion_cost_per_1k
        return prompt_cost + completion_cost

    @classmethod
    def get_required_env_vars(cls) -> list[str]:
        """Mock provider doesn't require any environment variables."""
        return []

    @classmethod
    def _config_from_env(
        cls, env_vars: dict[str, str], model: str | None = None
    ) -> ProviderConfig:
        """Create mock config from environment (uses defaults)."""
        return ProviderConfig(
            name=cls.name,
            model=model or cls.default_model,
            temperature=0.0,
            max_tokens=1000,
        )

    def get_stats(self) -> dict[str, Any]:
        """Get mock provider statistics.

        Returns:
            Dictionary with usage statistics
        """
        return {
            "call_count": self._call_count,
            "total_tokens": self._total_tokens,
            "mock_responses_count": len(self.mock_responses),
            "mock_errors_count": len(self.mock_errors),
            "response_delay": self.response_delay,
        }

    def set_response(self, prompt: str, response: str) -> None:
        """Set a mock response for a specific prompt.

        Args:
            prompt: The prompt to match
            response: The response to return
        """
        self.mock_responses[prompt] = response

    def set_error(self, prompt: str, error: Exception) -> None:
        """Set a mock error for a specific prompt.

        Args:
            prompt: The prompt to match
            error: The exception to raise
        """
        self.mock_errors[prompt] = error

    def clear_mocks(self) -> None:
        """Clear all mock responses and errors."""
        self.mock_responses.clear()
        self.mock_errors.clear()
        self._call_count = 0
        self._total_tokens = 0
