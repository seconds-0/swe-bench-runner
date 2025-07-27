"""Provider wrappers for additional functionality."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from .async_bridge import AsyncBridge
from .base import ModelProvider, ModelResponse, ProviderCapabilities, ProviderConfig
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from .exceptions import CircuitBreakerError, ProviderError

logger = logging.getLogger(__name__)


class CircuitBreakerProvider(ModelProvider):
    """Wrapper that adds circuit breaker functionality to any provider.

    This wrapper protects against cascading failures by temporarily blocking
    calls to a failing provider. It maintains the same interface as the
    wrapped provider while adding fault tolerance.
    """

    def __init__(
        self,
        provider: ModelProvider,
        circuit_config: CircuitBreakerConfig | None = None,
        on_state_change: Callable[..., Any] | None = None
    ):
        """Initialize circuit breaker wrapper.

        Args:
            provider: The provider instance to wrap
            circuit_config: Optional circuit breaker configuration
            on_state_change: Optional callback for state changes
        """
        # Store the wrapped provider first
        self._wrapped_provider = provider

        # Copy provider metadata before calling super().__init__
        self.name = f"{provider.name}_circuit_breaker"
        self.description = f"{provider.description} (with circuit breaker)"
        self.api_version = provider.api_version
        self.requires_api_key = provider.requires_api_key
        self.supported_models = provider.supported_models
        self.supports_streaming = provider.supports_streaming
        self.default_model = provider.default_model

        # Now initialize with the wrapped provider's config
        # The parent class will see our overridden attributes
        super().__init__(provider.config)

        # Initialize circuit breaker
        self._circuit = CircuitBreaker(
            name=provider.name,
            config=circuit_config or CircuitBreakerConfig(
                failure_threshold=5,
                recovery_timeout=60.0,
                expected_exception=ProviderError,
                success_threshold=2
            ),
            on_state_change=on_state_change
        )

    def _init_capabilities(self) -> ProviderCapabilities:
        """Initialize capabilities from wrapped provider."""
        return self._wrapped_provider.capabilities

    async def generate(self, prompt: str, **kwargs: Any) -> ModelResponse:
        """Generate response through circuit breaker.

        Args:
            prompt: The input prompt
            **kwargs: Additional provider-specific parameters

        Returns:
            ModelResponse from the wrapped provider

        Raises:
            CircuitBreakerError: If circuit is open
            ProviderError: Various provider-specific errors
        """
        try:
            # Call through circuit breaker
            response = await self._circuit.call(
                self._wrapped_provider.generate,
                prompt,
                **kwargs
            )
            return response  # type: ignore[no-any-return]
        except CircuitBreakerError:
            # Re-raise circuit breaker errors as-is
            raise
        except Exception as e:
            # Log the error for monitoring
            logger.warning(f"Provider {self._wrapped_provider.name} error: {e}")
            raise

    async def validate_connection(self) -> bool:
        """Validate connection through circuit breaker."""
        try:
            # If circuit is open, return False immediately
            if self._circuit.is_open():
                self._health_status = "unhealthy"
                return False

            # Otherwise check the wrapped provider
            result = await self._circuit.call(
                self._wrapped_provider.validate_connection
            )
            self._health_status = "healthy" if result else "unhealthy"
            return result  # type: ignore[no-any-return]
        except Exception:
            self._health_status = "unhealthy"
            return False

    def estimate_cost(self, prompt_tokens: int, max_tokens: int) -> float:
        """Estimate cost using wrapped provider."""
        return self._wrapped_provider.estimate_cost(prompt_tokens, max_tokens)

    @classmethod
    def get_required_env_vars(cls) -> list[str]:
        """Circuit breaker doesn't add env var requirements."""
        # This is a wrapper, so we can't determine this at class level
        return []

    @classmethod
    def _config_from_env(
        cls, env_vars: dict[str, str], model: str | None = None
    ) -> ProviderConfig:
        """Circuit breaker wrapper doesn't have its own env config."""
        # This method shouldn't be called on the wrapper
        raise NotImplementedError(
            "CircuitBreakerProvider must be initialized with an existing provider "
            "instance"
        )

    def get_token_limit(self) -> int:
        """Get token limit from wrapped provider."""
        return self._wrapped_provider.get_token_limit()

    def get_generation_params(self) -> dict[str, Any]:
        """Get generation parameters from wrapped provider."""
        return self._wrapped_provider.get_generation_params()

    async def health_check(self) -> dict[str, Any]:
        """Get health status including circuit breaker state."""
        base_health = await self._wrapped_provider.health_check()

        # Add circuit breaker information
        circuit_stats = self._circuit.get_stats()
        base_health.update({
            "circuit_state": self._circuit.state.value,
            "circuit_stats": {
                "total_calls": circuit_stats.total_calls,
                "total_failures": circuit_stats.total_failures,
                "total_successes": circuit_stats.total_successes,
                "consecutive_failures": circuit_stats.consecutive_failures,
                "consecutive_successes": circuit_stats.consecutive_successes,
            }
        })

        return base_health

    def get_circuit_stats(self) -> dict[str, Any]:
        """Get detailed circuit breaker statistics.

        Returns:
            Dictionary with circuit breaker stats
        """
        stats = self._circuit.get_stats()
        return {
            "state": self._circuit.state.value,
            "failure_count": stats.failure_count,
            "success_count": stats.success_count,
            "total_calls": stats.total_calls,
            "total_failures": stats.total_failures,
            "total_successes": stats.total_successes,
            "consecutive_failures": stats.consecutive_failures,
            "consecutive_successes": stats.consecutive_successes,
            "last_failure_time": (
                stats.last_failure_time.isoformat() if stats.last_failure_time else None
            ),
            "last_success_time": (
                stats.last_success_time.isoformat() if stats.last_success_time else None
            ),
            "state_changes": {
                state: time.isoformat() for state, time in stats.state_changes.items()
            }
        }

    def reset_circuit(self) -> None:
        """Reset the circuit breaker to closed state."""
        self._circuit.reset()

    @property
    def wrapped_provider(self) -> ModelProvider:
        """Get the wrapped provider instance."""
        return self._wrapped_provider


class SyncProviderWrapper(ModelProvider):
    """Wrapper that provides synchronous interface for async providers.

    This wrapper uses AsyncBridge to allow async providers to be used
    in synchronous contexts, such as CLI applications. It maintains
    full provider functionality while handling the async/sync conversion.
    """

    def __init__(self, async_provider: ModelProvider):
        """Initialize sync wrapper.

        Args:
            async_provider: The async provider to wrap
        """
        # Store the async provider first
        self._async_provider = async_provider

        # Copy provider metadata before calling super().__init__
        self.name = f"{async_provider.name}_sync"
        self.description = f"{async_provider.description} (sync wrapper)"
        self.api_version = async_provider.api_version
        self.requires_api_key = async_provider.requires_api_key
        self.supported_models = async_provider.supported_models
        self.supports_streaming = async_provider.supports_streaming
        self.default_model = async_provider.default_model

        # Now initialize with the wrapped provider's config
        # The parent class will see our overridden attributes
        super().__init__(async_provider.config)

        # Get the AsyncBridge singleton
        self._bridge = AsyncBridge()

    def _init_capabilities(self) -> ProviderCapabilities:
        """Initialize capabilities from wrapped provider."""
        return self._async_provider.capabilities

    def generate_sync(self, prompt: str, **kwargs: Any) -> ModelResponse:
        """Synchronous generate method.

        Args:
            prompt: The input prompt
            **kwargs: Additional provider-specific parameters

        Returns:
            ModelResponse from the async provider

        Raises:
            ProviderTimeoutError: If operation times out
            ProviderError: Various provider-specific errors
        """
        # Get timeout from kwargs or config
        timeout = kwargs.get('timeout', self.config.timeout)

        # Run async method synchronously
        return self._bridge.run(
            self._async_provider.generate(prompt, **kwargs),
            timeout=timeout
        )

    async def generate(self, prompt: str, **kwargs: Any) -> ModelResponse:
        """Async generate method (delegates to wrapped provider).

        This allows the sync wrapper to still be used in async contexts.
        """
        return await self._async_provider.generate(prompt, **kwargs)

    def validate_connection_sync(self) -> bool:
        """Synchronous connection validation."""
        return self._bridge.run(
            self._async_provider.validate_connection(),
            timeout=10.0  # Short timeout for validation
        )

    async def validate_connection(self) -> bool:
        """Async connection validation (delegates to wrapped provider)."""
        return await self._async_provider.validate_connection()

    def health_check_sync(self) -> dict[str, Any]:
        """Synchronous health check."""
        return self._bridge.run(
            self._async_provider.health_check(),
            timeout=5.0  # Short timeout for health check
        )

    async def health_check(self) -> dict[str, Any]:
        """Async health check (delegates to wrapped provider)."""
        return await self._async_provider.health_check()

    def estimate_cost(self, prompt_tokens: int, max_tokens: int) -> float:
        """Estimate cost using wrapped provider."""
        return self._async_provider.estimate_cost(prompt_tokens, max_tokens)

    @classmethod
    def get_required_env_vars(cls) -> list[str]:
        """Sync wrapper doesn't add env var requirements."""
        # This is a wrapper, so we can't determine this at class level
        return []

    @classmethod
    def _config_from_env(
        cls, env_vars: dict[str, str], model: str | None = None
    ) -> ProviderConfig:
        """Sync wrapper doesn't have its own env config."""
        # This method shouldn't be called on the wrapper
        raise NotImplementedError(
            "SyncProviderWrapper must be initialized with an existing provider instance"
        )

    def get_token_limit(self) -> int:
        """Get token limit from wrapped provider."""
        return self._async_provider.get_token_limit()

    def get_generation_params(self) -> dict[str, Any]:
        """Get generation parameters from wrapped provider."""
        return self._async_provider.get_generation_params()

    def get_bridge_stats(self) -> dict[str, Any]:
        """Get async bridge statistics.

        Returns:
            Dictionary with bridge performance stats
        """
        return self._bridge.get_stats()

    @property
    def async_provider(self) -> ModelProvider:
        """Get the wrapped async provider instance."""
        return self._async_provider

    def __enter__(self) -> SyncProviderWrapper:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        # The AsyncBridge singleton manages its own lifecycle
        pass
