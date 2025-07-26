"""Model provider infrastructure for SWE-bench runner."""

# Base classes and types
# Async/sync bridge
from .async_bridge import (
    AsyncBridge,
    async_to_sync,
    run_async,
)
from .base import (
    ModelProvider,
    ModelResponse,
    ProviderCapabilities,
    ProviderConfig,
)

# Circuit breaker
from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerStats,
    CircuitState,
)

# Configuration management
from .config import ProviderConfigManager

# Exceptions
from .exceptions import (
    CircuitBreakerError,
    ProviderAuthenticationError,
    ProviderConfigurationError,
    ProviderConnectionError,
    ProviderError,
    ProviderNotFoundError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderTokenLimitError,
)

# Providers
from .mock import MockProvider
from .openai import OpenAIProvider
from .openrouter import OpenRouterProvider

# Registry and registration
from .registry import (
    ProviderRegistry,
    get_registry,
    register_provider,
)

# Wrappers
from .wrappers import (
    CircuitBreakerProvider,
    SyncProviderWrapper,
)

__all__ = [
    # Base classes and types
    "ModelProvider",
    "ModelResponse",
    "ProviderConfig",
    "ProviderCapabilities",
    # Registry and registration
    "ProviderRegistry",
    "register_provider",
    "get_registry",
    # Configuration management
    "ProviderConfigManager",
    # Exceptions
    "ProviderError",
    "ProviderNotFoundError",
    "ProviderConnectionError",
    "ProviderAuthenticationError",
    "ProviderRateLimitError",
    "ProviderResponseError",
    "ProviderConfigurationError",
    "ProviderTimeoutError",
    "ProviderTokenLimitError",
    "CircuitBreakerError",
    # Circuit breaker
    "CircuitBreaker",
    "CircuitState",
    "CircuitBreakerConfig",
    "CircuitBreakerStats",
    # Async/sync bridge
    "AsyncBridge",
    "run_async",
    "async_to_sync",
    # Providers
    "MockProvider",
    "OpenAIProvider",
    "OpenRouterProvider",
    # Wrappers
    "CircuitBreakerProvider",
    "SyncProviderWrapper",
]

# Version info
__version__ = "0.1.0"
