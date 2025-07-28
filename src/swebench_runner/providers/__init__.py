"""Model provider infrastructure for SWE-bench runner."""

from __future__ import annotations

# Providers
from .anthropic import AnthropicProvider

# Async/sync bridge
from .async_bridge import (
    AsyncBridge,
    async_to_sync,
    run_async,
)

# Base classes and types
# Authentication strategies
from .auth_strategies import (
    ApiKeyAuth,
    AuthConfig,
    AuthStrategy,
    AuthStrategyFactory,
    AuthType,
    BearerTokenAuth,
    NoAuth,
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
from .mock import MockProvider
from .ollama import OllamaProvider
from .openai import OpenAIProvider
from .openrouter import OpenRouterProvider

# Rate limiting
from .rate_limiters import (
    AcquisitionRequest,
    AcquisitionResult,
    CompositeLimiter,
    LimiterType,
    RateLimitConfig,
    RateLimitCoordinator,
    RateLimiter,
    SemaphoreLimiter,
    SlidingWindowLimiter,
    TokenBucketLimiter,
    create_anthropic_limiter,
    create_coordinator_from_config,
    create_ollama_limiter,
    create_openai_limiter,
)

# Registry and registration
from .registry import (
    ProviderRegistry,
    get_registry,
    register_provider,
)

# Streaming adapters
from .streaming_adapters import (
    JSONLinesAdapter,
    PlainTextAdapter,
    SSEAdapter,
    StreamChunk,
    StreamingAdapter,
    StreamingFormat,
    StreamingManager,
    create_streaming_manager,
    stream_anthropic_response,
    stream_ollama_response,
    stream_openai_response,
)

# Token counting
from .token_counters import (
    AnthropicAPICounter,
    MetadataTokenCounter,
    TiktokenCounter,
    TokenCounter,
    TokenCounterType,
    TokenCountRequest,
    TokenCountResult,
    UnifiedTokenCounter,
    create_unified_counter,
)

# Transform pipeline
from .transform_pipeline import (
    AnthropicRequestTransformer,
    AnthropicResponseParser,
    OllamaRequestTransformer,
    OllamaResponseParser,
    OpenAIRequestTransformer,
    OpenAIResponseParser,
    RequestTransformer,
    ResponseParser,
    TransformPipeline,
    TransformPipelineConfig,
)

# Unified models
from .unified_models import (
    FinishReason,
    TokenUsage,
    UnifiedRequest,
    UnifiedResponse,
)

# Wrappers
from .wrappers import (
    CircuitBreakerProvider,
    SyncProviderWrapper,
)

__all__ = [
    # Authentication strategies
    "AuthType",
    "AuthConfig",
    "AuthStrategy",
    "BearerTokenAuth",
    "ApiKeyAuth",
    "NoAuth",
    "AuthStrategyFactory",
    # Base classes and types
    "ModelProvider",
    "ModelResponse",
    "ProviderConfig",
    "ProviderCapabilities",
    # Registry and registration
    "ProviderRegistry",
    "register_provider",
    "get_registry",
    # Token counting
    "TokenCounter",
    "TokenCounterType",
    "TokenCountRequest",
    "TokenCountResult",
    "TiktokenCounter",
    "AnthropicAPICounter",
    "MetadataTokenCounter",
    "UnifiedTokenCounter",
    "create_unified_counter",
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
    "AnthropicProvider",
    "MockProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "OpenRouterProvider",
    # Transform pipeline
    "RequestTransformer",
    "ResponseParser",
    "TransformPipeline",
    "TransformPipelineConfig",
    "OpenAIRequestTransformer",
    "OpenAIResponseParser",
    "AnthropicRequestTransformer",
    "AnthropicResponseParser",
    "OllamaRequestTransformer",
    "OllamaResponseParser",
    # Unified models
    "TokenUsage",
    "UnifiedRequest",
    "UnifiedResponse",
    "FinishReason",
    # Wrappers
    "CircuitBreakerProvider",
    "SyncProviderWrapper",
    # Rate limiting
    "AcquisitionRequest",
    "AcquisitionResult",
    "CompositeLimiter",
    "LimiterType",
    "RateLimitConfig",
    "RateLimitCoordinator",
    "RateLimiter",
    "SemaphoreLimiter",
    "SlidingWindowLimiter",
    "TokenBucketLimiter",
    "create_anthropic_limiter",
    "create_coordinator_from_config",
    "create_ollama_limiter",
    "create_openai_limiter",
    # Streaming adapters
    "StreamingFormat",
    "StreamChunk",
    "StreamingAdapter",
    "SSEAdapter",
    "JSONLinesAdapter",
    "PlainTextAdapter",
    "StreamingManager",
    "stream_openai_response",
    "stream_anthropic_response",
    "stream_ollama_response",
    "create_streaming_manager",
]

# Version info
__version__ = "0.1.0"
