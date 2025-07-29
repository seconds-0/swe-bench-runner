"""Provider-specific exceptions."""

from __future__ import annotations


class ProviderError(Exception):
    """Base exception for all provider-related errors."""

    def __init__(self, message: str, provider: str | None = None):
        super().__init__(message)
        self.provider = provider


class ProviderNotFoundError(ProviderError):
    """Raised when a requested provider is not found."""

    def __init__(self, message: str, provider: str | None = None):
        super().__init__(message, provider)


class ProviderConnectionError(ProviderError):
    """Raised when connection to provider fails."""

    def __init__(self, message: str, provider: str | None = None):
        super().__init__(message, provider)


class ProviderAuthenticationError(ProviderError):
    """Raised when provider authentication fails."""

    def __init__(self, message: str, provider: str | None = None):
        super().__init__(message, provider)


class ProviderRateLimitError(ProviderError):
    """Raised when provider rate limit is exceeded."""

    def __init__(
        self, message: str, retry_after: int | None = None, provider: str | None = None
    ):
        super().__init__(message, provider)
        self.retry_after = retry_after


class ProviderResponseError(ProviderError):
    """Raised when provider returns an invalid response."""

    def __init__(self, message: str, provider: str | None = None):
        super().__init__(message, provider)


class ProviderConfigurationError(ProviderError):
    """Raised when provider configuration is invalid."""

    def __init__(self, message: str, provider: str | None = None):
        super().__init__(message, provider)


class ProviderTimeoutError(ProviderError):
    """Raised when provider request times out."""

    def __init__(self, message: str, provider: str | None = None):
        super().__init__(message, provider)


class ProviderTokenLimitError(ProviderError):
    """Raised when input exceeds token limit."""

    def __init__(
        self,
        message: str,
        token_count: int | None = None,
        limit: int | None = None,
        provider: str | None = None
    ):
        super().__init__(message, provider)
        self.token_count = token_count
        self.limit = limit


class CircuitBreakerError(ProviderError):
    """Raised when circuit breaker is open."""

    def __init__(
        self, message: str, provider: str | None = None, wait_time: float | None = None
    ):
        super().__init__(message, provider)
        self.wait_time = wait_time
