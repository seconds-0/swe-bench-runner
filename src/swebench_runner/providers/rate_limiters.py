"""Rate limiting coordinator for provider abstraction layer."""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class LimiterType(Enum):
    """Types of rate limiters."""

    TOKEN_BUCKET = "token_bucket"  # noqa: S105
    SLIDING_WINDOW = "sliding_window"  # noqa: S105
    SEMAPHORE = "semaphore"  # noqa: S105
    COMPOSITE = "composite"  # noqa: S105


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    requests_per_minute: int | None = None
    tokens_per_minute: int | None = None
    concurrent_requests: int | None = None
    burst_allowance: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if (
            not self.requests_per_minute
            and not self.tokens_per_minute
            and not self.concurrent_requests
        ):
            raise ValueError("At least one rate limit must be specified")


@dataclass
class AcquisitionRequest:
    """Request to acquire rate limit resources."""

    estimated_tokens: int = 1
    priority: int = 0  # Higher number = higher priority
    timeout: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AcquisitionResult:
    """Result of rate limit acquisition."""

    acquired: bool
    wait_time: float = 0.0
    retry_after: float | None = None
    limited_by: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class RateLimiter(ABC):
    """Abstract base class for rate limiters."""

    @abstractmethod
    async def acquire(self, request: AcquisitionRequest) -> AcquisitionResult:
        """Attempt to acquire rate limit permission."""
        pass

    @abstractmethod
    def release(self, tokens_used: int = 1) -> None:
        """Release resources (for semaphore-style limiters)."""
        pass

    @abstractmethod
    def get_status(self) -> dict[str, Any]:
        """Get current rate limiter status."""
        pass

    @property
    @abstractmethod
    def limiter_type(self) -> LimiterType:
        """Get the type of this rate limiter."""
        pass


class TokenBucketLimiter(RateLimiter):
    """Token bucket rate limiter for smooth rate limiting."""

    def __init__(
        self,
        capacity: int,
        refill_rate: float,
        burst_allowance: int | None = None,
    ):
        """Initialize token bucket limiter.

        Args:
            capacity: Maximum number of tokens in bucket
            refill_rate: Tokens added per second
            burst_allowance: Additional tokens for burst (default: capacity)
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.burst_allowance = burst_allowance or capacity

        self._tokens = float(capacity)
        self._last_refill = time.time()
        self._lock: asyncio.Lock | None = None

    async def acquire(self, request: AcquisitionRequest) -> AcquisitionResult:
        """Acquire tokens from the bucket."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            self._refill_tokens()

            tokens_needed = request.estimated_tokens

            if self._tokens >= tokens_needed:
                self._tokens -= tokens_needed
                return AcquisitionResult(
                    acquired=True,
                    metadata={
                        "tokens_remaining": self._tokens,
                        "capacity": self.capacity,
                        "refill_rate": self.refill_rate,
                    },
                )
            else:
                # Calculate wait time
                tokens_deficit = tokens_needed - self._tokens
                wait_time = tokens_deficit / self.refill_rate

                return AcquisitionResult(
                    acquired=False,
                    wait_time=wait_time,
                    retry_after=wait_time,
                    limited_by="token_bucket",
                    metadata={
                        "tokens_available": self._tokens,
                        "tokens_needed": tokens_needed,
                        "wait_time": wait_time,
                    },
                )

    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_refill

        tokens_to_add = elapsed * self.refill_rate
        max_tokens = self.capacity + self.burst_allowance

        self._tokens = min(max_tokens, self._tokens + tokens_to_add)
        self._last_refill = now

    def release(self, tokens_used: int = 1) -> None:
        """No-op for token bucket (tokens are consumed on acquire)."""
        pass

    def get_status(self) -> dict[str, Any]:
        """Get current bucket status."""
        self._refill_tokens()
        return {
            "type": "token_bucket",
            "tokens_available": self._tokens,
            "capacity": self.capacity,
            "refill_rate": self.refill_rate,
            "utilization": 1.0 - (self._tokens / self.capacity),
        }

    @property
    def limiter_type(self) -> LimiterType:
        """Get the type of this rate limiter."""
        return LimiterType.TOKEN_BUCKET


class SlidingWindowLimiter(RateLimiter):
    """Sliding window rate limiter for request counting."""

    def __init__(self, limit: int, window_seconds: int = 60):
        """Initialize sliding window limiter.

        Args:
            limit: Maximum requests in window
            window_seconds: Window size in seconds
        """
        self.limit = limit
        self.window_seconds = window_seconds

        self._requests: deque[float] = deque()
        self._lock: asyncio.Lock | None = None

    async def acquire(self, request: AcquisitionRequest) -> AcquisitionResult:
        """Acquire request permission."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            self._cleanup_old_requests()

            if len(self._requests) < self.limit:
                self._requests.append(time.time())
                return AcquisitionResult(
                    acquired=True,
                    metadata={
                        "requests_in_window": len(self._requests),
                        "limit": self.limit,
                        "window_seconds": self.window_seconds,
                    },
                )
            else:
                # Calculate wait time until oldest request expires
                oldest_request = self._requests[0]
                wait_time = self.window_seconds - (time.time() - oldest_request)

                return AcquisitionResult(
                    acquired=False,
                    wait_time=max(0, wait_time),
                    retry_after=max(0, wait_time),
                    limited_by="sliding_window",
                    metadata={
                        "requests_in_window": len(self._requests),
                        "limit": self.limit,
                        "wait_time": wait_time,
                    },
                )

    def _cleanup_old_requests(self) -> None:
        """Remove requests outside the window."""
        cutoff = time.time() - self.window_seconds
        while self._requests and self._requests[0] < cutoff:
            self._requests.popleft()

    def release(self, tokens_used: int = 1) -> None:
        """No-op for sliding window (requests age out naturally)."""
        pass

    def get_status(self) -> dict[str, Any]:
        """Get current window status."""
        self._cleanup_old_requests()
        return {
            "type": "sliding_window",
            "requests_in_window": len(self._requests),
            "limit": self.limit,
            "window_seconds": self.window_seconds,
            "utilization": len(self._requests) / self.limit,
        }

    @property
    def limiter_type(self) -> LimiterType:
        """Get the type of this rate limiter."""
        return LimiterType.SLIDING_WINDOW


class SemaphoreLimiter(RateLimiter):
    """Semaphore-based rate limiter for concurrent request limiting."""

    def __init__(self, concurrent_limit: int):
        """Initialize semaphore limiter."""
        self.concurrent_limit = concurrent_limit
        self._semaphore: asyncio.Semaphore | None = None
        self._active_requests = 0
        self._lock: asyncio.Lock | None = None

    async def acquire(self, request: AcquisitionRequest) -> AcquisitionResult:
        """Acquire semaphore permission."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.concurrent_limit)
        if self._lock is None:
            self._lock = asyncio.Lock()

        timeout = request.timeout

        try:
            acquired = await asyncio.wait_for(
                self._semaphore.acquire(), timeout=timeout
            )

            if acquired:
                async with self._lock:
                    self._active_requests += 1

                return AcquisitionResult(
                    acquired=True,
                    metadata={
                        "active_requests": self._active_requests,
                        "concurrent_limit": self.concurrent_limit,
                    },
                )

        except asyncio.TimeoutError:
            return AcquisitionResult(
                acquired=False,
                wait_time=timeout or 0,
                limited_by="semaphore",
                metadata={
                    "timeout_exceeded": True,
                    "concurrent_limit": self.concurrent_limit,
                },
            )

        return AcquisitionResult(acquired=False, limited_by="semaphore")

    def release(self, tokens_used: int = 1) -> None:
        """Release semaphore."""
        if self._semaphore is not None:
            self._semaphore.release()
        # Note: Counter decrement happens in acquire method
        # We don't need to track it separately in release


    def get_status(self) -> dict[str, Any]:
        """Get current semaphore status."""
        return {
            "type": "semaphore",
            "active_requests": self._active_requests,
            "concurrent_limit": self.concurrent_limit,
            "available": self._semaphore._value if self._semaphore else self.concurrent_limit,
            "utilization": self._active_requests / self.concurrent_limit,
        }

    @property
    def limiter_type(self) -> LimiterType:
        """Get the type of this rate limiter."""
        return LimiterType.SEMAPHORE


class CompositeLimiter(RateLimiter):
    """Composite limiter that combines multiple rate limiters."""

    def __init__(self, limiters: list[RateLimiter]):
        """Initialize composite limiter."""
        if not limiters:
            raise ValueError("At least one limiter required")
        self.limiters = limiters

    async def acquire(self, request: AcquisitionRequest) -> AcquisitionResult:
        """Acquire permission from all limiters."""
        results = []
        acquired_limiters = []

        try:
            for limiter in self.limiters:
                result = await limiter.acquire(request)
                results.append(result)

                if result.acquired:
                    acquired_limiters.append(limiter)
                else:
                    # Failed to acquire, rollback previous acquisitions
                    for acquired_limiter in acquired_limiters:
                        acquired_limiter.release(request.estimated_tokens)

                    return AcquisitionResult(
                        acquired=False,
                        wait_time=result.wait_time,
                        retry_after=result.retry_after,
                        limited_by=result.limited_by,
                        metadata={
                            "failed_limiter": limiter.limiter_type.value,
                            "all_results": [r.__dict__ for r in results],
                        },
                    )

            # All limiters acquired successfully
            return AcquisitionResult(
                acquired=True,
                metadata={
                    "limiters_used": [
                        limiter.limiter_type.value for limiter in self.limiters
                    ],
                    "all_results": [r.__dict__ for r in results],
                },
            )

        except Exception:
            # Rollback any successful acquisitions
            for acquired_limiter in acquired_limiters:
                acquired_limiter.release(request.estimated_tokens)
            raise

    def release(self, tokens_used: int = 1) -> None:
        """Release all limiters."""
        for limiter in self.limiters:
            limiter.release(tokens_used)

    def get_status(self) -> dict[str, Any]:
        """Get status of all limiters."""
        return {
            "type": "composite",
            "limiters": [limiter.get_status() for limiter in self.limiters],
        }

    @property
    def limiter_type(self) -> LimiterType:
        """Get the type of this rate limiter."""
        return LimiterType.COMPOSITE


class RateLimitCoordinator:
    """Coordinator for managing rate limits across multiple providers."""

    def __init__(self) -> None:
        """Initialize coordinator."""
        self._provider_limiters: dict[str, RateLimiter] = {}
        self._global_limiter: RateLimiter | None = None

    def add_provider_limiter(self, provider: str, limiter: RateLimiter) -> None:
        """Add rate limiter for a specific provider."""
        self._provider_limiters[provider] = limiter

    def set_global_limiter(self, limiter: RateLimiter) -> None:
        """Set global rate limiter applied to all providers."""
        self._global_limiter = limiter

    async def acquire(
        self, provider: str, request: AcquisitionRequest
    ) -> AcquisitionResult:
        """Acquire rate limit permission for provider."""
        limiters_to_check = []

        # Add global limiter if exists
        if self._global_limiter:
            limiters_to_check.append(("global", self._global_limiter))

        # Add provider-specific limiter if exists
        if provider in self._provider_limiters:
            limiters_to_check.append((provider, self._provider_limiters[provider]))

        if not limiters_to_check:
            # No limiters configured, allow immediately
            return AcquisitionResult(acquired=True)

        # Try to acquire from all applicable limiters
        acquired_limiters = []

        try:
            for limiter_name, limiter in limiters_to_check:
                result = await limiter.acquire(request)

                if result.acquired:
                    acquired_limiters.append((limiter_name, limiter))
                else:
                    # Failed to acquire, rollback
                    for _, acquired_limiter in acquired_limiters:
                        acquired_limiter.release(request.estimated_tokens)

                    # Add context about which limiter failed
                    result.metadata["failed_limiter"] = limiter_name
                    return result

            # All limiters acquired successfully
            return AcquisitionResult(
                acquired=True,
                metadata={
                    "limiters_acquired": [name for name, _ in acquired_limiters]
                },
            )

        except Exception:
            # Rollback any successful acquisitions
            for _, acquired_limiter in acquired_limiters:
                acquired_limiter.release(request.estimated_tokens)
            raise

    def release(self, provider: str, tokens_used: int = 1) -> None:
        """Release rate limit resources for provider."""
        if self._global_limiter:
            self._global_limiter.release(tokens_used)

        if provider in self._provider_limiters:
            self._provider_limiters[provider].release(tokens_used)

    def get_status(self, provider: str | None = None) -> dict[str, Any]:
        """Get rate limiting status."""
        status: dict[str, Any] = {"global": None, "providers": {}}

        if self._global_limiter:
            status["global"] = self._global_limiter.get_status()

        if provider:
            if provider in self._provider_limiters:
                status["providers"][provider] = self._provider_limiters[
                    provider
                ].get_status()
        else:
            for prov, limiter in self._provider_limiters.items():
                status["providers"][prov] = limiter.get_status()

        return status


# Factory functions for easy setup


def create_openai_limiter(
    requests_per_minute: int = 1000, tokens_per_minute: int = 40000
) -> CompositeLimiter:
    """Create rate limiter for OpenAI with RPM and TPM limits."""
    return CompositeLimiter([
        SlidingWindowLimiter(requests_per_minute, 60),
        TokenBucketLimiter(tokens_per_minute, tokens_per_minute / 60.0),
    ])


def create_anthropic_limiter(tokens_per_minute: int = 50000) -> TokenBucketLimiter:
    """Create rate limiter for Anthropic with token bucket."""
    return TokenBucketLimiter(
        capacity=tokens_per_minute,
        refill_rate=tokens_per_minute / 60.0,
        burst_allowance=tokens_per_minute // 10,  # 10% burst
    )


def create_ollama_limiter(concurrent_requests: int = 3) -> SemaphoreLimiter:
    """Create rate limiter for Ollama with concurrent request limit."""
    return SemaphoreLimiter(concurrent_requests)


def create_coordinator_from_config(
    config: dict[str, RateLimitConfig],
) -> RateLimitCoordinator:
    """Create rate limit coordinator from configuration."""
    coordinator = RateLimitCoordinator()

    for provider, provider_config in config.items():
        if provider == "global":
            # Handle global configuration
            continue

        # Create appropriate limiter based on provider
        limiter: RateLimiter
        if provider.lower() == "openai":
            limiter = create_openai_limiter(
                provider_config.requests_per_minute or 1000,
                provider_config.tokens_per_minute or 40000,
            )
        elif provider.lower() == "anthropic":
            limiter = create_anthropic_limiter(
                provider_config.tokens_per_minute or 50000
            )
        elif provider.lower() == "ollama":
            limiter = create_ollama_limiter(
                provider_config.concurrent_requests or 3
            )
        else:
            # Generic provider
            limiters: list[RateLimiter] = []
            if provider_config.requests_per_minute:
                limiters.append(
                    SlidingWindowLimiter(provider_config.requests_per_minute, 60)
                )
            if provider_config.tokens_per_minute:
                limiters.append(
                    TokenBucketLimiter(
                        provider_config.tokens_per_minute,
                        provider_config.tokens_per_minute / 60.0,
                    )
                )
            if provider_config.concurrent_requests:
                limiters.append(
                    SemaphoreLimiter(provider_config.concurrent_requests)
                )

            if limiters:
                limiter = (
                    CompositeLimiter(limiters) if len(limiters) > 1 else limiters[0]
                )
            else:
                continue

        coordinator.add_provider_limiter(provider, limiter)

    return coordinator
