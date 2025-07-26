"""Circuit breaker implementation for fault tolerance."""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from threading import Lock
from typing import Any

from .exceptions import CircuitBreakerError

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failures exceeded, blocking calls
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5  # Number of failures before opening
    recovery_timeout: float = 60.0  # Seconds before trying half-open
    expected_exception: type = Exception  # Which exceptions count as failures
    success_threshold: int = 1  # Successes needed to close from half-open


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker monitoring."""

    failure_count: int = 0
    success_count: int = 0
    last_failure_time: datetime | None = None
    last_success_time: datetime | None = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    total_calls: int = 0
    total_failures: int = 0
    total_successes: int = 0
    state_changes: dict[str, datetime] = field(default_factory=dict)


class CircuitBreaker:
    """Circuit breaker for fault tolerance.

    Prevents cascading failures by temporarily blocking calls to a failing service.
    """

    def __init__(self,
                 name: str,
                 config: CircuitBreakerConfig | None = None,
                 on_state_change: Callable[[CircuitState, CircuitState], None] | None = None):
        """Initialize circuit breaker.

        Args:
            name: Identifier for this circuit breaker
            config: Configuration settings
            on_state_change: Callback when state changes
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.on_state_change = on_state_change

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float | None = None
        self._half_open_test_time: float | None = None
        self._lock = Lock()
        self._stats = CircuitBreakerStats()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            self._check_state()
            return self._state

    def _check_state(self):
        """Check if state should transition."""
        if self._state == CircuitState.OPEN:
            if self._last_failure_time and \
               time.time() - self._last_failure_time >= self.config.recovery_timeout:
                self._transition_to(CircuitState.HALF_OPEN)

    def _transition_to(self, new_state: CircuitState):
        """Transition to a new state."""
        if new_state != self._state:
            old_state = self._state
            self._state = new_state
            self._stats.state_changes[new_state.value] = datetime.utcnow()

            logger.info(f"Circuit breaker '{self.name}' transitioned from {old_state.value} to {new_state.value}")

            if self.on_state_change:
                self.on_state_change(old_state, new_state)

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker.

        Args:
            func: Async function to call
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result from func

        Raises:
            CircuitBreakerError: If circuit is open
            Exception: If func raises an exception
        """
        with self._lock:
            self._check_state()

            if self._state == CircuitState.OPEN:
                wait_time = self.config.recovery_timeout - (time.time() - self._last_failure_time)
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' is OPEN. Wait {wait_time:.1f}s before retry.",
                    provider=self.name,
                    wait_time=wait_time
                )

            self._stats.total_calls += 1

        try:
            # Execute the function
            result = await func(*args, **kwargs)

            with self._lock:
                self._on_success()

            return result

        except self.config.expected_exception:
            with self._lock:
                self._on_failure()
            raise

    def _on_success(self):
        """Handle successful call."""
        self._stats.success_count += 1
        self._stats.total_successes += 1
        self._stats.consecutive_successes += 1
        self._stats.consecutive_failures = 0
        self._stats.last_success_time = datetime.utcnow()

        if self._state == CircuitState.HALF_OPEN:
            if self._stats.consecutive_successes >= self.config.success_threshold:
                self._failure_count = 0
                self._transition_to(CircuitState.CLOSED)

    def _on_failure(self):
        """Handle failed call."""
        self._failure_count += 1
        self._stats.failure_count += 1
        self._stats.total_failures += 1
        self._stats.consecutive_failures += 1
        self._stats.consecutive_successes = 0
        self._stats.last_failure_time = datetime.utcnow()
        self._last_failure_time = time.time()

        if self._state == CircuitState.HALF_OPEN:
            self._transition_to(CircuitState.OPEN)
        elif self._failure_count >= self.config.failure_threshold:
            self._transition_to(CircuitState.OPEN)

    def get_stats(self) -> CircuitBreakerStats:
        """Get circuit breaker statistics.

        Returns:
            Copy of current statistics
        """
        with self._lock:
            # Return a copy to prevent external modification
            return CircuitBreakerStats(
                failure_count=self._stats.failure_count,
                success_count=self._stats.success_count,
                last_failure_time=self._stats.last_failure_time,
                last_success_time=self._stats.last_success_time,
                consecutive_failures=self._stats.consecutive_failures,
                consecutive_successes=self._stats.consecutive_successes,
                total_calls=self._stats.total_calls,
                total_failures=self._stats.total_failures,
                total_successes=self._stats.total_successes,
                state_changes=self._stats.state_changes.copy()
            )

    def reset(self):
        """Reset circuit breaker to closed state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._last_failure_time = None
            self._stats = CircuitBreakerStats()
            logger.info(f"Circuit breaker '{self.name}' reset")

    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self.state == CircuitState.CLOSED

    def is_open(self) -> bool:
        """Check if circuit is open (blocking calls)."""
        return self.state == CircuitState.OPEN

    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        return self.state == CircuitState.HALF_OPEN
