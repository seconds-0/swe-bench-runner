"""Async/sync bridge for provider operations."""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from collections.abc import Callable, Coroutine
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from threading import Lock, Thread
from typing import Any, TypeVar, Union, Optional

from .exceptions import ProviderTimeoutError

logger = logging.getLogger(__name__)

T = TypeVar('T')


class AsyncBridge:
    """Production async/sync bridge with monitoring.

    Provides a thread-safe way to call async functions from sync code,
    with proper timeout handling and resource management.
    """

    _instance: Optional[AsyncBridge] = None
    _lock = Lock()

    def __new__(cls) -> AsyncBridge:
        """Singleton pattern for shared event loop."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def __init__(self) -> None:
        """Initialize the async bridge."""
        if hasattr(self, '_initialized'):
            return

        self._initialized = True
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="async-bridge"
        )
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[Thread] = None
        self._setup_lock = Lock()
        self._call_count = 0
        self._total_time = 0.0
        self._ensure_event_loop()

    def _ensure_event_loop(self) -> None:
        """Ensure we have a dedicated event loop for async operations."""
        with self._setup_lock:
            if self._loop is None or not self._loop.is_running():
                # Create new event loop
                future: Future[asyncio.AbstractEventLoop] = self._executor.submit(
                    self._create_event_loop
                )
                self._loop = future.result(timeout=5.0)

                # Run the loop in a dedicated thread
                self._loop_thread = Thread(
                    target=self._run_event_loop,
                    name="async-bridge-loop",
                    daemon=True
                )
                self._loop_thread.start()

    def _create_event_loop(self) -> asyncio.AbstractEventLoop:
        """Create and configure event loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

    def _run_event_loop(self) -> None:
        """Run the event loop in a thread."""
        if self._loop:
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()

    def run(self,
            coro: Coroutine[Any, Any, T],
            timeout: Optional[float] = None) -> T:
        """Run async coroutine from sync code.

        Args:
            coro: Async coroutine to run
            timeout: Optional timeout in seconds

        Returns:
            Result from coroutine

        Raises:
            ProviderTimeoutError: If operation times out
            Exception: Any exception from the coroutine
        """
        if asyncio.iscoroutine(coro):
            return self._run_coroutine(coro, timeout)
        else:
            raise TypeError(f"Expected coroutine, got {type(coro)}")

    def _run_coroutine(self,
                       coro: Coroutine[Any, Any, T],
                       timeout: Optional[float]) -> T:
        """Internal method to run coroutine."""
        self._ensure_event_loop()

        if self._loop is None:
            raise RuntimeError("Event loop not initialized")

        start_time = time.time()
        self._call_count += 1

        try:
            # Schedule coroutine on the event loop
            future = asyncio.run_coroutine_threadsafe(coro, self._loop)

            # Wait for result with timeout
            try:
                result = future.result(timeout=timeout)
                return result
            except FutureTimeoutError:
                # Cancel the coroutine
                future.cancel()
                raise ProviderTimeoutError(
                    f"Operation timed out after {timeout} seconds"
                ) from None

        finally:
            self._total_time += time.time() - start_time

    def call_async(self,
                   func: Callable[..., Coroutine[Any, Any, T]],
                   *args: Any,
                   timeout: Optional[float] = None,
                   **kwargs: Any) -> T:
        """Call async function from sync code.

        Args:
            func: Async function to call
            *args: Positional arguments
            timeout: Optional timeout in seconds
            **kwargs: Keyword arguments

        Returns:
            Result from function
        """
        coro = func(*args, **kwargs)
        return self.run(coro, timeout)

    def wrap_async(self,
                   func: Callable[..., Coroutine[Any, Any, T]],
                   timeout: Optional[float] = None) -> Callable[..., T]:
        """Create sync wrapper for async function.

        Args:
            func: Async function to wrap
            timeout: Default timeout for calls

        Returns:
            Sync function that calls the async function
        """
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return self.call_async(func, *args, timeout=timeout, **kwargs)

        return wrapper

    def get_stats(self) -> dict[str, Union[int, float, bool]]:
        """Get bridge statistics.

        Returns:
            Dictionary with call statistics
        """
        avg_time = self._total_time / self._call_count if self._call_count > 0 else 0

        return {
            "call_count": self._call_count,
            "total_time": self._total_time,
            "average_time": avg_time,
            "loop_running": self._loop.is_running() if self._loop else False,
            "thread_alive": self._loop_thread.is_alive() if self._loop_thread else False
        }

    def shutdown(self) -> None:
        """Shutdown the async bridge."""
        if self._loop and self._loop.is_running():
            # Schedule loop stop
            self._loop.call_soon_threadsafe(self._loop.stop)

            # Wait for thread to finish
            if self._loop_thread and self._loop_thread.is_alive():
                self._loop_thread.join(timeout=5.0)

        # Shutdown executor
        self._executor.shutdown(wait=True)

        # Reset instance
        with self._lock:
            if self._instance is self:
                self._instance = None

    def __enter__(self) -> AsyncBridge:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.shutdown()


# Global instance for convenience
_bridge = AsyncBridge()


def run_async(coro: Coroutine[Any, Any, T],
              timeout: Optional[float] = None) -> T:
    """Convenience function to run async code from sync context.

    Args:
        coro: Coroutine to run
        timeout: Optional timeout in seconds

    Returns:
        Result from coroutine
    """
    return _bridge.run(coro, timeout)


def async_to_sync(
    timeout: Optional[float] = None
) -> Callable[[Callable[..., Coroutine[Any, Any, T]]], Callable[..., T]]:
    """Decorator to convert async function to sync.

    Args:
        timeout: Default timeout for calls

    Returns:
        Decorator function
    """
    def decorator(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., T]:
        return _bridge.wrap_async(func, timeout)

    return decorator
