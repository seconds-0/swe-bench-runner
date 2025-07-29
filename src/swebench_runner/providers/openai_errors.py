"""OpenAI-specific error handling and classification."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

import aiohttp

from .exceptions import (
    CircuitBreakerError,
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderTokenLimitError,
)

logger = logging.getLogger(__name__)


class OpenAIRateLimitError(ProviderRateLimitError):
    """OpenAI-specific rate limit error with enhanced details."""

    def __init__(
        self,
        message: str,
        retry_after: int | None = None,
        limit_type: str | None = None,
        daily_limit_reset: str | None = None,
    ):
        super().__init__(message, retry_after, provider="openai")
        self.limit_type = limit_type  # "requests" or "tokens"
        self.daily_limit_reset = daily_limit_reset


class OpenAIAuthenticationError(ProviderAuthenticationError):
    """OpenAI-specific authentication error."""

    def __init__(self, message: str, error_code: str | None = None):
        super().__init__(message, provider="openai")
        self.error_code = error_code


class OpenAIModelNotFoundError(ProviderResponseError):
    """OpenAI model not found or not accessible."""

    def __init__(self, message: str, model: str | None = None):
        super().__init__(message, provider="openai")
        self.model = model


class OpenAIContextLengthError(ProviderTokenLimitError):
    """OpenAI context length exceeded error."""

    def __init__(
        self,
        message: str,
        token_count: int | None = None,
        limit: int | None = None,
        suggested_max_tokens: int | None = None,
    ):
        super().__init__(message, token_count, limit, provider="openai")
        self.suggested_max_tokens = suggested_max_tokens


class OpenAIContentFilterError(ProviderResponseError):
    """OpenAI content policy violation error."""

    def __init__(self, message: str, filter_type: str | None = None):
        super().__init__(message, provider="openai")
        self.filter_type = filter_type  # "content_policy", "safety", etc.


class OpenAIServerError(ProviderConnectionError):
    """OpenAI server-side error."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        server_error_type: str | None = None,
    ):
        super().__init__(message, provider="openai")
        self.status_code = status_code
        self.server_error_type = server_error_type


class OpenAIErrorHandler:
    """Handles OpenAI-specific errors and integrates with unified error system."""

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        """Initialize OpenAI error handler.

        Args:
            max_retries: Maximum number of retries for transient errors
            base_delay: Base delay for exponential backoff
        """
        self.max_retries = max_retries
        self.base_delay = base_delay

    def classify_error(
        self, error: Exception, response: aiohttp.ClientResponse | None = None
    ) -> ProviderError:
        """Classify errors into unified error categories.

        Args:
            error: The original exception
            response: HTTP response if available

        Returns:
            Classified provider error with OpenAI-specific details
        """
        # Handle aiohttp errors
        if isinstance(error, aiohttp.ClientError):
            return self._classify_http_error(error, response)

        # Handle asyncio timeout
        if isinstance(error, asyncio.TimeoutError):
            return ProviderTimeoutError(
                "Request timed out. This usually indicates network issues or API "
                "server overload. Try again in a few minutes or check your "
                "internet connection."
            )

        # Handle circuit breaker errors
        if isinstance(error, CircuitBreakerError):
            return error  # Already properly classified

        # Handle existing provider errors (pass through)
        if isinstance(error, ProviderError):
            return error

        # Generic error fallback
        return ProviderError(f"Unexpected error: {str(error)}")

    def _classify_http_error(
        self, error: Exception, response: aiohttp.ClientResponse | None
    ) -> ProviderError:
        """Classify HTTP-related errors."""
        if not response:
            if isinstance(error, aiohttp.ClientConnectionError):
                return ProviderConnectionError(
                    f"Failed to connect to OpenAI API: {str(error)}. "
                    "Check your internet connection or try again later."
                )
            return ProviderConnectionError(f"Network error: {str(error)}")

        return asyncio.create_task(self.classify_response_error(response))

    async def classify_response_error(
        self, response: aiohttp.ClientResponse
    ) -> ProviderError:
        """Classify errors based on HTTP response."""
        status = response.status
        error_data = await self._safe_json(response)
        error_info = error_data.get("error", {})
        error_message = error_info.get("message", "Unknown error")
        error_type = error_info.get("type", "")
        error_code = error_info.get("code", "")

        if status == 401:
            return self._handle_authentication_error(error_message, error_code)
        elif status == 403:
            return self._handle_permission_error(error_message, error_code)
        elif status == 404:
            return self._handle_not_found_error(error_message, error_code)
        elif status == 400:
            return self._handle_bad_request_error(error_message, error_type, error_data)
        elif status == 429:
            return await self._handle_rate_limit_error(
                response, error_message, error_type
            )
        elif 500 <= status <= 599:
            return self._handle_server_error(status, error_message, error_type)
        else:
            return ProviderResponseError(f"HTTP {status}: {error_message}")

    def _handle_authentication_error(
        self, message: str, code: str
    ) -> OpenAIAuthenticationError:
        """Handle authentication errors with actionable guidance."""
        if "invalid" in message.lower() or "unauthorized" in message.lower():
            user_message = (
                f"Invalid OpenAI API key: {message}. "
                "Please check that your OPENAI_API_KEY environment variable is set "
                "correctly. "
                "You can find your API key at https://platform.openai.com/api-keys"
            )
        else:
            user_message = (
                f"Authentication failed: {message}. "
                "Verify your API key is valid and has the necessary permissions."
            )

        return OpenAIAuthenticationError(user_message, code)

    def _handle_permission_error(
        self, message: str, code: str
    ) -> ProviderResponseError:
        """Handle permission/access errors."""
        if "billing" in message.lower() or "quota" in message.lower():
            user_message = (
                f"Billing or quota issue: {message}. "
                "Check your OpenAI account billing status at https://platform.openai.com/usage"
            )
        elif "organization" in message.lower():
            user_message = (
                f"Organization access issue: {message}. "
                "Verify your API key has access to the requested organization."
            )
        else:
            user_message = f"Permission denied: {message}"

        return ProviderResponseError(user_message)

    def _handle_not_found_error(
        self, message: str, code: str
    ) -> OpenAIModelNotFoundError:
        """Handle model not found errors."""
        # Extract model name if present - handle various formats
        model_match = (
            re.search(r"Model\s+'([^']+)'", message) or  # "Model 'gpt-5' not found"
            re.search(r"Model\s+\"([^\"]+)\"", message) or  # "Model "gpt-5" not found"
            re.search(
                r"model['\"]?\s*[:\-]\s*['\"]?([^'\"\\s]+)", message, re.IGNORECASE
            )  # Other formats
        )
        model = model_match.group(1) if model_match else None

        if model:
            user_message = (
                f"Model '{model}' not found or not accessible: {message}. "
                f"Use a supported model like 'gpt-4o', 'gpt-4', or 'gpt-3.5-turbo'. "
                f"Check available models at https://platform.openai.com/docs/models"
            )
        else:
            user_message = (
                f"Resource not found: {message}. "
                "Check that the model name is correct and you have access to it."
            )

        return OpenAIModelNotFoundError(user_message, model)

    def _handle_bad_request_error(
        self, message: str, error_type: str, error_data: dict
    ) -> ProviderError:
        """Handle bad request errors (400)."""
        # Context length exceeded
        if (
            "context_length_exceeded" in error_type
            or "maximum context length" in message.lower()
            or "context length" in message.lower()
        ):
            return self._handle_context_length_error(message, error_data)

        # Content policy violation
        if (
            "content_policy_violation" in error_type
            or "content policy" in message.lower()
            or "violates" in message.lower()
        ):
            return self._handle_content_filter_error(message, error_type)

        # Invalid parameters
        if "invalid" in message.lower() and "parameter" in message.lower():
            user_message = (
                f"Invalid request parameters: {message}. "
                "Check your request format and parameter values."
            )
            return ProviderResponseError(user_message)

        # Generic bad request
        return ProviderResponseError(f"Bad request: {message}")

    def _handle_context_length_error(
        self, message: str, error_data: dict
    ) -> OpenAIContextLengthError:
        """Handle context length exceeded errors."""
        # Extract token counts from error message
        # Look for patterns like "you requested 10000 tokens" or
        # "However, you requested 10000 tokens"
        requested_match = re.search(
            r"(?:you\s+)?requested\s+(\d+)\s*tokens?", message, re.IGNORECASE
        )
        # Look for patterns like "maximum context length is 8192" or
        # "maximum.*?8192"
        limit_match = re.search(
            r"maximum.*?(?:is\s+)?(\d+)", message, re.IGNORECASE
        )

        token_count = int(requested_match.group(1)) if requested_match else None
        limit = int(limit_match.group(1)) if limit_match else None

        # Suggest a reasonable max_tokens value
        suggested_max_tokens = None
        if token_count and limit:
            # Leave some room for the response
            suggested_max_tokens = max(100, limit - token_count - 100)

        user_message = (
            f"Input too long: {message}. "
            f"Try reducing your prompt length"
        )

        if suggested_max_tokens:
            user_message += f" or set max_tokens to {suggested_max_tokens} or less"

        user_message += ". Consider breaking large inputs into smaller chunks."

        return OpenAIContextLengthError(
            user_message, token_count, limit, suggested_max_tokens
        )

    def _handle_content_filter_error(
        self, message: str, error_type: str
    ) -> OpenAIContentFilterError:
        """Handle content policy violations."""
        filter_type = "content_policy"
        if "safety" in error_type.lower():
            filter_type = "safety"

        user_message = (
            f"Content policy violation: {message}. "
            "Please revise your prompt to comply with OpenAI's usage policies. "
            "See https://platform.openai.com/docs/usage-policies for guidelines."
        )

        return OpenAIContentFilterError(user_message, filter_type)

    async def _handle_rate_limit_error(
        self, response: aiohttp.ClientResponse, message: str, error_type: str
    ) -> OpenAIRateLimitError:
        """Handle rate limit errors with intelligent backoff."""
        # Extract retry-after from headers
        retry_after = self._extract_retry_after(response.headers)

        # Determine limit type
        limit_type = "requests"
        if "token" in message.lower() or "tpm" in error_type.lower():
            limit_type = "tokens"

        # Extract daily limit reset time
        daily_reset = response.headers.get("x-ratelimit-reset-requests")

        # Create user-friendly message
        if limit_type == "tokens":
            user_message = (
                f"Token rate limit exceeded: {message}. "
                f"Wait {retry_after}s before retrying, or reduce your request size. "
                "Consider using a smaller model or breaking large requests into chunks."
            )
        else:
            user_message = (
                f"Request rate limit exceeded: {message}. "
                f"Wait {retry_after}s before retrying, or reduce your request "
                "frequency."
            )

        if daily_reset:
            user_message += f" Daily limits reset at {daily_reset}."

        return OpenAIRateLimitError(user_message, retry_after, limit_type, daily_reset)

    def _handle_server_error(
        self, status: int, message: str, error_type: str
    ) -> OpenAIServerError:
        """Handle server-side errors."""
        server_type = "internal_error"
        if status == 502:
            server_type = "bad_gateway"
        elif status == 503:
            server_type = "service_unavailable"
        elif status == 504:
            server_type = "gateway_timeout"

        user_message = (
            f"OpenAI server error ({status}): {message}. "
            "This is a temporary issue with OpenAI's servers. "
            "Please try again in a few minutes."
        )

        return OpenAIServerError(user_message, status, server_type)

    def should_retry(self, error: ProviderError) -> bool:
        """Determine if an error should be retried.

        Args:
            error: The provider error

        Returns:
            True if the error is retryable
        """
        # Never retry authentication or permission errors
        if isinstance(error, OpenAIAuthenticationError | ProviderAuthenticationError):
            return False

        # Never retry content policy violations
        if isinstance(error, OpenAIContentFilterError):
            return False

        # Never retry context length errors (requires input modification)
        if isinstance(error, OpenAIContextLengthError):
            return False

        # Never retry model not found errors
        if isinstance(error, OpenAIModelNotFoundError):
            return False

        # Retry rate limits with backoff
        if isinstance(error, OpenAIRateLimitError | ProviderRateLimitError):
            return True

        # Retry server errors
        if isinstance(error, OpenAIServerError | ProviderConnectionError):
            return True

        # Retry timeouts
        if isinstance(error, ProviderTimeoutError):
            return True

        # Don't retry other response errors
        if isinstance(error, ProviderResponseError):
            return False

        # Default to retry for unknown errors
        return True

    def get_retry_delay(self, error: ProviderError, attempt: int) -> float:
        """Calculate retry delay with exponential backoff.

        Args:
            error: The provider error
            attempt: Current attempt number (0-based)

        Returns:
            Delay in seconds
        """
        # Use specific retry-after for rate limits
        if isinstance(error, OpenAIRateLimitError | ProviderRateLimitError):
            if error.retry_after:
                # Add small jitter to avoid thundering herd
                jitter = min(5.0, error.retry_after * 0.1)
                return error.retry_after + (
                    jitter * (0.5 + 0.5 * hash(time.time()) % 1)
                )
            # Fallback to exponential backoff for rate limits without retry-after
            return min(60.0, self.base_delay * (3 ** attempt))

        # Longer delays for server errors
        if isinstance(error, OpenAIServerError | ProviderConnectionError):
            return min(30.0, self.base_delay * (2 ** attempt))

        # Standard exponential backoff for other retryable errors
        return min(16.0, self.base_delay * (2 ** attempt))

    def get_user_message(self, error: ProviderError) -> str:
        """Get user-friendly error message with actionable steps.

        Args:
            error: The provider error

        Returns:
            User-friendly error message
        """
        # OpenAI-specific errors already have user-friendly messages
        if isinstance(
            error,
            (
                OpenAIRateLimitError
                | OpenAIAuthenticationError
                | OpenAIModelNotFoundError
                | OpenAIContextLengthError
                | OpenAIContentFilterError
                | OpenAIServerError
            ),
        ):
            return str(error)

        # Generic provider errors with OpenAI-specific guidance
        if isinstance(error, ProviderTimeoutError):
            return (
                f"{str(error)} "
                "This may be due to network issues or high API load. "
                "Try again in a few minutes."
            )

        if isinstance(error, ProviderConnectionError):
            return (
                f"{str(error)} "
                "Check your internet connection and OpenAI API status at "
                "https://status.openai.com/"
            )

        # Generic fallback
        return str(error)

    def _extract_retry_after(self, headers: Any) -> int:
        """Extract retry-after value from headers.

        Args:
            headers: Response headers

        Returns:
            Retry delay in seconds
        """
        # Try standard Retry-After header
        if "retry-after" in headers:
            try:
                return int(headers["retry-after"])
            except (ValueError, TypeError):
                pass

        # Try OpenAI-specific rate limit headers
        for header in ["x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"]:
            if header in headers:
                try:
                    # These headers contain timestamp, calculate delay
                    reset_time = float(headers[header])
                    current_time = time.time()
                    delay = max(1, int(reset_time - current_time))
                    return min(300, delay)  # Cap at 5 minutes
                except (ValueError, TypeError):
                    continue

        # Default fallback
        return 60

    async def _safe_json(self, response: aiohttp.ClientResponse) -> dict[str, Any]:
        """Safely parse JSON from response.

        Args:
            response: HTTP response

        Returns:
            Parsed JSON or error dict
        """
        try:
            return await response.json()  # type: ignore[no-any-return]
        except Exception:
            try:
                text = await response.text()
                return {"error": {"message": text or "Unknown error"}}
            except Exception:
                return {"error": {"message": f"HTTP {response.status}"}}
