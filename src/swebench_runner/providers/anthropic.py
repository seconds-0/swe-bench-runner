"""Anthropic Claude provider implementation with unified interface and full compatibility."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator
from typing import Any

import aiohttp

from .auth_strategies import AuthStrategyFactory
from .base import ModelProvider, ModelResponse, ProviderCapabilities, ProviderConfig
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from .exceptions import (
    ProviderAuthenticationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderTokenLimitError,
)
from .rate_limiters import (
    AcquisitionRequest,
    RateLimitCoordinator,
    create_anthropic_limiter,
)
from .registry import register_provider
from .streaming_adapters import SSEAdapter, StreamChunk
from .token_counters import AnthropicAPICounter, TokenCountRequest
from .transform_pipeline import (
    AnthropicRequestTransformer,
    AnthropicResponseParser,
    TransformPipeline,
    TransformPipelineConfig,
)
from .unified_models import UnifiedRequest, UnifiedResponse

logger = logging.getLogger(__name__)

# Enhanced model support with latest Anthropic models (2025)
MODEL_TOKEN_LIMITS = {
    # Claude 4 models
    "claude-opus-4-20250514": 200000,
    "claude-sonnet-4-20250514": 200000,
    # Claude 3.5 models
    "claude-haiku-3-5-20241022": 200000,
    "claude-sonnet-3-5-20241022": 200000,
    # Legacy models for backward compatibility
    "claude-3-opus-20240229": 200000,
    "claude-3-sonnet-20240229": 200000,
    "claude-3-haiku-20240307": 200000,
}

# Updated pricing per 1M tokens (2025 pricing)
MODEL_PRICING = {
    # Claude 4 models (per 1M tokens)
    "claude-opus-4-20250514": (15.0, 75.0),
    "claude-sonnet-4-20250514": (3.0, 15.0),
    # Claude 3.5 models (per 1M tokens)
    "claude-haiku-3-5-20241022": (0.8, 4.0),
    "claude-sonnet-3-5-20241022": (3.0, 15.0),
    # Legacy models for backward compatibility
    "claude-3-opus-20240229": (15.0, 75.0),
    "claude-3-sonnet-20240229": (3.0, 15.0),
    "claude-3-haiku-20240307": (0.25, 1.25),
}


class AnthropicErrorHandler:
    """Anthropic-specific error handling with retry logic."""

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay

    async def classify_response_error(self, response: aiohttp.ClientResponse) -> ProviderError:
        """Classify HTTP response errors into appropriate provider exceptions."""
        try:
            error_data = await response.json()
        except Exception:
            error_data = {"error": {"message": await response.text() or "Unknown error"}}

        error_info = error_data.get("error", {})
        error_type = error_info.get("type", "unknown")
        message = error_info.get("message", "Unknown error")

        if response.status == 401:
            return ProviderAuthenticationError(
                f"Anthropic authentication failed: {message}. "
                "Check your ANTHROPIC_API_KEY environment variable."
            )
        elif response.status == 429:
            # Extract retry-after from headers if available
            retry_after = response.headers.get("retry-after")
            retry_seconds = int(retry_after) if retry_after and retry_after.isdigit() else None
            return ProviderRateLimitError(
                f"Anthropic rate limit exceeded: {message}",
                retry_after=retry_seconds
            )
        elif response.status == 400:
            # Check for specific input validation errors
            if "max_tokens" in message.lower():
                return ProviderTokenLimitError(
                    f"Anthropic token limit error: {message}. "
                    "Note: max_tokens is required for Anthropic API."
                )
            else:
                return ProviderError(f"Anthropic validation error: {message}")
        elif response.status == 422:
            return ProviderError(f"Anthropic request validation failed: {message}")
        elif response.status >= 500:
            return ProviderResponseError(
                f"Anthropic server error ({response.status}): {message}. "
                "This is likely a temporary issue with Anthropic's servers."
            )
        else:
            return ProviderError(f"Anthropic API error ({response.status}): {message}")

    def classify_error(self, error: Exception, response: Any = None) -> ProviderError:
        """Classify general exceptions into appropriate provider exceptions."""
        if isinstance(error, ProviderError):
            return error
        elif isinstance(error, asyncio.TimeoutError):
            return ProviderTimeoutError(
                "Request to Anthropic API timed out. "
                "Try increasing the timeout or check your network connection."
            )
        elif isinstance(error, aiohttp.ClientConnectorError):
            return ProviderResponseError(
                "Failed to connect to Anthropic API. "
                "Check your internet connection."
            )
        else:
            return ProviderError(f"Unexpected error with Anthropic API: {error}")

    def should_retry(self, error: ProviderError) -> bool:
        """Determine if an error should be retried."""
        # Don't retry authentication or validation errors
        if isinstance(error, (ProviderAuthenticationError, ProviderTokenLimitError)):
            return False
        # Retry rate limit, timeout, and server errors
        return isinstance(error, (ProviderRateLimitError, ProviderTimeoutError, ProviderResponseError))

    def get_retry_delay(self, error: ProviderError, attempt: int) -> float:
        """Calculate retry delay with exponential backoff."""
        base_delay = self.base_delay

        # Use retry-after for rate limit errors
        if isinstance(error, ProviderRateLimitError) and error.retry_after:
            return min(error.retry_after, 60.0)  # Cap at 60 seconds

        # Exponential backoff for other retryable errors
        return min(base_delay * (2 ** attempt), 60.0)

    def get_user_message(self, error: ProviderError) -> str:
        """Get user-friendly error message."""
        return str(error)


@register_provider
class AnthropicProvider(ModelProvider):
    """Enhanced Anthropic Claude API provider with unified interface.

    Features:
    - Unified interface abstraction layer
    - Backward compatibility with existing ModelProvider interface
    - Advanced authentication with API key headers
    - Integrated rate limiting with token bucket pattern
    - API-based token counting with /v1/messages/count_tokens
    - Streaming support via SSE with Anthropic event types
    - Transform pipeline for request/response handling
    - Latest model support (Claude Opus 4, Claude Sonnet 4)
    - 2025 pricing with cost optimization
    - Enhanced error handling with Anthropic-specific messages
    - Required max_tokens handling
    - System message separation per Anthropic API format
    """

    name = "anthropic"
    description = "Anthropic Claude API with unified interface"
    api_version = "2023-06-01"
    requires_api_key = True
    supported_models = list(MODEL_TOKEN_LIMITS.keys())
    supports_streaming = True
    default_model = "claude-sonnet-4-20250514"

    def __init__(self, config: ProviderConfig):
        """Initialize enhanced Anthropic provider with unified components.

        Args:
            config: Provider configuration
        """
        super().__init__(config)

        # Set base URL (default to Anthropic API)
        self.base_url = config.endpoint or "https://api.anthropic.com/v1"
        if self.base_url.endswith("/"):
            self.base_url = self.base_url[:-1]

        # Initialize unified components
        self._init_unified_components()

        # Request settings
        self.max_retries = 3
        self.retry_delay = 1.0  # Initial retry delay in seconds
        self.timeout = aiohttp.ClientTimeout(total=config.timeout)

        # Initialize Anthropic-specific error handler
        self.error_handler = AnthropicErrorHandler(
            max_retries=self.max_retries,
            base_delay=self.retry_delay
        )

        # Initialize circuit breaker for fault tolerance
        circuit_config = CircuitBreakerConfig(
            failure_threshold=5,  # Open after 5 consecutive failures
            recovery_timeout=60.0,  # Try half-open after 60 seconds
            expected_exception=ProviderError,  # Consider all provider errors as failures
            success_threshold=2  # Require 2 successes to close from half-open
        )
        self.circuit_breaker = CircuitBreaker(
            name=f"anthropic-{id(self)}",
            config=circuit_config,
            on_state_change=self._on_circuit_state_change
        )

        # Rate limiting tracking (legacy compatibility)
        self._rate_limit_info: dict[str, int | str | None] = {
            "requests_limit": None,
            "requests_remaining": None,
            "requests_reset": None,
            "tokens_limit": None,
            "tokens_remaining": None,
            "tokens_reset": None,
        }

    def _init_unified_components(self) -> None:
        """Initialize unified abstraction layer components."""
        # Authentication strategy with Anthropic-specific headers
        credentials = {"api_key": self.config.api_key or ""}
        if self.config.extra_params:
            # Support for beta features
            if beta_features := self.config.extra_params.get("anthropic_beta"):
                credentials["beta_features"] = beta_features

        self.auth_strategy = AuthStrategyFactory.create_from_provider(
            "anthropic", credentials
        )

        # Transform pipeline
        transformer = AnthropicRequestTransformer()
        parser = AnthropicResponseParser()
        pipeline_config = TransformPipelineConfig(
            provider_name="anthropic",
            default_model=self.default_model,
            supported_models=self.supported_models,
            max_tokens_limit=200000,  # Use default max context for Claude
        )
        self.transform_pipeline = TransformPipeline(transformer, parser, pipeline_config)

        # Token counter - will be initialized with API client later
        self.token_counter = AnthropicAPICounter()

        # Streaming adapter
        self.streaming_adapter = SSEAdapter(provider="anthropic")

        # Rate limiter
        self.rate_coordinator = RateLimitCoordinator()
        # Initialize with conservative default limits for Anthropic
        default_limiter = create_anthropic_limiter(
            tokens_per_minute=50000  # Conservative default
        )
        self.rate_coordinator.add_provider_limiter("anthropic", default_limiter)

    def _on_circuit_state_change(self, old_state, new_state) -> None:
        """Handle circuit breaker state changes."""
        logger.warning(
            f"Anthropic circuit breaker state changed from {old_state.value} to {new_state.value}. "
            f"Provider health: {new_state.value}"
        )

        # Update health status based on circuit state
        if new_state.value == "open":
            self._health_status = "unhealthy"
        elif new_state.value == "closed":
            self._health_status = "healthy"
        else:  # half-open
            self._health_status = "recovering"

    def _init_capabilities(self) -> ProviderCapabilities:
        """Initialize enhanced provider capabilities."""
        model = self.config.model or self.default_model
        token_limit = MODEL_TOKEN_LIMITS.get(model, 200000)  # Default to higher limit
        pricing = MODEL_PRICING.get(model, (3.0, 15.0))  # Default to Sonnet 4 pricing

        # Determine rate limits based on model tier
        if "opus-4" in model:
            rate_limits = {
                "requests_per_minute": 4000,
                "tokens_per_minute": 400000,
            }
        elif "sonnet-4" in model:
            rate_limits = {
                "requests_per_minute": 4000,
                "tokens_per_minute": 400000,
            }
        elif "haiku" in model:
            rate_limits = {
                "requests_per_minute": 4000,
                "tokens_per_minute": 400000,
            }
        else:  # Legacy models
            rate_limits = {
                "requests_per_minute": 1000,
                "tokens_per_minute": 80000,
            }

        return ProviderCapabilities(
            max_context_length=token_limit,
            supports_streaming=True,
            supports_json_mode=False,  # Anthropic doesn't have JSON mode
            supports_function_calling=True,  # Claude supports tool use
            rate_limits=rate_limits,
            supported_models=self.supported_models,
            cost_per_1k_prompt_tokens=pricing[0] / 1000,  # Convert to per-1k pricing
            cost_per_1k_completion_tokens=pricing[1] / 1000,
        )

    async def generate_unified(self, request: UnifiedRequest) -> UnifiedResponse:
        """Generate response using unified interface.
        
        This is the main implementation using all unified components.
        The legacy generate() method delegates to this.
        
        Args:
            request: Unified request format
            
        Returns:
            Unified response format
            
        Raises:
            Various provider exceptions
        """
        # Ensure max_tokens is set (required by Anthropic)
        if not request.max_tokens:
            request.max_tokens = 4000  # Anthropic requires max_tokens

        # Validate and transform request
        provider_request = self.transform_pipeline.process_request(request)

        # Estimate tokens for rate limiting
        token_estimate = await self._estimate_request_tokens(request)

        # Check rate limits
        rate_request = AcquisitionRequest(
            estimated_tokens=token_estimate,
            timeout=30.0,  # 30 second timeout for rate limit acquisition
        )

        rate_result = await self.rate_coordinator.acquire("anthropic", rate_request)
        if not rate_result.acquired:
            if rate_result.retry_after:
                await asyncio.sleep(rate_result.retry_after)
                # Retry once after waiting
                rate_result = await self.rate_coordinator.acquire("anthropic", rate_request)
                if not rate_result.acquired:
                    retry_after = int(rate_result.retry_after) if rate_result.retry_after else None
                    raise ProviderRateLimitError(
                        f"Rate limit exceeded: {rate_result.limited_by}",
                        retry_after=retry_after
                    )

        try:
            # Make the request with retries
            start_time = time.time()
            if request.stream:
                # Handle streaming
                response_data = await self._make_streaming_request(
                    "messages", provider_request
                )
            else:
                # Handle regular request
                response_data = await self._make_request(
                    "messages", provider_request
                )

            latency_ms = int((time.time() - start_time) * 1000)

            # Process response through pipeline
            if request.stream and isinstance(response_data, str):
                # For streaming, create a mock response format
                mock_response = {
                    "content": [{"text": response_data}],
                    "model": request.model or self.default_model,
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 0, "output_tokens": 0}
                }
                unified_response = self.transform_pipeline.process_response(
                    mock_response, request, latency_ms
                )
            else:
                unified_response = self.transform_pipeline.process_response(
                    response_data, request, latency_ms
                )

            # Calculate cost
            if unified_response.usage:
                unified_response.cost = self._calculate_cost_unified(
                    request.model or self.default_model,
                    unified_response.usage.prompt_tokens,
                    unified_response.usage.completion_tokens
                )

            return unified_response

        finally:
            # Release rate limit resources
            actual_tokens = token_estimate  # Could be refined based on actual usage
            self.rate_coordinator.release("anthropic", actual_tokens)

    async def _estimate_request_tokens(self, request: UnifiedRequest) -> int:
        """Estimate tokens for a unified request."""
        try:
            # Set up the token counter with a mock API client for the estimate
            if not self.token_counter.api_client:
                # Create a simple mock client for token counting
                class MockAPIClient:
                    async def post(self, endpoint: str, json: dict):
                        # Return a mock response for estimation
                        text_length = len(json.get("messages", [{}])[0].get("content", ""))
                        if "system" in json:
                            text_length += len(json["system"])

                        # Rough estimation: ~4 chars per token for Anthropic
                        estimated_tokens = max(1, text_length // 4)

                        class MockResponse:
                            def json(self):
                                return {"input_tokens": estimated_tokens}

                        return MockResponse()

                self.token_counter.api_client = MockAPIClient()

            token_request = TokenCountRequest(
                text=request.prompt,
                model=request.model or self.default_model,
                system_message=request.system_message,
                include_system=request.system_message is not None
            )
            result = await self.token_counter.count_tokens(token_request)
            return result.token_count
        except Exception:
            # Fallback estimation: ~4 chars per token
            text_length = len(request.prompt)
            if request.system_message:
                text_length += len(request.system_message)
            return max(1, text_length // 4)

    def _calculate_cost_unified(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost using 2025 pricing structure."""
        pricing = MODEL_PRICING.get(model, (3.0, 15.0))  # Default to Sonnet 4 pricing

        # Pricing is per 1M tokens
        prompt_cost = (prompt_tokens / 1_000_000) * pricing[0]
        completion_cost = (completion_tokens / 1_000_000) * pricing[1]
        return prompt_cost + completion_cost

    async def generate_stream(self, request: UnifiedRequest) -> AsyncIterator[StreamChunk]:
        """Generate streaming response using unified interface.
        
        Args:
            request: Unified request with stream=True
            
        Yields:
            StreamChunk objects with accumulated content
        """
        # Ensure max_tokens is set (required by Anthropic)
        if not request.max_tokens:
            request.max_tokens = 4000

        # Set streaming in request
        stream_request = UnifiedRequest(
            prompt=request.prompt,
            system_message=request.system_message,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stream=True,  # Force streaming
            model=request.model,
            stop_sequences=request.stop_sequences
        )

        # Transform to provider format
        provider_request = self.transform_pipeline.process_request(stream_request)

        # Handle rate limiting similar to regular request
        token_estimate = await self._estimate_request_tokens(stream_request)
        rate_request = AcquisitionRequest(estimated_tokens=token_estimate)
        rate_result = await self.rate_coordinator.acquire("anthropic", rate_request)

        if not rate_result.acquired and rate_result.retry_after:
            await asyncio.sleep(rate_result.retry_after)

        try:
            # Make streaming request
            async for chunk in self._make_streaming_request_chunks(
                "messages", provider_request
            ):
                yield chunk
        finally:
            self.rate_coordinator.release("anthropic", token_estimate)

    async def generate(self, prompt: str, **kwargs: Any) -> ModelResponse:
        """Generate a response from Anthropic (legacy interface).
        
        This method provides backward compatibility with the existing ModelProvider
        interface while delegating to the new unified implementation.

        Args:
            prompt: The input prompt
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            ModelResponse with generated text

        Raises:
            Various provider exceptions
        """
        # Convert legacy parameters to unified format
        model = kwargs.get("model", self.config.model or self.default_model)

        # Handle messages format if provided
        messages = kwargs.get("messages", [{"role": "user", "content": prompt}])
        system_message = kwargs.get("system")

        # Extract system message if present in messages
        user_content = prompt

        if messages and isinstance(messages, list):
            system_messages = [msg for msg in messages if msg.get("role") == "system"]
            user_messages = [msg for msg in messages if msg.get("role") == "user"]

            if system_messages:
                system_message = system_messages[0].get("content")
            if user_messages:
                user_content = user_messages[0].get("content", prompt)

        # Create unified request
        unified_request = UnifiedRequest(
            prompt=user_content,
            system_message=system_message,
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens or 4000),
            temperature=kwargs.get("temperature", self.config.temperature),
            stream=kwargs.get("stream", False),
            model=model,
            stop_sequences=kwargs.get("stop_sequences", None)
        )

        # Use unified interface
        unified_response = await self.generate_unified(unified_request)

        # Convert back to legacy format
        return ModelResponse(
            content=unified_response.content,
            model=unified_response.model,
            usage={
                "prompt_tokens": unified_response.usage.prompt_tokens,
                "completion_tokens": unified_response.usage.completion_tokens,
                "total_tokens": unified_response.usage.total_tokens,
            } if unified_response.usage else None,
            cost=unified_response.cost,
            latency_ms=unified_response.latency_ms,
            provider=unified_response.provider,
            finish_reason=unified_response.finish_reason,
            raw_response=unified_response.raw_response,
        )

    async def _make_streaming_request(
        self, endpoint: str, data: dict[str, Any]
    ) -> str:
        """Make a streaming request and return accumulated content."""
        accumulated_content = ""

        async for chunk in self._make_streaming_request_chunks(endpoint, data):
            if chunk.delta:
                accumulated_content += chunk.delta

        return accumulated_content

    async def _make_streaming_request_chunks(
        self, endpoint: str, data: dict[str, Any]
    ) -> AsyncIterator[StreamChunk]:
        """Make a streaming request and yield chunks."""
        url = f"{self.base_url}/{endpoint}"
        headers = self._prepare_headers()

        logger.debug(f"Making streaming request to {url}")

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status != 200:
                        # Use comprehensive error handler for streaming errors
                        error = await self.error_handler.classify_response_error(response)
                        raise error

                    # Use streaming adapter to parse SSE response
                    async for chunk in self.streaming_adapter.stream_response(
                        response.content
                    ):
                        yield chunk
        except Exception as e:
            # Classify and re-raise streaming errors
            if not isinstance(e, ProviderError):
                error = self.error_handler.classify_error(e)
                raise error
            raise

    def _prepare_headers(self) -> dict[str, str]:
        """Prepare request headers using auth strategy."""
        base_headers = {"Content-Type": "application/json"}
        return self.auth_strategy.prepare_headers(base_headers)

    async def _make_request(
        self,
        endpoint: str,
        data: dict[str, Any] | None = None,
        method: str = "POST"
    ) -> Any:
        """Make an HTTP request to the Anthropic API with retry logic.

        Args:
            endpoint: API endpoint (e.g., "messages")
            data: Request data
            method: HTTP method

        Returns:
            Response data

        Raises:
            Various provider exceptions
        """
        url = f"{self.base_url}/{endpoint}"
        headers = self._prepare_headers()

        # Log request (sanitized)
        logger.debug(f"Making {method} request to {url}")
        if data and logger.isEnabledFor(logging.DEBUG):
            sanitized_data = {k: v for k, v in data.items() if k not in ["messages", "system"]}
            if "messages" in data:
                sanitized_data["messages"] = f"[{len(data['messages'])} messages]"
            if "system" in data:
                sanitized_data["system"] = f"[system message: {len(data['system'])} chars]"
            logger.debug(f"Request data: {sanitized_data}")

        # Use circuit breaker to wrap the request
        async def _make_single_request():
            # Enhanced retry logic with comprehensive error handling
            last_error = None
            for attempt in range(self.max_retries + 1):
                try:
                    async with aiohttp.ClientSession(timeout=self.timeout) as session:
                        async with session.request(
                            method,
                            url,
                            headers=headers,
                            json=data if method == "POST" else None,
                        ) as response:
                            # Extract rate limit info from headers
                            self._extract_rate_limit_info(response.headers)

                            # Handle successful responses
                            if response.status == 200:
                                if data and data.get("stream"):
                                    # Handle streaming response using adapter
                                    return await self._handle_streaming_response_legacy(response)
                                else:
                                    return await response.json()

                            # Handle error responses using comprehensive error handler
                            classified_error = await self.error_handler.classify_response_error(response)
                            raise classified_error

                except Exception as e:
                    # For already classified errors, pass through
                    if isinstance(e, ProviderError):
                        classified_error = e
                    else:
                        # For other errors, use general classification
                        classified_error = self.error_handler.classify_error(e)
                    last_error = classified_error

                    # Check if we should retry this error
                    if not self.error_handler.should_retry(classified_error) or attempt >= self.max_retries:
                        # Don't retry or exhausted retries
                        raise classified_error

                    # Calculate retry delay
                    delay = self.error_handler.get_retry_delay(classified_error, attempt)

                    # Log retry attempt with user-friendly message
                    user_message = self.error_handler.get_user_message(classified_error)
                    logger.warning(
                        f"Request failed on attempt {attempt + 1}/{self.max_retries + 1}: {user_message}. "
                        f"Retrying in {delay:.1f}s"
                    )

                    await asyncio.sleep(delay)
                    continue

            # If we exhausted retries, raise the last error with user-friendly message
            if last_error:
                raise last_error
            else:
                raise ProviderError("Request failed after all retries")

        # Execute request through circuit breaker
        return await self.circuit_breaker.call(_make_single_request)

    async def _handle_streaming_response_legacy(self, response: aiohttp.ClientResponse) -> str:
        """Handle streaming response for legacy interface compatibility.

        Args:
            response: The streaming response

        Returns:
            Accumulated content from stream
        """
        accumulated_content = ""

        # Use the streaming adapter for consistent parsing
        async for chunk in self.streaming_adapter.stream_response(response.content):
            if chunk.delta:
                accumulated_content += chunk.delta

        return accumulated_content

    def _extract_rate_limit_info(self, headers: Any) -> None:
        """Extract rate limit information from response headers.

        Args:
            headers: Response headers
        """
        # Anthropic rate limit headers (if available)
        if "anthropic-ratelimit-requests-limit" in headers:
            self._rate_limit_info["requests_limit"] = (
                int(headers["anthropic-ratelimit-requests-limit"])
            )
        if "anthropic-ratelimit-requests-remaining" in headers:
            self._rate_limit_info["requests_remaining"] = (
                int(headers["anthropic-ratelimit-requests-remaining"])
            )
        if "anthropic-ratelimit-requests-reset" in headers:
            self._rate_limit_info["requests_reset"] = (
                headers["anthropic-ratelimit-requests-reset"]
            )

        if "anthropic-ratelimit-tokens-limit" in headers:
            self._rate_limit_info["tokens_limit"] = (
                int(headers["anthropic-ratelimit-tokens-limit"])
            )
        if "anthropic-ratelimit-tokens-remaining" in headers:
            self._rate_limit_info["tokens_remaining"] = (
                int(headers["anthropic-ratelimit-tokens-remaining"])
            )
        if "anthropic-ratelimit-tokens-reset" in headers:
            self._rate_limit_info["tokens_reset"] = (
                headers["anthropic-ratelimit-tokens-reset"]
            )

    def _calculate_cost(
        self, model: str, prompt_tokens: int, completion_tokens: int
    ) -> float:
        """Calculate cost for a generation (legacy interface).

        Args:
            model: Model name
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens

        Returns:
            Cost in USD
        """
        return self._calculate_cost_unified(model, prompt_tokens, completion_tokens)

    async def validate_connection(self) -> bool:
        """Validate the Anthropic API connection.

        Returns:
            True if connection is valid
        """
        try:
            # Try a minimal generation as a connection test
            test_response = await self.generate(
                "Respond with 'OK' if you receive this.",
                max_tokens=10
            )
            self._health_status = "healthy"
            return bool(test_response.content)
        except ProviderAuthenticationError:
            logger.error("Anthropic API authentication failed")
            self._health_status = "unhealthy"
            return False
        except Exception as e:
            logger.warning(f"Anthropic connection validation failed: {e}")
            self._health_status = "unhealthy"
            return False

    def estimate_cost(self, prompt_tokens: int, max_tokens: int) -> float:
        """Estimate the cost of a generation.

        Args:
            prompt_tokens: Number of tokens in the prompt
            max_tokens: Maximum number of completion tokens

        Returns:
            Estimated cost in USD
        """
        model = self.config.model or self.default_model
        return self._calculate_cost_unified(model, prompt_tokens, max_tokens)

    async def estimate_cost_unified(self, request: UnifiedRequest) -> float:
        """Estimate cost for a unified request.
        
        Args:
            request: Unified request format
            
        Returns:
            Estimated cost in USD
        """
        # Get token count estimate
        token_estimate = await self._estimate_request_tokens(request)
        max_completion = request.max_tokens or 1000  # Default estimate

        model = request.model or self.default_model
        return self._calculate_cost_unified(model, token_estimate, max_completion)

    @classmethod
    def get_required_env_vars(cls) -> list[str]:
        """Get required environment variables."""
        return ["ANTHROPIC_API_KEY"]

    @classmethod
    def _config_from_env(
        cls, env_vars: dict[str, str], model: str | None = None
    ) -> ProviderConfig:
        """Create config from environment variables.

        Args:
            env_vars: Environment variables
            model: Optional model override

        Returns:
            Provider configuration
        """
        config = ProviderConfig(
            name=cls.name,
            api_key=env_vars["ANTHROPIC_API_KEY"],
            model=model or os.getenv("ANTHROPIC_MODEL") or cls.default_model,
        )

        # Optional endpoint override
        if "ANTHROPIC_BASE_URL" in os.environ:
            config.endpoint = os.environ["ANTHROPIC_BASE_URL"]

        # Optional beta features
        if "ANTHROPIC_BETA" in os.environ:
            config.extra_params = {"anthropic_beta": os.environ["ANTHROPIC_BETA"]}

        return config

    def get_rate_limit_info(self) -> dict[str, Any]:
        """Get current rate limit information.

        Returns:
            Dictionary with rate limit details
        """
        # Combine legacy info with unified rate limiter status
        unified_status = self.rate_coordinator.get_status("anthropic")
        legacy_info = self._rate_limit_info.copy()

        return {
            "legacy_info": legacy_info,
            "unified_status": unified_status,
            "provider": "anthropic"
        }

    def configure_rate_limits(
        self,
        tokens_per_minute: int | None = None
    ) -> None:
        """Configure rate limits for this provider.
        
        Args:
            tokens_per_minute: Tokens per minute limit
        """
        if tokens_per_minute:
            limiter = create_anthropic_limiter(tokens_per_minute)
            self.rate_coordinator.add_provider_limiter("anthropic", limiter)
            logger.info(f"Updated Anthropic rate limits: {tokens_per_minute} TPM")

    def get_token_limit(self) -> int:
        """Get the token limit for the current model.

        Returns:
            Maximum number of tokens
        """
        model = self.config.model or self.default_model
        return MODEL_TOKEN_LIMITS.get(model, 200000)

    async def count_tokens(self, text: str, model: str) -> int:
        """Token counting via Anthropic API.
        
        Args:
            text: Text to count tokens for
            model: Model name for tokenization
            
        Returns:
            Number of tokens
        """
        token_request = TokenCountRequest(
            text=text,
            model=model,
            system_message=None,
            include_system=False
        )
        result = await self.token_counter.count_tokens(token_request)
        return result.token_count
