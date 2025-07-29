"""OpenAI provider implementation with unified interface and full compatibility."""

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
)
from .openai_errors import OpenAIErrorHandler
from .rate_limiters import (
    AcquisitionRequest,
    RateLimitCoordinator,
    create_openai_limiter,
)
from .registry import register_provider
from .streaming_adapters import SSEAdapter, StreamChunk
from .token_counters import TiktokenCounter, TokenCountRequest
from .transform_pipeline import (
    OpenAIRequestTransformer,
    OpenAIResponseParser,
    TransformPipeline,
    TransformPipelineConfig,
)
from .unified_models import UnifiedRequest, UnifiedResponse

logger = logging.getLogger(__name__)

# Enhanced model support with latest OpenAI models (2025)
MODEL_TOKEN_LIMITS = {
    # Latest GPT-4 models
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4.1": 128000,
    "gpt-4.1-mini": 128000,
    # GPT-4 variants
    "gpt-4-turbo-preview": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4-1106-preview": 128000,
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-0613": 8192,
    "gpt-4-32k-0613": 32768,
    # GPT-3.5 variants
    "gpt-3.5-turbo": 16385,
    "gpt-3.5-turbo-16k": 16385,
    "gpt-3.5-turbo-1106": 16385,
    "gpt-3.5-turbo-0125": 16385,
    "gpt-3.5-turbo-0613": 4097,
    "gpt-3.5-turbo-16k-0613": 16385,
}

# Updated pricing per 1M tokens (2025 pricing)
MODEL_PRICING = {
    # Latest models (per 1M tokens)
    "gpt-4o": (5.0, 20.0),
    "gpt-4o-mini": (0.15, 0.6),
    "gpt-4.1": (5.0, 20.0),
    "gpt-4.1-mini": (0.15, 0.6),
    # GPT-4 variants (per 1M tokens)
    "gpt-4-turbo-preview": (10.0, 30.0),
    "gpt-4-turbo": (10.0, 30.0),
    "gpt-4-1106-preview": (10.0, 30.0),
    "gpt-4": (30.0, 60.0),
    "gpt-4-32k": (60.0, 120.0),
    # GPT-3.5 variants (per 1M tokens)
    "gpt-3.5-turbo": (1.5, 2.0),
    "gpt-3.5-turbo-16k": (3.0, 4.0),
    "gpt-3.5-turbo-1106": (1.0, 2.0),
    "gpt-3.5-turbo-0125": (1.5, 2.0),
}


@register_provider
class OpenAIProvider(ModelProvider):
    """Enhanced OpenAI API provider with unified interface.

    Features:
    - Unified interface abstraction layer
    - Backward compatibility with existing ModelProvider interface
    - Advanced authentication with BearerTokenAuth
    - Integrated rate limiting with smart backoff
    - Unified token counting with tiktoken
    - Streaming support via SSEAdapter
    - Transform pipeline for request/response handling
    - Latest model support (GPT-4o, GPT-4.1)
    - 2025 pricing with cost optimization
    - Enhanced error handling and recovery
    """

    name = "openai"
    description = "OpenAI and compatible APIs with unified interface"
    api_version = "2.0"
    requires_api_key = True
    supported_models = list(MODEL_TOKEN_LIMITS.keys())
    supports_streaming = True
    default_model = "gpt-4o"

    def __init__(self, config: ProviderConfig):
        """Initialize enhanced OpenAI provider with unified components.

        Args:
            config: Provider configuration
        """
        super().__init__(config)

        # Set base URL (default to OpenAI API)
        self.base_url = config.endpoint or "https://api.openai.com/v1"
        if self.base_url.endswith("/"):
            self.base_url = self.base_url[:-1]

        # Initialize unified components
        self._init_unified_components()

        # Request settings
        self.max_retries = 3
        self.retry_delay = 1.0  # Initial retry delay in seconds
        self.timeout = aiohttp.ClientTimeout(total=config.timeout)

        # Initialize OpenAI-specific error handler
        self.error_handler = OpenAIErrorHandler(
            max_retries=self.max_retries,
            base_delay=self.retry_delay
        )

        # Initialize circuit breaker for fault tolerance
        circuit_config = CircuitBreakerConfig(
            failure_threshold=5,  # Open after 5 consecutive failures
            recovery_timeout=60.0,  # Try half-open after 60 seconds
            expected_exception=ProviderError,
            success_threshold=2  # Require 2 successes to close from half-open
        )
        self.circuit_breaker = CircuitBreaker(
            name=f"openai-{id(self)}",
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

        # Available models cache
        self._available_models: list[str] | None = None
        self._models_last_fetched: float | None = None
        self._models_cache_duration = 3600  # 1 hour

    def _init_unified_components(self) -> None:
        """Initialize unified abstraction layer components."""
        # Authentication strategy
        credentials = {"api_key": self.config.api_key or ""}
        if self.config.extra_params:
            if org_id := self.config.extra_params.get("organization"):
                credentials["organization_id"] = org_id
            if project_id := self.config.extra_params.get("project"):
                credentials["project_id"] = project_id

        self.auth_strategy = AuthStrategyFactory.create_from_provider(
            "openai", credentials
        )

        # Transform pipeline
        transformer = OpenAIRequestTransformer()
        parser = OpenAIResponseParser()
        pipeline_config = TransformPipelineConfig(
            provider_name="openai",
            default_model=self.default_model,
            supported_models=self.supported_models,
            max_tokens_limit=128000,  # Use default max context
        )
        self.transform_pipeline = TransformPipeline(
            transformer, parser, pipeline_config
        )

        # Token counter
        self.token_counter = TiktokenCounter()

        # Streaming adapter
        self.streaming_adapter = SSEAdapter(provider="openai")

        # Rate limiter
        self.rate_coordinator = RateLimitCoordinator()
        # Initialize with conservative default limits
        default_limiter = create_openai_limiter(
            requests_per_minute=60,  # Conservative for new users
            tokens_per_minute=10000
        )
        self.rate_coordinator.add_provider_limiter("openai", default_limiter)

    def _on_circuit_state_change(self, old_state: Any, new_state: Any) -> None:
        """Handle circuit breaker state changes."""
        logger.warning(
            f"OpenAI circuit breaker state changed from "
            f"{old_state.value} to {new_state.value}. "
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
        token_limit = MODEL_TOKEN_LIMITS.get(model, 128000)  # Default to higher limit
        pricing = MODEL_PRICING.get(model, (5.0, 20.0))  # Default to GPT-4o pricing

        # Determine rate limits based on model tier
        if "gpt-4o" in model or "gpt-4.1" in model:
            rate_limits = {
                "requests_per_minute": 5000,
                "tokens_per_minute": 500000,
            }
        elif "gpt-4" in model:
            rate_limits = {
                "requests_per_minute": 1000,
                "tokens_per_minute": 40000,
            }
        else:  # GPT-3.5 and others
            rate_limits = {
                "requests_per_minute": 3000,
                "tokens_per_minute": 160000,
            }

        return ProviderCapabilities(
            max_context_length=token_limit,
            supports_streaming=True,
            supports_json_mode=True,  # All modern models support JSON mode
            supports_function_calling=True,  # All models support function calling
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
        # Validate and transform request
        provider_request = self.transform_pipeline.process_request(request)

        # Estimate tokens for rate limiting
        token_estimate = await self._estimate_request_tokens(request)

        # Check rate limits
        rate_request = AcquisitionRequest(
            estimated_tokens=token_estimate,
            timeout=30.0,  # 30 second timeout for rate limit acquisition
        )

        rate_result = await self.rate_coordinator.acquire("openai", rate_request)
        if not rate_result.acquired:
            if rate_result.retry_after:
                await asyncio.sleep(rate_result.retry_after)
                # Retry once after waiting
                rate_result = await self.rate_coordinator.acquire(
                    "openai", rate_request
                )
                if not rate_result.acquired:
                    retry_after = (
                        int(rate_result.retry_after)
                        if rate_result.retry_after else None
                    )
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
                    "chat/completions", provider_request
                )
            else:
                # Handle regular request
                response_data = await self._make_request(
                    "chat/completions", provider_request
                )

            latency_ms = int((time.time() - start_time) * 1000)

            # Process response through pipeline
            if request.stream and isinstance(response_data, str):
                # For streaming, create a mock response format
                mock_response = {
                    "choices": [{
                        "message": {"content": response_data},
                        "finish_reason": "stop"
                    }],
                    "model": request.model or self.default_model,
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                }
                unified_response = self.transform_pipeline.process_response(
                    mock_response, request, latency_ms
                )
            else:
                # response_data should already be a dict[str, Any] for non-streaming
                if not isinstance(response_data, dict):
                    raise ValueError(
                        f"Expected dict response, got {type(response_data)}"
                    )
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
            self.rate_coordinator.release("openai", actual_tokens)

    async def _estimate_request_tokens(self, request: UnifiedRequest) -> int:
        """Estimate tokens for a unified request."""
        try:
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

    def _calculate_cost_unified(
        self, model: str, prompt_tokens: int, completion_tokens: int
    ) -> float:
        """Calculate cost using 2025 pricing structure."""
        pricing = MODEL_PRICING.get(model, (5.0, 20.0))  # Default to GPT-4o pricing

        # Pricing is per 1M tokens
        prompt_cost = (prompt_tokens / 1_000_000) * pricing[0]
        completion_cost = (completion_tokens / 1_000_000) * pricing[1]
        return prompt_cost + completion_cost

    async def generate_stream(
        self, request: UnifiedRequest
    ) -> AsyncIterator[StreamChunk]:
        """Generate streaming response using unified interface.

        Args:
            request: Unified request with stream=True

        Yields:
            StreamChunk objects with accumulated content
        """
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
        rate_result = await self.rate_coordinator.acquire("openai", rate_request)

        if not rate_result.acquired and rate_result.retry_after:
            await asyncio.sleep(rate_result.retry_after)

        try:
            # Make streaming request
            async for chunk in self._make_streaming_request_chunks(
                "chat/completions", provider_request
            ):
                yield chunk
        finally:
            self.rate_coordinator.release("openai", token_estimate)

    async def generate(self, prompt: str, **kwargs: Any) -> ModelResponse:
        """Generate a response from OpenAI (legacy interface).

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
        messages = kwargs.get("messages", [{"role": "user", "content": prompt}])

        # Extract system message if present in messages
        system_message = None
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
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            stream=kwargs.get("stream", False),
            model=model,
            stop_sequences=kwargs.get("stop", None)
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
                        error = await self.error_handler.classify_response_error(
                            response
                        )
                        raise error from None

                    # Use streaming adapter to parse SSE response
                    async for chunk in self.streaming_adapter.stream_response(
                        response.content
                    ):
                        yield chunk
        except Exception as e:
            # Classify and re-raise streaming errors
            if not isinstance(e, ProviderError):
                error = self.error_handler.classify_error(e)
                raise error from e
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
        """Make an HTTP request to the OpenAI API with retry logic.

        Args:
            endpoint: API endpoint (e.g., "chat/completions")
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
            sanitized_data = {k: v for k, v in data.items() if k != "messages"}
            if "messages" in data:
                sanitized_data["messages"] = f"[{len(data['messages'])} messages]"
            logger.debug(f"Request data: {sanitized_data}")

        # Use circuit breaker to wrap the request
        async def _make_single_request() -> Any:
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
                                    return await self._handle_streaming_response_legacy(
                                response
                            )
                                else:
                                    return await response.json()

                            # Handle error responses
                            # Create custom exception with response
                            http_error = aiohttp.ClientResponseError(
                                request_info=response.request_info,
                                history=response.history,
                                status=response.status
                            )
                            # Attach response for error handler
                            http_error.response = response
                            raise http_error

                except Exception as e:
                    # For HTTP errors, pass the response for detailed classification
                    response_obj = getattr(e, 'response', None)
                    if response_obj and hasattr(response_obj, 'status'):
                        classified_error = await self.error_handler.\
                            classify_response_error(response_obj)
                    else:
                        # For other errors, use general classification
                        classified_error = self.error_handler.classify_error(
                            e, response_obj
                        )
                    last_error = classified_error

                    # Check if we should retry this error
                    if (not self.error_handler.should_retry(classified_error) or
                            attempt >= self.max_retries):
                        # Don't retry or exhausted retries
                        raise classified_error from e

                    # Calculate retry delay
                    delay = self.error_handler.get_retry_delay(
                        classified_error, attempt
                    )

                    # Log retry attempt with user-friendly message
                    user_message = self.error_handler.get_user_message(classified_error)
                    logger.warning(
                        f"Request failed on attempt "
                        f"{attempt + 1}/{self.max_retries + 1}: "
                        f"{user_message}. Retrying in {delay:.1f}s"
                    )

                    await asyncio.sleep(delay)
                    continue

            # If we exhausted retries, raise the last error with user-friendly message
            if last_error:
                raise last_error
            else:
                raise ProviderError(
                    "Request failed after all retries", provider="openai"
                )

        # Execute request through circuit breaker
        return await self.circuit_breaker.call(_make_single_request)

    async def _handle_streaming_response_legacy(
        self, response: aiohttp.ClientResponse
    ) -> str:
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
            text = await response.text()
            return {"error": {"message": text or "Unknown error"}}

    def _extract_rate_limit_info(self, headers: Any) -> None:
        """Extract rate limit information from response headers.

        Args:
            headers: Response headers
        """
        # OpenAI rate limit headers
        if "x-ratelimit-limit-requests" in headers:
            self._rate_limit_info["requests_limit"] = (
                int(headers["x-ratelimit-limit-requests"])
            )
        if "x-ratelimit-remaining-requests" in headers:
            self._rate_limit_info["requests_remaining"] = (
                int(headers["x-ratelimit-remaining-requests"])
            )
        if "x-ratelimit-reset-requests" in headers:
            self._rate_limit_info["requests_reset"] = (
                headers["x-ratelimit-reset-requests"]
            )

        if "x-ratelimit-limit-tokens" in headers:
            self._rate_limit_info["tokens_limit"] = (
                int(headers["x-ratelimit-limit-tokens"])
            )
        if "x-ratelimit-remaining-tokens" in headers:
            self._rate_limit_info["tokens_remaining"] = (
                int(headers["x-ratelimit-remaining-tokens"])
            )
        if "x-ratelimit-reset-tokens" in headers:
            self._rate_limit_info["tokens_reset"] = (
                headers["x-ratelimit-reset-tokens"]
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
        """Validate the OpenAI API connection.

        Returns:
            True if connection is valid
        """
        try:
            # Try to list models as a connection test
            await self._make_request("models", method="GET")
            self._health_status = "healthy"
            return True
        except ProviderAuthenticationError:
            logger.error("OpenAI API authentication failed")
            self._health_status = "unhealthy"
            return False
        except Exception as e:
            logger.warning(f"OpenAI connection validation failed: {e}")
            self._health_status = "unhealthy"
            return False

    async def get_available_models(self) -> list[str]:
        """Get list of available models from the API.

        Returns:
            List of model IDs
        """
        # Check cache
        if (
            self._available_models is not None
            and self._models_last_fetched is not None
            and time.time() - self._models_last_fetched < self._models_cache_duration
        ):
            return self._available_models

        try:
            response = await self._make_request("models", method="GET")
            models = [model["id"] for model in response.get("data", [])]

            # Cache the result
            self._available_models = models
            self._models_last_fetched = time.time()

            return models
        except Exception as e:
            logger.warning(f"Failed to fetch available models: {e}")
            # Return default list if API call fails
            return self.supported_models

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
        return ["OPENAI_API_KEY"]

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
            api_key=env_vars["OPENAI_API_KEY"],
            model=model or os.getenv("OPENAI_MODEL") or cls.default_model,
        )

        # Optional endpoint override
        if "OPENAI_BASE_URL" in os.environ:
            config.endpoint = os.environ["OPENAI_BASE_URL"]

        # Optional organization ID
        if "OPENAI_ORG_ID" in os.environ:
            config.extra_params = {"organization": os.environ["OPENAI_ORG_ID"]}

        return config

    def get_rate_limit_info(self) -> dict[str, Any]:
        """Get current rate limit information.

        Returns:
            Dictionary with rate limit details
        """
        # Combine legacy info with unified rate limiter status
        unified_status = self.rate_coordinator.get_status("openai")
        legacy_info = self._rate_limit_info.copy()

        return {
            "legacy_info": legacy_info,
            "unified_status": unified_status,
            "provider": "openai"
        }

    def configure_rate_limits(
        self,
        requests_per_minute: int | None = None,
        tokens_per_minute: int | None = None
    ) -> None:
        """Configure rate limits for this provider.

        Args:
            requests_per_minute: Requests per minute limit
            tokens_per_minute: Tokens per minute limit
        """
        if requests_per_minute or tokens_per_minute:
            limiter = create_openai_limiter(
                requests_per_minute or 1000,
                tokens_per_minute or 40000
            )
            self.rate_coordinator.add_provider_limiter("openai", limiter)
            logger.info(
                f"Updated OpenAI rate limits: {requests_per_minute} RPM, "
                f"{tokens_per_minute} TPM"
            )

    def get_token_limit(self) -> int:
        """Get the token limit for the current model.

        Returns:
            Maximum number of tokens
        """
        model = self.config.model or self.default_model
        return MODEL_TOKEN_LIMITS.get(model, 4096)
