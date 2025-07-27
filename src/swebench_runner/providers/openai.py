"""OpenAI provider implementation with full compatibility."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

import aiohttp

from .base import ModelProvider, ModelResponse, ProviderCapabilities, ProviderConfig
from .exceptions import (
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderTokenLimitError,
)
from .registry import register_provider

logger = logging.getLogger(__name__)

# Token limits for OpenAI models (as of January 2025)
MODEL_TOKEN_LIMITS = {
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

# Pricing per 1K tokens (as of January 2025)
MODEL_PRICING = {
    # Format: (prompt_cost, completion_cost)
    "gpt-4-turbo-preview": (0.01, 0.03),
    "gpt-4-turbo": (0.01, 0.03),
    "gpt-4-1106-preview": (0.01, 0.03),
    "gpt-4": (0.03, 0.06),
    "gpt-4-32k": (0.06, 0.12),
    "gpt-3.5-turbo": (0.0005, 0.0015),
    "gpt-3.5-turbo-16k": (0.003, 0.004),
    "gpt-3.5-turbo-1106": (0.001, 0.002),
    "gpt-3.5-turbo-0125": (0.0005, 0.0015),
}


@register_provider
class OpenAIProvider(ModelProvider):
    """OpenAI API provider with full compatibility.

    Supports:
    - OpenAI API
    - Compatible endpoints (vLLM, FastChat, llama.cpp)
    - Comprehensive error handling
    - Rate limiting and retries
    - Cost calculation
    - Streaming support
    """

    name = "openai"
    description = "OpenAI and compatible APIs (vLLM, FastChat, llama.cpp)"
    api_version = "1.0"
    requires_api_key = True
    supported_models = list(MODEL_TOKEN_LIMITS.keys())
    supports_streaming = True
    default_model = "gpt-4-turbo-preview"

    def __init__(self, config: ProviderConfig):
        """Initialize OpenAI provider.

        Args:
            config: Provider configuration
        """
        super().__init__(config)

        # Set base URL (default to OpenAI API)
        self.base_url = config.endpoint or "https://api.openai.com/v1"
        if self.base_url.endswith("/"):
            self.base_url = self.base_url[:-1]

        # Set organization ID if provided
        self.organization = (
            config.extra_params.get("organization") if config.extra_params else None
        )

        # Request settings
        self.max_retries = 3
        self.retry_delay = 1.0  # Initial retry delay in seconds
        self.timeout = aiohttp.ClientTimeout(total=config.timeout)

        # Rate limiting tracking
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

    def _init_capabilities(self) -> ProviderCapabilities:
        """Initialize provider capabilities based on model."""
        model = self.config.model or self.default_model
        token_limit = MODEL_TOKEN_LIMITS.get(model, 4096)
        pricing = MODEL_PRICING.get(model, (0.01, 0.03))

        return ProviderCapabilities(
            max_context_length=token_limit,
            supports_streaming=True,
            supports_json_mode=True,  # GPT-4 Turbo supports JSON mode
            supports_function_calling=True,
            rate_limits={
                "requests_per_minute": 3 if "gpt-4" in model else 60,
                "tokens_per_minute": 10000 if "gpt-4" in model else 90000,
            },
            supported_models=self.supported_models,
            cost_per_1k_prompt_tokens=pricing[0],
            cost_per_1k_completion_tokens=pricing[1],
        )

    async def generate(self, prompt: str, **kwargs: Any) -> ModelResponse:
        """Generate a response from OpenAI.

        Args:
            prompt: The input prompt
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            ModelResponse with generated text

        Raises:
            Various provider exceptions
        """
        # Prepare the request
        model = kwargs.get("model", self.config.model or self.default_model)
        messages = kwargs.get("messages", [{"role": "user", "content": prompt}])

        # Build request parameters
        params = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }

        # Add optional parameters
        if kwargs.get("stream", False):
            params["stream"] = True
        if kwargs.get("response_format"):
            params["response_format"] = kwargs["response_format"]
        if kwargs.get("tools"):
            params["tools"] = kwargs["tools"]
        if kwargs.get("tool_choice"):
            params["tool_choice"] = kwargs["tool_choice"]
        if kwargs.get("top_p") is not None:
            params["top_p"] = kwargs["top_p"]
        if kwargs.get("frequency_penalty") is not None:
            params["frequency_penalty"] = kwargs["frequency_penalty"]
        if kwargs.get("presence_penalty") is not None:
            params["presence_penalty"] = kwargs["presence_penalty"]
        if kwargs.get("stop"):
            params["stop"] = kwargs["stop"]
        if kwargs.get("n"):
            params["n"] = kwargs["n"]

        # Make the request with retries
        start_time = time.time()
        response_data = await self._make_request("chat/completions", params)
        latency_ms = int((time.time() - start_time) * 1000)

        # Extract response
        if params.get("stream"):
            # For streaming, response_data will be the accumulated content
            content = response_data
            usage = None
            finish_reason = "stop"
        else:
            # Regular response
            choice = response_data["choices"][0]
            content = choice["message"]["content"]
            finish_reason = choice.get("finish_reason", "stop")
            usage = response_data.get("usage")

        # Calculate cost if usage is available
        cost = None
        if usage:
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            cost = self._calculate_cost(model, prompt_tokens, completion_tokens)

        return ModelResponse(
            content=content,
            model=model,
            usage=usage,
            cost=cost,
            latency_ms=latency_ms,
            provider=self.name,
            finish_reason=finish_reason,
            raw_response=response_data if not params.get("stream") else None,
        )

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
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        if self.organization:
            headers["OpenAI-Organization"] = self.organization

        # Log request (sanitized)
        logger.debug(f"Making {method} request to {url}")
        if data and logger.isEnabledFor(logging.DEBUG):
            sanitized_data = {k: v for k, v in data.items() if k != "messages"}
            if "messages" in data:
                sanitized_data["messages"] = f"[{len(data['messages'])} messages]"
            logger.debug(f"Request data: {sanitized_data}")

        # Retry logic
        last_error = None
        for attempt in range(self.max_retries):
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

                        # Handle different status codes
                        if response.status == 200:
                            if data and data.get("stream"):
                                # Handle streaming response
                                return await self._handle_streaming_response(response)
                            else:
                                return await response.json()

                        elif response.status == 401:
                            error_data = await self._safe_json(response)
                            error_msg = error_data.get('error', {}).get(
                                'message', 'Invalid API key'
                            )
                            raise ProviderAuthenticationError(
                                f"Authentication failed: {error_msg}"
                            )

                        elif response.status == 429:
                            # Rate limit exceeded
                            error_data = await self._safe_json(response)
                            retry_after = int(response.headers.get("Retry-After", 60))
                            error_msg = error_data.get('error', {}).get(
                                'message', 'Too many requests'
                            )
                            raise ProviderRateLimitError(
                                f"Rate limit exceeded: {error_msg}",
                                retry_after=retry_after
                            )

                        elif response.status == 400:
                            # Bad request - check for context length error
                            error_data = await self._safe_json(response)
                            error_message = (
                                error_data.get("error", {}).get("message", "")
                            )

                            if (
                                "context_length_exceeded" in error_message
                                or "maximum context length" in error_message
                            ):
                                # Extract token count from error message if possible
                                import re
                                token_match = re.search(r"(\d+) tokens", error_message)
                                token_count = (
                                    int(token_match.group(1)) if token_match else None
                                )

                                raise ProviderTokenLimitError(
                                    f"Token limit exceeded: {error_message}",
                                    token_count=token_count,
                                    limit=self.capabilities.max_context_length
                                )
                            else:
                                raise ProviderResponseError(
                                    f"Bad request: {error_message}"
                                )

                        elif response.status >= 500:
                            # Server error - retry
                            error_data = await self._safe_json(response)
                            error_message = error_data.get("error", {}).get(
                                "message", f"Server error {response.status}"
                            )
                            last_error = ProviderConnectionError(
                                f"Server error: {error_message}"
                            )

                            # Exponential backoff
                            if attempt < self.max_retries - 1:
                                delay = self.retry_delay * (2 ** attempt)
                                logger.warning(
                                    f"Server error on attempt {attempt + 1}, "
                                    f"retrying in {delay}s: {error_message}"
                                )
                                await asyncio.sleep(delay)
                                continue

                        else:
                            # Other error
                            error_data = await self._safe_json(response)
                            error_msg = error_data.get('error', {}).get(
                                'message', 'Unknown error'
                            )
                            raise ProviderResponseError(
                                f"Unexpected status {response.status}: {error_msg}"
                            )

            except asyncio.TimeoutError:
                raise ProviderTimeoutError(
                    f"Request to {url} timed out after {self.config.timeout}s"
                ) from None
            except aiohttp.ClientError as e:
                last_error = ProviderConnectionError(f"Connection error: {str(e)}")
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Connection error on attempt {attempt + 1}, "
                        f"retrying in {delay}s: {e}"
                    )
                    await asyncio.sleep(delay)
                    continue
            except (
                ProviderAuthenticationError,
                ProviderTokenLimitError,
                ProviderResponseError
            ):
                # Don't retry these errors
                raise
            except ProviderRateLimitError as e:
                # For rate limits, wait if we have retries left
                if attempt < self.max_retries - 1 and e.retry_after:
                    logger.warning(
                        f"Rate limited, waiting {e.retry_after}s before retry"
                    )
                    await asyncio.sleep(e.retry_after)
                    continue
                raise

        # If we exhausted retries, raise the last error
        if last_error:
            raise last_error
        else:
            raise ProviderError("Request failed after all retries")

    async def _handle_streaming_response(self, response: aiohttp.ClientResponse) -> str:
        """Handle streaming response from OpenAI API.

        Args:
            response: The streaming response

        Returns:
            Accumulated content from stream
        """
        content_parts = []

        async for line_bytes in response.content:
            line = line_bytes.decode('utf-8').strip()
            if line.startswith("data: "):
                data_str = line[6:]  # Remove "data: " prefix

                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                    if "choices" in data and data["choices"]:
                        delta = data["choices"][0].get("delta", {})
                        if "content" in delta:
                            content_parts.append(delta["content"])
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse streaming data: {data_str}")

        return "".join(content_parts)

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
        """Calculate cost for a generation.

        Args:
            model: Model name
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens

        Returns:
            Cost in USD
        """
        pricing = MODEL_PRICING.get(model)
        if not pricing:
            # Unknown model - use default pricing
            pricing = (0.01, 0.03)

        prompt_cost = (prompt_tokens / 1000) * pricing[0]
        completion_cost = (completion_tokens / 1000) * pricing[1]
        return prompt_cost + completion_cost

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
        return self._calculate_cost(model, prompt_tokens, max_tokens)

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
        return self._rate_limit_info.copy()

    def get_token_limit(self) -> int:
        """Get the token limit for the current model.

        Returns:
            Maximum number of tokens
        """
        model = self.config.model or self.default_model
        return MODEL_TOKEN_LIMITS.get(model, 4096)
