"""OpenRouter provider implementation for accessing 100+ models."""

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


@register_provider
class OpenRouterProvider(ModelProvider):
    """OpenRouter provider for accessing multiple model providers.

    Features:
    - Access to 100+ models from various providers
    - Dynamic model discovery
    - Cost optimization
    - Automatic fallback support
    - Per-model pricing
    """

    name = "openrouter"
    description = "Access 100+ models through OpenRouter"
    api_version = "1.0"
    requires_api_key = True
    supports_streaming = True
    default_model = "anthropic/claude-3-sonnet"

    # Default list of supported models (will be updated dynamically)
    supported_models = [
        # Anthropic models
        "anthropic/claude-3-opus",
        "anthropic/claude-3-sonnet",
        "anthropic/claude-3-haiku",
        "anthropic/claude-2.1",
        "anthropic/claude-2",
        "anthropic/claude-instant-1.2",
        # OpenAI models
        "openai/gpt-4-turbo",
        "openai/gpt-4-turbo-preview",
        "openai/gpt-4",
        "openai/gpt-3.5-turbo",
        "openai/gpt-3.5-turbo-16k",
        # Google models
        "google/gemini-pro",
        "google/gemini-pro-vision",
        "google/palm-2-chat-bison",
        # Meta models
        "meta-llama/llama-2-70b-chat",
        "meta-llama/llama-2-13b-chat",
        "meta-llama/codellama-34b-instruct",
        # Mistral models
        "mistralai/mistral-7b-instruct",
        "mistralai/mixtral-8x7b-instruct",
        "mistralai/mistral-medium",
        # Other models
        "nousresearch/nous-hermes-2-mixtral-8x7b-dpo",
        "phind/phind-codellama-34b-v2",
        "deepseek/deepseek-coder-33b-instruct",
    ]

    # Required headers for OpenRouter
    REQUIRED_HEADERS = {
        "HTTP-Referer": "https://github.com/princeton-nlp/SWE-bench",
        "X-Title": "SWE-bench Runner",
    }

    def __init__(self, config: ProviderConfig):
        """Initialize OpenRouter provider.

        Args:
            config: Provider configuration
        """
        super().__init__(config)

        # OpenRouter API endpoint
        self.base_url = "https://openrouter.ai/api/v1"

        # Request settings
        self.max_retries = 3
        self.retry_delay = 1.0
        self.timeout = aiohttp.ClientTimeout(total=config.timeout)

        # Model information cache
        self._models_cache: list[dict[str, Any]] | None = None
        self._models_last_fetched: float | None = None
        self._models_cache_duration = 3600  # 1 hour

        # Model pricing cache
        self._model_pricing: dict[str, tuple[float, float]] = {}

    def _init_capabilities(self) -> ProviderCapabilities:
        """Initialize provider capabilities."""
        # OpenRouter supports various models with different capabilities
        # We'll use conservative defaults that work for most models
        return ProviderCapabilities(
            max_context_length=32768,  # Many models support this or more
            supports_streaming=True,
            supports_json_mode=False,  # Varies by model
            supports_function_calling=False,  # Varies by model
            rate_limits={
                "requests_per_minute": 60,
                "tokens_per_minute": 100000,
            },
            supported_models=self.supported_models,
            cost_per_1k_prompt_tokens=0.001,  # Will be updated per model
            cost_per_1k_completion_tokens=0.002,  # Will be updated per model
        )

    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate a response through OpenRouter.

        Args:
            prompt: The input prompt
            **kwargs: Additional parameters

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
        if kwargs.get("top_p") is not None:
            params["top_p"] = kwargs["top_p"]
        if kwargs.get("frequency_penalty") is not None:
            params["frequency_penalty"] = kwargs["frequency_penalty"]
        if kwargs.get("presence_penalty") is not None:
            params["presence_penalty"] = kwargs["presence_penalty"]
        if kwargs.get("stop"):
            params["stop"] = kwargs["stop"]

        # Add OpenRouter-specific parameters
        if kwargs.get("provider_order"):
            params["provider_order"] = kwargs["provider_order"]
        if kwargs.get("provider_fallbacks"):
            params["provider_fallbacks"] = kwargs["provider_fallbacks"]

        # Make the request
        start_time = time.time()
        response_data = await self._make_request("chat/completions", params)
        latency_ms = int((time.time() - start_time) * 1000)

        # Extract response
        if params.get("stream"):
            # For streaming, response_data will be the accumulated content
            content = response_data
            usage = None
            finish_reason = "stop"
            model_used = model  # OpenRouter might use a different model
        else:
            # Regular response
            choice = response_data["choices"][0]
            content = choice["message"]["content"]
            finish_reason = choice.get("finish_reason", "stop")
            usage = response_data.get("usage")
            # OpenRouter returns the actual model used
            model_used = response_data.get("model", model)

        # Calculate cost if usage is available
        cost = None
        if usage:
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            cost = await self._calculate_cost(model_used, prompt_tokens, completion_tokens)

        return ModelResponse(
            content=content,
            model=model_used,
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
        """Make an HTTP request to OpenRouter API.

        Args:
            endpoint: API endpoint
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
            **self.REQUIRED_HEADERS,  # Add required OpenRouter headers
        }

        # Log request
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
                        # Handle different status codes
                        if response.status == 200:
                            if data and data.get("stream"):
                                return await self._handle_streaming_response(response)
                            else:
                                return await response.json()

                        elif response.status == 401:
                            error_data = await self._safe_json(response)
                            raise ProviderAuthenticationError(
                                f"Authentication failed: {error_data.get('error', {}).get('message', 'Invalid API key')}"
                            )

                        elif response.status == 429:
                            # Rate limit exceeded
                            error_data = await self._safe_json(response)
                            retry_after = int(response.headers.get("Retry-After", 60))
                            raise ProviderRateLimitError(
                                f"Rate limit exceeded: {error_data.get('error', {}).get('message', 'Too many requests')}",
                                retry_after=retry_after
                            )

                        elif response.status == 400:
                            # Bad request
                            error_data = await self._safe_json(response)
                            error_message = error_data.get("error", {}).get("message", "")

                            # Check for token limit errors
                            if "token" in error_message.lower() and "limit" in error_message.lower():
                                raise ProviderTokenLimitError(
                                    f"Token limit exceeded: {error_message}"
                                )
                            else:
                                raise ProviderResponseError(f"Bad request: {error_message}")

                        elif response.status == 402:
                            # Payment required
                            error_data = await self._safe_json(response)
                            raise ProviderError(
                                f"Payment required: {error_data.get('error', {}).get('message', 'Insufficient credits')}"
                            )

                        elif response.status >= 500:
                            # Server error - retry
                            error_data = await self._safe_json(response)
                            error_message = error_data.get("error", {}).get("message", f"Server error {response.status}")
                            last_error = ProviderConnectionError(f"Server error: {error_message}")

                            if attempt < self.max_retries - 1:
                                delay = self.retry_delay * (2 ** attempt)
                                logger.warning(f"Server error on attempt {attempt + 1}, retrying in {delay}s: {error_message}")
                                await asyncio.sleep(delay)
                                continue

                        else:
                            # Other error
                            error_data = await self._safe_json(response)
                            raise ProviderResponseError(
                                f"Unexpected status {response.status}: {error_data.get('error', {}).get('message', 'Unknown error')}"
                            )

            except asyncio.TimeoutError:
                raise ProviderTimeoutError(f"Request to {url} timed out after {self.config.timeout}s")
            except aiohttp.ClientError as e:
                last_error = ProviderConnectionError(f"Connection error: {str(e)}")
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Connection error on attempt {attempt + 1}, retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                    continue
            except (ProviderAuthenticationError, ProviderTokenLimitError, ProviderResponseError, ProviderError):
                # Don't retry these errors
                raise
            except ProviderRateLimitError as e:
                # For rate limits, wait if we have retries left
                if attempt < self.max_retries - 1 and e.retry_after:
                    logger.warning(f"Rate limited, waiting {e.retry_after}s before retry")
                    await asyncio.sleep(e.retry_after)
                    continue
                raise

        # If we exhausted retries, raise the last error
        if last_error:
            raise last_error
        else:
            raise ProviderError("Request failed after all retries")

    async def _handle_streaming_response(self, response: aiohttp.ClientResponse) -> str:
        """Handle streaming response from OpenRouter.

        Args:
            response: The streaming response

        Returns:
            Accumulated content from stream
        """
        content_parts = []

        async for line in response.content:
            line = line.decode('utf-8').strip()
            if line.startswith("data: "):
                data_str = line[6:]

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
            return await response.json()
        except Exception:
            text = await response.text()
            return {"error": {"message": text or "Unknown error"}}

    async def _fetch_available_models(self) -> list[dict[str, Any]]:
        """Fetch available models from OpenRouter API.

        Returns:
            List of model information dictionaries
        """
        # Check cache
        if (
            self._models_cache is not None
            and self._models_last_fetched is not None
            and time.time() - self._models_last_fetched < self._models_cache_duration
        ):
            return self._models_cache

        try:
            response = await self._make_request("models", method="GET")
            models = response.get("data", [])

            # Cache the result
            self._models_cache = models
            self._models_last_fetched = time.time()

            # Update pricing information
            for model in models:
                model_id = model.get("id")
                pricing = model.get("pricing", {})
                if model_id and pricing:
                    prompt_price = float(pricing.get("prompt", 0)) * 1000  # Convert to per 1K tokens
                    completion_price = float(pricing.get("completion", 0)) * 1000
                    self._model_pricing[model_id] = (prompt_price, completion_price)

            # Update supported models list
            self.supported_models = [model["id"] for model in models]

            return models
        except Exception as e:
            logger.warning(f"Failed to fetch available models: {e}")
            return []

    async def get_available_models(self) -> list[str]:
        """Get list of available model IDs.

        Returns:
            List of model IDs
        """
        models = await self._fetch_available_models()
        return [model["id"] for model in models]

    async def _filter_code_models(self) -> list[dict[str, Any]]:
        """Filter models that are suitable for code generation.

        Returns:
            List of code-capable models
        """
        models = await self._fetch_available_models()

        # Keywords that indicate code capability
        code_keywords = [
            "code", "instruct", "chat", "gpt", "claude", "gemini",
            "mixtral", "llama", "deepseek", "phind", "nous"
        ]

        code_models = []
        for model in models:
            model_id = model.get("id", "").lower()
            # Check if model name contains code-related keywords
            if any(keyword in model_id for keyword in code_keywords):
                code_models.append(model)

        return code_models

    async def select_model_by_cost(self, max_context_length: int | None = None) -> str:
        """Select the cheapest suitable model.

        Args:
            max_context_length: Required context length (optional)

        Returns:
            Model ID of the cheapest suitable model
        """
        code_models = await self._filter_code_models()

        # Filter by context length if specified
        if max_context_length:
            code_models = [
                m for m in code_models
                if m.get("context_length", 0) >= max_context_length
            ]

        if not code_models:
            return self.default_model

        # Sort by total cost (prompt + completion)
        def get_total_cost(model: dict[str, Any]) -> float:
            pricing = model.get("pricing", {})
            prompt_cost = float(pricing.get("prompt", 1.0))
            completion_cost = float(pricing.get("completion", 1.0))
            return prompt_cost + completion_cost

        code_models.sort(key=get_total_cost)

        # Return the cheapest model
        return code_models[0]["id"]

    async def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost for a generation.

        Args:
            model: Model ID
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens

        Returns:
            Cost in USD
        """
        # Try to get pricing from cache
        if model in self._model_pricing:
            pricing = self._model_pricing[model]
        else:
            # Fetch models to update pricing
            await self._fetch_available_models()
            pricing = self._model_pricing.get(model, (0.001, 0.002))

        prompt_cost = (prompt_tokens / 1000) * pricing[0]
        completion_cost = (completion_tokens / 1000) * pricing[1]
        return prompt_cost + completion_cost

    async def validate_connection(self) -> bool:
        """Validate the OpenRouter API connection.

        Returns:
            True if connection is valid
        """
        try:
            # Try to fetch models as a connection test
            models = await self._fetch_available_models()
            self._health_status = "healthy" if models else "unhealthy"
            return bool(models)
        except ProviderAuthenticationError:
            logger.error("OpenRouter API authentication failed")
            self._health_status = "unhealthy"
            return False
        except Exception as e:
            logger.warning(f"OpenRouter connection validation failed: {e}")
            self._health_status = "unhealthy"
            return False

    async def get_model_pricing(self, model: str | None = None) -> dict[str, tuple[float, float]]:
        """Get pricing information for models.

        Args:
            model: Specific model ID or None for all models

        Returns:
            Dictionary mapping model IDs to (prompt_cost, completion_cost) tuples
        """
        # Ensure we have pricing data
        await self._fetch_available_models()

        if model:
            pricing = self._model_pricing.get(model)
            return {model: pricing} if pricing else {}
        else:
            return self._model_pricing.copy()

    def estimate_cost(self, prompt_tokens: int, max_tokens: int) -> float:
        """Estimate the cost of a generation.

        Args:
            prompt_tokens: Number of tokens in the prompt
            max_tokens: Maximum number of completion tokens

        Returns:
            Estimated cost in USD
        """
        model = self.config.model or self.default_model
        # Use cached pricing if available, otherwise use defaults
        pricing = self._model_pricing.get(model, (0.001, 0.002))

        prompt_cost = (prompt_tokens / 1000) * pricing[0]
        completion_cost = (max_tokens / 1000) * pricing[1]
        return prompt_cost + completion_cost

    @classmethod
    def get_required_env_vars(cls) -> list[str]:
        """Get required environment variables."""
        return ["OPENROUTER_API_KEY"]

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
        return ProviderConfig(
            name=cls.name,
            api_key=env_vars["OPENROUTER_API_KEY"],
            model=model or os.getenv("OPENROUTER_MODEL") or cls.default_model,
        )

    async def get_model_info(self, model_id: str) -> dict[str, Any] | None:
        """Get detailed information about a specific model.

        Args:
            model_id: Model ID to look up

        Returns:
            Model information dictionary or None if not found
        """
        models = await self._fetch_available_models()
        for model in models:
            if model.get("id") == model_id:
                return model
        return None
