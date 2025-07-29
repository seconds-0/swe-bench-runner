"""Ollama provider implementation for local model execution.

This module provides a complete Ollama provider that integrates with the unified
abstraction layer for local model execution. Unlike cloud providers, Ollama runs
locally with different constraints (hardware-limited, no auth, different streaming).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import aiohttp

from .auth_strategies import NoAuth
from .base import ModelProvider, ModelResponse, ProviderCapabilities, ProviderConfig
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from .exceptions import (
    ProviderConnectionError,
    ProviderError,
    ProviderResponseError,
    ProviderTimeoutError,
)
from .rate_limiters import create_ollama_limiter
from .registry import register_provider
from .streaming_adapters import JSONLinesAdapter, StreamChunk
from .token_counters import MetadataTokenCounter
from .transform_pipeline import (
    OllamaRequestTransformer,
    OllamaResponseParser,
    TransformPipeline,
    TransformPipelineConfig,
)
from .unified_models import UnifiedRequest, UnifiedResponse

logger = logging.getLogger(__name__)

# Popular Ollama models in 2025 with their typical memory requirements
OLLAMA_MODELS = {
    # DeepSeek models (2025 trending)
    "deepseek-r1": {"memory_gb": 16, "description": "DeepSeek R1 - reasoning model"},
    "deepseek-r1:7b": {"memory_gb": 8, "description": "DeepSeek R1 7B"},
    "deepseek-r1:14b": {"memory_gb": 16, "description": "DeepSeek R1 14B"},

    # Llama 3.3 and 3.2 models
    "llama3.3": {"memory_gb": 42, "description": "Llama 3.3 70B (default)"},
    "llama3.3:70b": {"memory_gb": 42, "description": "Llama 3.3 70B"},
    "llama3.3:8b": {"memory_gb": 8, "description": "Llama 3.3 8B"},
    "llama3.2": {"memory_gb": 4, "description": "Llama 3.2 3B (default)"},
    "llama3.2:3b": {"memory_gb": 4, "description": "Llama 3.2 3B"},
    "llama3.2:1b": {"memory_gb": 2, "description": "Llama 3.2 1B"},

    # Code-focused models
    "codellama": {"memory_gb": 16, "description": "CodeLlama 13B (default)"},
    "codellama:7b": {"memory_gb": 8, "description": "CodeLlama 7B"},
    "codellama:13b": {"memory_gb": 16, "description": "CodeLlama 13B"},
    "codellama:34b": {"memory_gb": 24, "description": "CodeLlama 34B"},

    # Other popular models
    "mistral": {"memory_gb": 8, "description": "Mistral 7B"},
    "phi-4": {"memory_gb": 8, "description": "Microsoft Phi-4"},
    "qwen2.5": {"memory_gb": 8, "description": "Qwen 2.5 7B (default)"},
    "qwen2.5:7b": {"memory_gb": 8, "description": "Qwen 2.5 7B"},
    "qwen2.5:14b": {"memory_gb": 16, "description": "Qwen 2.5 14B"},
    "gemma2": {"memory_gb": 6, "description": "Google Gemma2 9B"},
}

# Default Ollama endpoints
DEFAULT_ENDPOINTS = {
    "base": "http://localhost:11434",
    "generate": "/api/generate",
    "tags": "/api/tags",
    "pull": "/api/pull",
    "show": "/api/show",
    "ps": "/api/ps",
}


@register_provider
class OllamaProvider(ModelProvider):
    """Ollama provider for local model execution with unified interface.

    Features:
    - Local model execution (no cloud dependency)
    - Model management (pull, list, delete)
    - Hardware resource monitoring
    - JSON Lines streaming support
    - NoAuth authentication strategy (no API keys)
    - MetadataTokenCounter for token counting from response
    - Free cost estimation (hardware cost only)
    - Popular 2025 model support (DeepSeek R1, Llama 3.3, etc.)
    """

    name = "ollama"
    description = "Ollama local model execution with hardware constraints"
    api_version = "1.0"
    requires_api_key = False
    supported_models = list(OLLAMA_MODELS.keys())
    supports_streaming = True
    default_model = "llama3.3:8b"  # Good balance of capability and resource usage

    def __init__(self, config: ProviderConfig):
        """Initialize Ollama provider with unified components.

        Args:
            config: Provider configuration (api_key not required)
        """
        # Override config to not require API key
        config.api_key = None
        super().__init__(config)

        # Set base URL (default to localhost Ollama)
        self.base_url = config.endpoint or DEFAULT_ENDPOINTS["base"]
        if self.base_url.endswith("/"):
            self.base_url = self.base_url[:-1]

        # Initialize unified components
        self._init_unified_components()

        # Request settings for local execution
        self.max_retries = 2  # Lower retries for local
        self.retry_delay = 0.5  # Faster retry for local
        self.timeout = aiohttp.ClientTimeout(total=config.timeout)

        # Initialize circuit breaker for local service monitoring
        circuit_config = CircuitBreakerConfig(
            failure_threshold=3,  # Open after 3 consecutive failures
            recovery_timeout=30.0,  # Try half-open after 30 seconds (local is faster)
            expected_exception=ProviderError,
            success_threshold=1  # Just 1 success to close (local is more reliable)
        )
        self.circuit_breaker = CircuitBreaker(
            name=f"ollama-{id(self)}",
            config=circuit_config,
            on_state_change=self._on_circuit_state_change
        )

        # Track loaded models for performance
        self._loaded_models: set[str] = set()
        self._last_health_check = 0.0
        self._health_cache_duration = 30.0  # Cache health for 30 seconds

    def _init_unified_components(self) -> None:
        """Initialize all unified abstraction components for Ollama."""
        # Authentication (no auth needed for local Ollama)
        from .auth_strategies import AuthConfig, AuthType
        auth_config = AuthConfig(
            auth_type=AuthType.NONE,
            credentials={}
        )
        self.auth_strategy = NoAuth(auth_config)

        # Transform pipeline for request/response handling
        transformer = OllamaRequestTransformer()
        parser = OllamaResponseParser()
        pipeline_config = TransformPipelineConfig(
            provider_name="ollama",
            default_model=self.default_model,
            supported_models=self.supported_models,
            max_tokens_limit=8192,  # Conservative limit for local models
            temperature_range=(0.0, 2.0)
        )
        self.transform_pipeline = TransformPipeline(
            transformer, parser, pipeline_config
        )

        # Token counting (extract from response metadata)
        self.token_counter = MetadataTokenCounter()

        # Streaming adapter for JSON Lines format
        self.streaming_adapter = JSONLinesAdapter()

        # Rate limiter (concurrent request limiting for local hardware)
        self.rate_limiter = create_ollama_limiter(concurrent_requests=3)

    def _init_capabilities(self) -> ProviderCapabilities:
        """Initialize Ollama-specific capabilities."""
        return ProviderCapabilities(
            max_context_length=8192,  # Conservative for local models
            supports_streaming=True,
            supports_json_mode=False,  # Ollama doesn't have structured output
            supports_function_calling=False,
            rate_limits={"concurrent_requests": 3},  # Hardware constraint
            supported_models=self.supported_models,
            cost_per_1k_prompt_tokens=0.0,  # Free but hardware cost
            cost_per_1k_completion_tokens=0.0
        )

    def _on_circuit_state_change(self, old_state: Any, new_state: Any) -> None:
        """Handle circuit breaker state changes."""
        logger.info(f"Ollama circuit breaker state: {old_state} -> {new_state}")
        if new_state == "open":
            logger.warning("Ollama service appears to be down or unresponsive")
        elif new_state == "closed":
            logger.info("Ollama service connection restored")

    @classmethod
    def get_required_env_vars(cls) -> list[str]:
        """Get list of required environment variables (none for Ollama)."""
        return []  # No API key or env vars required

    @classmethod
    def _config_from_env(
        cls, env_vars: dict[str, str], model: str | None = None
    ) -> ProviderConfig:
        """Create provider config from environment variables.

        Args:
            env_vars: Dictionary of environment variables (ignored for Ollama)
            model: Optional model override

        Returns:
            ProviderConfig instance for Ollama
        """
        return ProviderConfig(
            name="ollama",
            api_key=None,  # No API key needed
            endpoint=None,  # Use default localhost
            model=model or cls.default_model,
            temperature=0.7,
            max_tokens=1000,
            timeout=120
        )

    async def generate(self, prompt: str, **kwargs: Any) -> ModelResponse:
        """Legacy interface for backward compatibility.

        Args:
            prompt: Input prompt
            **kwargs: Additional parameters

        Returns:
            ModelResponse in legacy format
        """
        # Convert to unified request
        unified_request = UnifiedRequest(
            prompt=prompt,
            model=kwargs.get('model', self.config.model),
            max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
            temperature=kwargs.get('temperature', self.config.temperature),
            system_message=kwargs.get('system_message'),
            stream=False
        )

        # Use unified interface
        unified_response = await self.generate_unified(unified_request)

        # Convert to legacy format
        return ModelResponse(
            content=unified_response.content,
            model=unified_response.model,
            usage={
                "prompt_tokens": unified_response.usage.prompt_tokens,
                "completion_tokens": unified_response.usage.completion_tokens,
                "total_tokens": unified_response.usage.total_tokens
            },
            cost=unified_response.cost,
            latency_ms=unified_response.latency_ms,
            provider=unified_response.provider,
            finish_reason=unified_response.finish_reason,
            raw_response=unified_response.raw_response
        )

    async def generate_unified(self, request: UnifiedRequest) -> UnifiedResponse:
        """Main unified interface for generation.

        Args:
            request: Unified request format

        Returns:
            UnifiedResponse with generated content

        Raises:
            ProviderError: For various Ollama-specific errors
        """
        # Validate service availability
        if not await self._check_service_health():
            raise ProviderConnectionError(
                "Ollama service is not running. Start it with: ollama serve",
                provider="ollama"
            )

        # Check if model is available locally
        if request.model and not await self._check_model_available(request.model):
            logger.info(
                f"Model {request.model} not found locally, attempting to pull..."
            )
            await self.pull_model(request.model)

        # Acquire rate limit before processing
        start_time = time.time()

        try:
            # Transform request to Ollama format
            ollama_request = self.transform_pipeline.process_request(request)

            # Make API call
            raw_response = await self._make_api_call(
                "POST",
                DEFAULT_ENDPOINTS["generate"],
                ollama_request
            )

            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)

            # Transform response to unified format
            unified_response = self.transform_pipeline.process_response(
                raw_response, request, latency_ms
            )

            # Add cost estimation (free but note hardware usage)
            unified_response.cost = self.estimate_cost(
                unified_response.usage.prompt_tokens,
                unified_response.usage.completion_tokens
            )

            return unified_response

        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            if isinstance(e, ProviderError):
                raise
            raise ProviderError(
                f"Ollama generation failed: {e}", provider="ollama"
            ) from e

    async def generate_stream(
        self, request: UnifiedRequest
    ) -> AsyncIterator[StreamChunk]:
        """Streaming generation using JSON Lines format.

        Args:
            request: Unified request format

        Yields:
            StreamChunk objects with incremental content

        Raises:
            ProviderError: For various Ollama-specific errors
        """
        # Validate service availability
        if not await self._check_service_health():
            raise ProviderConnectionError(
                "Ollama service is not running. Start it with: ollama serve",
                provider="ollama"
            )

        # Check model availability
        if request.model and not await self._check_model_available(request.model):
            logger.info(
                f"Model {request.model} not found locally, attempting to pull..."
            )
            await self.pull_model(request.model)

        # Enable streaming in request
        request.stream = True

        try:
            # Transform request to Ollama format
            ollama_request = self.transform_pipeline.process_request(request)

            # Make streaming API call
            async for chunk in self._stream_api_call(
                "POST",
                DEFAULT_ENDPOINTS["generate"],
                ollama_request
            ):
                yield chunk

                # Stop if stream is complete
                if chunk.done:
                    break

        except Exception as e:
            logger.error(f"Ollama streaming failed: {e}")
            if isinstance(e, ProviderError):
                raise
            raise ProviderError(
                f"Ollama streaming failed: {e}", provider="ollama"
            ) from e

    def estimate_cost(self, prompt_tokens: int, max_tokens: int) -> float:
        """Estimate cost for Ollama (free but hardware cost).

        Args:
            prompt_tokens: Number of tokens in prompt
            max_tokens: Maximum completion tokens

        Returns:
            Cost estimate (0.0 for free local execution)
        """
        # Ollama is free but has hardware costs
        # Could estimate electricity/depreciation costs in future
        return 0.0

    async def pull_model(self, model: str) -> bool:
        """Download/pull model if not available locally.

        Args:
            model: Model name to pull

        Returns:
            True if successful, False otherwise

        Raises:
            ProviderError: If pull fails
        """
        logger.info(f"Pulling model {model} from Ollama registry...")

        try:
            pull_request = {"name": model, "stream": False}

            # Make pull API call
            response = await self._make_api_call(
                "POST",
                DEFAULT_ENDPOINTS["pull"],
                pull_request
            )

            # Check for successful pull
            if response.get("status") == "success" or "error" not in response:
                logger.info(f"Successfully pulled model {model}")
                self._loaded_models.add(model)
                return True
            else:
                error_msg = response.get("error", "Unknown error during model pull")
                raise ProviderError(
                    f"Failed to pull model {model}: {error_msg}", provider="ollama"
                )

        except Exception as e:
            logger.error(f"Model pull failed for {model}: {e}")
            if isinstance(e, ProviderError):
                raise
            raise ProviderError(
                f"Failed to pull model {model}: {e}", provider="ollama"
            ) from e

    async def list_models(self) -> list[str]:
        """List available local models.

        Returns:
            List of locally available model names

        Raises:
            ProviderConnectionError: If service is unavailable
        """
        try:
            response = await self._make_api_call("GET", DEFAULT_ENDPOINTS["tags"], None)

            models = []
            if "models" in response:
                for model_info in response["models"]:
                    models.append(model_info["name"])

            # Update loaded models cache
            self._loaded_models = set(models)

            return models

        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            raise ProviderConnectionError(
                f"Failed to list Ollama models: {e}. "
                "Ensure Ollama service is running.",
                provider="ollama"
            ) from e

    async def check_health(self) -> dict[str, Any]:
        """Check Ollama service health and resource usage.

        Returns:
            Dictionary with health status information
        """
        current_time = time.time()

        # Use cached health status if recent
        if (current_time - self._last_health_check) < self._health_cache_duration:
            return await super().health_check()

        try:
            # Check service availability
            service_up = await self._check_service_health()

            health_info = {
                "status": "healthy" if service_up else "unhealthy",
                "provider": self.name,
                "model": self.config.model or "unknown",
                "timestamp": current_time,
                "service_url": self.base_url,
                "models_loaded": len(self._loaded_models),
                "concurrent_limit": 3,
                "hardware_limited": True,
                "requires_local_service": True
            }

            if service_up:
                # Get additional info if service is up
                try:
                    models = await self.list_models()
                    health_info["available_models"] = len(models)

                    # Check running models/processes
                    ps_response = await self._make_api_call(
                        "GET", DEFAULT_ENDPOINTS["ps"], None
                    )
                    running_models = ps_response.get("models", [])
                    health_info["running_models"] = len(running_models)

                except Exception:
                    # Don't fail health check if extra info fails
                    pass

            self._last_health_check = current_time
            self._health_status = health_info["status"]

            return health_info

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "provider": self.name,
                "error": str(e),
                "timestamp": current_time,
                "service_url": self.base_url
            }

    async def _check_service_health(self) -> bool:
        """Check if Ollama service is running and responsive."""
        try:
            # Simple health check - try to list models
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                url = f"{self.base_url}{DEFAULT_ENDPOINTS['tags']}"
                async with session.get(url) as response:
                    return response.status == 200
        except Exception:
            return False

    async def _check_model_available(self, model: str) -> bool:
        """Check if model is available locally."""
        if model in self._loaded_models:
            return True

        try:
            available_models = await self.list_models()
            return model in available_models
        except Exception:
            return False

    async def _make_api_call(
        self, method: str, endpoint: str, data: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Make API call to Ollama service.

        Args:
            method: HTTP method
            endpoint: API endpoint path
            data: Request data

        Returns:
            Response data as dictionary

        Raises:
            ProviderError: For various API errors
        """
        url = f"{self.base_url}{endpoint}"
        headers = {"Content-Type": "application/json"}

        # Apply auth strategy (NoAuth - no headers added)
        headers = self.auth_strategy.prepare_headers(headers)

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                kwargs = {"headers": headers}
                if data is not None:
                    kwargs["json"] = data

                async with session.request(method, url, **kwargs) as response:
                    response_text = await response.text()

                    if response.status == 200:
                        try:
                            return json.loads(response_text)
                        except json.JSONDecodeError as e:
                            raise ProviderResponseError(
                                f"Invalid JSON response from Ollama: {e}",
                                provider="ollama"
                            ) from e
                    else:
                        # Handle Ollama error responses
                        await self._handle_api_error(response.status, response_text)
                        # This line should not be reached due to exception
                        raise ProviderError("API call failed", provider="ollama")

        except aiohttp.ClientConnectionError as e:
            raise ProviderConnectionError(
                f"Cannot connect to Ollama service at {self.base_url}. "
                "Ensure Ollama is running with: ollama serve",
                provider="ollama"
            ) from e
        except asyncio.TimeoutError as e:
            raise ProviderTimeoutError(
                f"Ollama request timed out after {self.timeout.total} seconds",
                provider="ollama"
            ) from e
        except Exception as e:
            if isinstance(e, ProviderError):
                raise
            raise ProviderError(
                f"Ollama API call failed: {e}", provider="ollama"
            ) from e

    async def _stream_api_call(
        self, method: str, endpoint: str, data: dict[str, Any]
    ) -> AsyncIterator[StreamChunk]:
        """Make streaming API call to Ollama service.

        Args:
            method: HTTP method
            endpoint: API endpoint path
            data: Request data

        Yields:
            StreamChunk objects with incremental content
        """
        url = f"{self.base_url}{endpoint}"
        headers = {"Content-Type": "application/json"}

        # Apply auth strategy (NoAuth - no headers added)
        headers = self.auth_strategy.prepare_headers(headers)

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.request(
                    method, url, json=data, headers=headers
                ) as response:
                    if response.status != 200:
                        response_text = await response.text()
                        await self._handle_api_error(response.status, response_text)

                    # Stream JSON Lines response
                    async for chunk in self.streaming_adapter.stream_response(
                        response.content
                    ):
                        yield chunk

        except aiohttp.ClientConnectionError as e:
            raise ProviderConnectionError(
                f"Cannot connect to Ollama service at {self.base_url}. "
                "Ensure Ollama is running with: ollama serve",
                provider="ollama"
            ) from e
        except asyncio.TimeoutError as e:
            raise ProviderTimeoutError(
                f"Ollama streaming request timed out after "
                f"{self.timeout.total} seconds",
                provider="ollama"
            ) from e
        except Exception as e:
            if isinstance(e, ProviderError):
                raise
            raise ProviderError(
                f"Ollama streaming failed: {e}", provider="ollama"
            ) from e

    async def _handle_api_error(self, status_code: int, response_text: str) -> None:
        """Handle Ollama API errors.

        Args:
            status_code: HTTP status code
            response_text: Response body text

        Raises:
            Appropriate ProviderError subclass
        """
        try:
            error_data = json.loads(response_text)
            error_message = error_data.get("error", "Unknown error")
        except json.JSONDecodeError:
            error_message = response_text or f"HTTP {status_code} error"

        if status_code == 404:
            if "model" in error_message.lower():
                raise ProviderError(
                    f"Model not found: {error_message}. "
                    "Pull the model with: ollama pull <model_name>",
                    provider="ollama"
                )
            else:
                raise ProviderError(
                    f"Ollama endpoint not found: {error_message}",
                    provider="ollama"
                )
        elif status_code == 400:
            raise ProviderError(
                f"Invalid request to Ollama: {error_message}",
                provider="ollama"
            )
        elif status_code == 500:
            raise ProviderError(
                f"Ollama server error: {error_message}. "
                "Check Ollama service logs for details.",
                provider="ollama"
            )
        else:
            raise ProviderError(
                f"Ollama API error ({status_code}): {error_message}",
                provider="ollama"
            )

    def get_token_limit(self) -> int:
        """Get token limit for current model."""
        # Conservative limits for local models
        model = self.config.model or self.default_model

        # Check if it's a smaller variant model
        if ":1b" in model or ":3b" in model:
            return 2048
        elif ":7b" in model or ":8b" in model:
            return 4096
        elif ":13b" in model or ":14b" in model:
            return 6144
        elif ":34b" in model or ":70b" in model:
            return 8192
        else:
            # Default conservative limit
            return 4096

    async def validate_connection(self) -> bool:
        """Test if Ollama service is accessible and has models.

        Returns:
            True if connection is valid, False otherwise
        """
        try:
            # Check service health
            if not await self._check_service_health():
                logger.warning("Ollama service health check failed")
                self._health_status = "unhealthy"
                return False

            # Check available models
            models = await self.list_models()
            if not models:
                logger.warning("No models available in Ollama")
                self._health_status = "unhealthy"
                return False

            # Try a minimal generation if default model is available
            if self.default_model in models:
                test_request = UnifiedRequest(
                    prompt="Respond with 'OK' if you receive this.",
                    model=self.default_model,
                    max_tokens=5,
                    temperature=0.0
                )

                test_response = await self.generate_unified(test_request)
                success = bool(test_response.content)
                self._health_status = "healthy" if success else "unhealthy"
                return success
            else:
                # Service is up and has models, but not our default
                logger.info(f"Default model {self.default_model} not available")
                self._health_status = "healthy"
                return True

        except Exception as e:
            logger.warning(f"Ollama connection validation failed: {e}")
            self._health_status = "unhealthy"
            return False
