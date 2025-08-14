"""Request/response transform pipeline for provider abstraction layer.

This module provides the core transformation pipeline that converts between our
unified format (UnifiedRequest/UnifiedResponse) and provider-specific formats
(OpenAI, Anthropic, Ollama). This is the core of the abstraction layer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from .unified_models import FinishReason, TokenUsage, UnifiedRequest, UnifiedResponse


class RequestTransformer(ABC):
    """Abstract base class for transforming unified requests to provider formats."""

    @abstractmethod
    def transform(self, unified_request: UnifiedRequest) -> dict[str, Any]:
        """Transform UnifiedRequest to provider-specific API format."""
        pass

    @abstractmethod
    def get_default_model(self) -> str:
        """Get the default model for this provider."""
        pass

    @abstractmethod
    def validate_model(self, model: str) -> bool:
        """Validate if model is supported by this provider."""
        pass


class ResponseParser(ABC):
    """Abstract base class for parsing provider responses to unified format."""

    @abstractmethod
    def parse(self, raw_response: dict[str, Any], request: UnifiedRequest,
              latency_ms: int) -> UnifiedResponse:
        """Parse provider response to UnifiedResponse."""
        pass

    @abstractmethod
    def parse_error(self, error_response: dict[str, Any]) -> str:
        """Parse provider error response to readable message."""
        pass


@dataclass
class TransformPipelineConfig:
    """Configuration for transform pipeline."""
    provider_name: str
    default_model: str
    supported_models: list[str]
    max_tokens_limit: int | None = None
    temperature_range: tuple[float, float] = (0.0, 2.0)


class TransformPipeline:
    """Pipeline for transforming requests/responses between formats."""

    def __init__(self,
                 transformer: RequestTransformer,
                 parser: ResponseParser,
                 config: TransformPipelineConfig):
        self.transformer = transformer
        self.parser = parser
        self.config = config

    def process_request(self, request: UnifiedRequest) -> dict[str, Any]:
        """Process unified request to provider format with validation."""
        # Validate and set default model if needed
        if not request.model:
            request.model = self.config.default_model
        # Allow dynamic models: if transformer rejects, but provider supports
        # dynamic acceptance (i.e., model present in capabilities.supported_models),
        # then pass through. Otherwise, raise.
        if not self.transformer.validate_model(request.model):
            if request.model not in self.config.supported_models:
                msg = (f"Model '{request.model}' not supported by "
                       f"{self.config.provider_name}")
                raise ValueError(msg)

        # Validate parameters
        self._validate_request_parameters(request)

        # Transform request
        try:
            return self.transformer.transform(request)
        except Exception as e:
            msg = f"Failed to transform request for {self.config.provider_name}: {e}"
            raise ValueError(msg) from e

    def process_response(self, raw_response: dict[str, Any],
                        request: UnifiedRequest, latency_ms: int) -> UnifiedResponse:
        """Process provider response to unified format."""
        try:
            response = self.parser.parse(raw_response, request, latency_ms)
            response.provider = self.config.provider_name
            return response
        except Exception as e:
            msg = f"Failed to parse response from {self.config.provider_name}: {e}"
            raise ValueError(msg) from e

    def process_error(self, error_response: dict[str, Any]) -> str:
        """Process provider error response to readable message."""
        try:
            return self.parser.parse_error(error_response)
        except Exception:
            return f"Error from {self.config.provider_name}: {error_response}"

    def _validate_request_parameters(self, request: UnifiedRequest) -> None:
        """Validate request parameters against provider limits."""
        # Validate temperature
        min_temp, max_temp = self.config.temperature_range
        if not (min_temp <= request.temperature <= max_temp):
            msg = (f"Temperature {request.temperature} outside range "
                   f"[{min_temp}, {max_temp}]")
            raise ValueError(msg)

        # Validate max_tokens
        if (self.config.max_tokens_limit and
            request.max_tokens and
            request.max_tokens > self.config.max_tokens_limit):
            msg = (f"max_tokens {request.max_tokens} exceeds limit "
                   f"{self.config.max_tokens_limit}")
            raise ValueError(msg)


# Provider-specific base transformers to be extended

class OpenAIRequestTransformer(RequestTransformer):
    """Base transformer for OpenAI-compatible APIs."""

    def __init__(self) -> None:
        """Initialize OpenAI request transformer with supported models."""
        self.supported_models = [
            "gpt-4o", "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo",
            "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini"
        ]

    def transform(self, unified_request: UnifiedRequest) -> dict[str, Any]:
        """Transform unified request to OpenAI API format."""
        openai_request: dict[str, Any] = {
            "model": unified_request.model,
            "messages": self._build_messages(unified_request),
            "stream": unified_request.stream
        }

        # Only include temperature for legacy models. Some newer models (e.g., GPT-5
        # series) only support the default temperature and reject overrides.
        model_id = (unified_request.model or "").lower()
        if not model_id.startswith("gpt-5") and unified_request.temperature is not None:
            openai_request["temperature"] = unified_request.temperature

        if unified_request.max_tokens:
            # Chat Completions API uses 'max_tokens'.
            # If/when we add the Responses API path, we will map to
            # 'max_output_tokens' there.
            openai_request["max_tokens"] = unified_request.max_tokens
        if unified_request.stop_sequences:
            openai_request["stop"] = unified_request.stop_sequences

        return openai_request

    def _build_messages(self, request: UnifiedRequest) -> list[dict[str, str]]:
        """Build OpenAI messages array from unified request."""
        messages = []
        if request.system_message:
            messages.append({"role": "system", "content": request.system_message})
        messages.append({"role": "user", "content": request.prompt})
        return messages

    def get_default_model(self) -> str:
        """Get default OpenAI model."""
        return "gpt-4o"

    def validate_model(self, model: str) -> bool:
        """Validate if model is supported by OpenAI."""
        return model in self.supported_models


class OpenAIResponseParser(ResponseParser):
    """Base parser for OpenAI-compatible API responses."""

    def parse(self, raw_response: dict[str, Any], request: UnifiedRequest,
              latency_ms: int) -> UnifiedResponse:
        """Parse OpenAI response to unified format."""
        choice = raw_response["choices"][0]
        usage = raw_response.get("usage", {})

        return UnifiedResponse(
            content=choice["message"]["content"],
            model=raw_response["model"],
            usage=TokenUsage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0)
            ),
            latency_ms=latency_ms,
            finish_reason=FinishReason.normalize(
                choice.get("finish_reason", "stop"), "openai"
            ),
            provider="openai",
            raw_response=raw_response
        )

    def parse_error(self, error_response: dict[str, Any]) -> str:
        """Parse OpenAI error response to readable message."""
        error = error_response.get("error", {})
        message = error.get("message", "Unknown error")
        error_type = error.get("type", "unknown")
        return f"OpenAI {error_type}: {message}"


class AnthropicRequestTransformer(RequestTransformer):
    """Base transformer for Anthropic Claude API."""

    def __init__(self) -> None:
        """Initialize Anthropic request transformer with supported models."""
        self.supported_models = [
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
            "claude-haiku-3-5-20241022"
        ]

    def transform(self, unified_request: UnifiedRequest) -> dict[str, Any]:
        """Transform unified request to Anthropic API format."""
        anthropic_request: dict[str, Any] = {
            "model": unified_request.model,
            "messages": [{"role": "user", "content": unified_request.prompt}],
            "max_tokens": unified_request.max_tokens or 4000,
            "temperature": unified_request.temperature,
            "stream": unified_request.stream
        }

        if unified_request.system_message:
            anthropic_request["system"] = unified_request.system_message
        if unified_request.stop_sequences:
            anthropic_request["stop_sequences"] = unified_request.stop_sequences

        return anthropic_request

    def get_default_model(self) -> str:
        """Get default Anthropic model."""
        return "claude-sonnet-4-20250514"

    def validate_model(self, model: str) -> bool:
        """Validate if model is supported by Anthropic."""
        return model in self.supported_models


class AnthropicResponseParser(ResponseParser):
    """Base parser for Anthropic Claude API responses."""

    def parse(self, raw_response: dict[str, Any], request: UnifiedRequest,
              latency_ms: int) -> UnifiedResponse:
        """Parse Anthropic response to unified format."""
        content = raw_response["content"][0]["text"]
        usage = raw_response.get("usage", {})

        return UnifiedResponse(
            content=content,
            model=raw_response["model"],
            usage=TokenUsage(
                prompt_tokens=usage.get("input_tokens", 0),
                completion_tokens=usage.get("output_tokens", 0),
                total_tokens=(
                    usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                )
            ),
            latency_ms=latency_ms,
            finish_reason=FinishReason.normalize(
                raw_response.get("stop_reason", "end_turn"), "anthropic"
            ),
            provider="anthropic",
            raw_response=raw_response
        )

    def parse_error(self, error_response: dict[str, Any]) -> str:
        """Parse Anthropic error response to readable message."""
        error = error_response.get("error", {})
        message = error.get("message", "Unknown error")
        error_type = error.get("type", "unknown")
        return f"Anthropic {error_type}: {message}"


class OllamaRequestTransformer(RequestTransformer):
    """Base transformer for Ollama API."""

    def __init__(self) -> None:
        """Initialize Ollama request transformer with supported models."""
        self.supported_models = [
            "llama3.3", "llama3.2", "mistral", "codellama",
            "deepseek-r1", "phi-4", "qwen2.5"
        ]

    def transform(self, unified_request: UnifiedRequest) -> dict[str, Any]:
        """Transform unified request to Ollama API format."""
        options: dict[str, Any] = {
            "temperature": unified_request.temperature,
            "num_ctx": 4096,  # Context window
        }

        ollama_request: dict[str, Any] = {
            "model": unified_request.model,
            "prompt": unified_request.prompt,
            "stream": unified_request.stream,
            "options": options
        }

        if unified_request.system_message:
            ollama_request["system"] = unified_request.system_message
        if unified_request.max_tokens:
            options["num_predict"] = unified_request.max_tokens
        if unified_request.stop_sequences:
            options["stop"] = unified_request.stop_sequences

        return ollama_request

    def get_default_model(self) -> str:
        """Get default Ollama model."""
        return "llama3.3"

    def validate_model(self, model: str) -> bool:
        """Validate if model is supported by Ollama."""
        # Handle versioned model names like "llama3.2:1b"
        base_model = model.split(":")[0]
        return base_model in self.supported_models


class OllamaResponseParser(ResponseParser):
    """Base parser for Ollama API responses."""

    def parse(self, raw_response: dict[str, Any], request: UnifiedRequest,
              latency_ms: int) -> UnifiedResponse:
        """Parse Ollama response to unified format."""
        # Use latency from response if available, otherwise use provided
        total_duration_ns = raw_response.get("total_duration", 0)
        response_latency = (
            int(total_duration_ns / 1_000_000) if total_duration_ns else latency_ms
        )

        return UnifiedResponse(
            content=raw_response["response"],
            model=raw_response["model"],
            usage=TokenUsage(
                prompt_tokens=raw_response.get("prompt_eval_count", 0),
                completion_tokens=raw_response.get("eval_count", 0),
                total_tokens=(
                    raw_response.get("prompt_eval_count", 0) +
                    raw_response.get("eval_count", 0)
                )
            ),
            latency_ms=response_latency,
            finish_reason=FinishReason.normalize(
                raw_response.get("done_reason", "stop"), "ollama"
            ),
            provider="ollama",
            raw_response=raw_response
        )

    def parse_error(self, error_response: dict[str, Any]) -> str:
        """Parse error from Ollama response."""
        error = error_response.get("error", {})
        message = error.get("message", "Unknown error")
        return f"Ollama error: {message}"
