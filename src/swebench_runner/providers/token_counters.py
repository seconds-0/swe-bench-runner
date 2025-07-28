"""Token counting unification system for different AI providers.

This module provides a unified interface for token counting across different
AI providers (OpenAI, Anthropic, Ollama) with fallback mechanisms.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any


class TokenCounterType(Enum):
    """Types of token counting methods"""
    TIKTOKEN = "tiktoken"      # Local tiktoken library
    API = "api"                # Remote API endpoint
    METADATA = "metadata"      # Extract from response metadata
    ESTIMATION = "estimation"  # Simple estimation fallback


@dataclass
class TokenCountRequest:
    """Request for token counting"""
    text: str
    model: str
    include_system: bool = True
    system_message: str | None = None


@dataclass
class TokenCountResult:
    """Result of token counting"""
    token_count: int
    method: TokenCounterType
    model: str
    estimated: bool = False
    details: dict[str, Any] | None = None


class TokenCounter(ABC):
    """Abstract base class for token counting strategies"""

    @abstractmethod
    async def count_tokens(self, request: TokenCountRequest) -> TokenCountResult:
        """Count tokens for the given text and model"""
        pass

    @abstractmethod
    def supports_model(self, model: str) -> bool:
        """Check if this counter supports the given model"""
        pass

    @property
    @abstractmethod
    def counter_type(self) -> TokenCounterType:
        """Get the type of this token counter"""
        pass


class TiktokenCounter(TokenCounter):
    """Token counter using OpenAI's tiktoken library"""

    def __init__(self):
        self._encodings = {}
        # Model to encoding mapping
        self._model_encodings = {
            "gpt-4o": "o200k_base",
            "gpt-4o-mini": "o200k_base",
            "gpt-4": "cl100k_base",
            "gpt-4-turbo": "cl100k_base",
            "gpt-3.5-turbo": "cl100k_base",
            "gpt-4.1": "o200k_base",
            "gpt-4.1-mini": "o200k_base",
        }

    async def count_tokens(self, request: TokenCountRequest) -> TokenCountResult:
        """Count tokens using tiktoken"""
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "tiktoken library required for OpenAI token counting. "
                "Install with: pip install tiktoken"
            )

        encoding_name = self._model_encodings.get(request.model, "cl100k_base")

        if encoding_name not in self._encodings:
            self._encodings[encoding_name] = tiktoken.get_encoding(encoding_name)

        encoding = self._encodings[encoding_name]

        # Count tokens for main text
        token_count = len(encoding.encode(request.text))

        # Add system message tokens if provided
        if request.include_system and request.system_message:
            system_tokens = len(encoding.encode(request.system_message))
            token_count += system_tokens
            # Add tokens for message structure (role, content fields)
            token_count += 3  # Approximate overhead per message

        # Add tokens for message structure
        token_count += 3  # Approximate overhead for user message

        return TokenCountResult(
            token_count=token_count,
            method=TokenCounterType.TIKTOKEN,
            model=request.model,
            estimated=False,
            details={
                "encoding": encoding_name,
                "include_system": request.include_system
            }
        )

    def supports_model(self, model: str) -> bool:
        """Check if model is supported by tiktoken"""
        return model in self._model_encodings

    @property
    def counter_type(self) -> TokenCounterType:
        return TokenCounterType.TIKTOKEN


class AnthropicAPICounter(TokenCounter):
    """Token counter using Anthropic's count_tokens API"""

    def __init__(self, api_client=None):
        self.api_client = api_client
        self._supported_models = {
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
            "claude-haiku-3-5-20241022"
        }

    async def count_tokens(self, request: TokenCountRequest) -> TokenCountResult:
        """Count tokens using Anthropic API"""
        if not self.api_client:
            raise ValueError("Anthropic API client required for token counting")

        # Build messages for API call
        messages = [{"role": "user", "content": request.text}]

        api_request = {
            "model": request.model,
            "messages": messages
        }

        # Add system message if provided
        if request.include_system and request.system_message:
            api_request["system"] = request.system_message

        try:
            # Make API call to count tokens
            response = await self.api_client.post(
                "/v1/messages/count_tokens",
                json=api_request
            )
            response_data = response.json()

            token_count = response_data["input_tokens"]

            return TokenCountResult(
                token_count=token_count,
                method=TokenCounterType.API,
                model=request.model,
                estimated=False,
                details={
                    "api_response": response_data,
                    "include_system": request.include_system
                }
            )

        except Exception:
            # Fallback to estimation
            return await self._estimate_tokens(request)

    async def _estimate_tokens(self, request: TokenCountRequest) -> TokenCountResult:
        """Fallback token estimation for Anthropic"""
        # Rough estimation: ~4 characters per token
        text_length = len(request.text)
        if request.include_system and request.system_message:
            text_length += len(request.system_message)

        estimated_tokens = max(1, text_length // 4)

        return TokenCountResult(
            token_count=estimated_tokens,
            method=TokenCounterType.ESTIMATION,
            model=request.model,
            estimated=True,
            details={
                "text_length": text_length,
                "estimation_ratio": 4,
                "fallback_reason": "API unavailable"
            }
        )

    def supports_model(self, model: str) -> bool:
        """Check if model is supported by Anthropic API"""
        return model in self._supported_models

    @property
    def counter_type(self) -> TokenCounterType:
        return TokenCounterType.API


class MetadataTokenCounter(TokenCounter):
    """Token counter that extracts counts from response metadata (Ollama)"""

    def __init__(self):
        self._last_response = None
        self._supported_models = {
            "llama3.3", "llama3.2", "mistral", "codellama",
            "deepseek-r1", "phi-4", "qwen2.5"
        }

    async def count_tokens(self, request: TokenCountRequest) -> TokenCountResult:
        """Count tokens using estimation (metadata available after generation)"""
        # For pre-generation counting, we need to estimate
        return await self._estimate_tokens(request)

    def count_from_response(self, response_data: dict[str, Any], model: str) -> TokenCountResult:
        """Extract token count from Ollama response metadata"""
        prompt_tokens = response_data.get("prompt_eval_count", 0)
        completion_tokens = response_data.get("eval_count", 0)
        total_tokens = prompt_tokens + completion_tokens

        return TokenCountResult(
            token_count=total_tokens,
            method=TokenCounterType.METADATA,
            model=model,
            estimated=False,
            details={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_duration": response_data.get("total_duration"),
                "source": "response_metadata"
            }
        )

    async def _estimate_tokens(self, request: TokenCountRequest) -> TokenCountResult:
        """Estimate tokens for Ollama models"""
        # Rough estimation: ~4 characters per token
        text_length = len(request.text)
        if request.include_system and request.system_message:
            text_length += len(request.system_message)

        estimated_tokens = max(1, text_length // 4)

        return TokenCountResult(
            token_count=estimated_tokens,
            method=TokenCounterType.ESTIMATION,
            model=request.model,
            estimated=True,
            details={
                "text_length": text_length,
                "estimation_ratio": 4,
                "note": "Pre-generation estimation"
            }
        )

    def supports_model(self, model: str) -> bool:
        """Check if model is supported"""
        return model in self._supported_models

    @property
    def counter_type(self) -> TokenCounterType:
        return TokenCounterType.METADATA


class UnifiedTokenCounter:
    """Unified interface for token counting across all providers"""

    def __init__(self):
        self._counters = {}
        self._setup_default_counters()

    def _setup_default_counters(self):
        """Setup default token counters"""
        self._counters["tiktoken"] = TiktokenCounter()
        self._counters["metadata"] = MetadataTokenCounter()
        # Note: AnthropicAPICounter requires client, set up separately

    def add_counter(self, name: str, counter: TokenCounter):
        """Add a token counter"""
        self._counters[name] = counter

    async def count_tokens(self, text: str, model: str,
                          system_message: str | None = None) -> TokenCountResult:
        """Count tokens using the best available counter for the model"""
        request = TokenCountRequest(
            text=text,
            model=model,
            system_message=system_message,
            include_system=system_message is not None
        )

        # Try counters in order of preference
        for counter in self._counters.values():
            if counter.supports_model(model):
                try:
                    return await counter.count_tokens(request)
                except Exception:
                    continue  # Try next counter

        # Fallback to simple estimation
        return await self._estimate_tokens(request)

    async def _estimate_tokens(self, request: TokenCountRequest) -> TokenCountResult:
        """Fallback token estimation"""
        text_length = len(request.text)
        if request.include_system and request.system_message:
            text_length += len(request.system_message)

        estimated_tokens = max(1, text_length // 4)

        return TokenCountResult(
            token_count=estimated_tokens,
            method=TokenCounterType.ESTIMATION,
            model=request.model,
            estimated=True,
            details={
                "text_length": text_length,
                "estimation_ratio": 4,
                "fallback_reason": "No counter available"
            }
        )

    def get_counter_for_model(self, model: str) -> TokenCounter | None:
        """Get the best counter for a specific model"""
        for counter in self._counters.values():
            if counter.supports_model(model):
                return counter
        return None


# Factory function for easy setup
def create_unified_counter(anthropic_client=None) -> UnifiedTokenCounter:
    """Create a unified token counter with all available counters"""
    counter = UnifiedTokenCounter()

    if anthropic_client:
        counter.add_counter("anthropic_api", AnthropicAPICounter(anthropic_client))

    return counter
