"""Unified data models for provider abstraction layer.

This module provides consistent data structures that work across all providers
(OpenAI, Anthropic, Ollama) to enable a unified interface regardless of the
underlying provider's specific API format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


@dataclass
class TokenUsage:
    """Unified token usage information across all providers.

    This provides a consistent way to track token consumption regardless
    of whether the provider is OpenAI (uses prompt_tokens/completion_tokens),
    Anthropic (uses input_tokens/output_tokens), or Ollama (uses metadata counts).
    """
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def __post_init__(self) -> None:
        """Ensure total_tokens is consistent with component counts.

        If total_tokens is 0 but we have component counts, calculate it.
        This handles providers that don't provide total_tokens directly.
        """
        if self.total_tokens == 0:
            self.total_tokens = self.prompt_tokens + self.completion_tokens


@dataclass
class UnifiedRequest:
    """Provider-agnostic request format that works across all providers.

    This unified format is transformed into provider-specific formats by
    each provider's request transformer. This allows the same request
    to work with OpenAI's messages format, Anthropic's separate system field,
    or Ollama's prompt-based format.
    """
    prompt: str
    system_message: str | None = None
    max_tokens: int | None = None
    temperature: float = 0.7
    stream: bool = False
    model: str | None = None
    stop_sequences: list[str] | None = None

    def __post_init__(self) -> None:
        """Validate request parameters for consistency across providers."""
        if not self.prompt:
            raise ValueError("Prompt cannot be empty")
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("Temperature must be between 0 and 2")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")


@dataclass
class UnifiedResponse:
    """Provider-agnostic response format that works across all providers.

    This unified format is created by each provider's response parser from
    their native response format. This ensures consistent access to response
    data regardless of whether it came from OpenAI's choices array,
    Anthropic's content array, or Ollama's direct response field.
    """
    content: str
    model: str
    usage: TokenUsage
    latency_ms: int
    finish_reason: str
    provider: str
    cost: float | None = None
    raw_response: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate response data for consistency.

        Note: Empty content is allowed in some cases (e.g., when hitting token limits).
        """
        if not self.content and self.finish_reason != "length":
            # Empty content is only valid if we hit token limit
            pass  # Allow empty content for some edge cases
        if self.latency_ms < 0:
            self.latency_ms = 0


class FinishReason(Enum):
    """Standardized finish reasons across providers.

    Different providers use different terminology for why generation stopped:
    - OpenAI: "stop", "length", "tool_calls", "content_filter"
    - Anthropic: "end_turn", "max_tokens", "tool_use"
    - Ollama: "stop", "length"

    This enum provides normalized values and mapping utilities.
    """
    STOP = "stop"                    # Natural completion
    LENGTH = "length"                # Token limit reached
    TOOL_CALLS = "tool_calls"        # Function calling
    CONTENT_FILTER = "content_filter" # Content filtered
    ERROR = "error"                  # Generation error

    @classmethod
    def normalize(cls, provider_reason: str, provider: str) -> str:
        """Normalize provider-specific finish reasons to our standard.

        Args:
            provider_reason: The reason string from the provider's response
            provider: The provider name ("openai", "anthropic", "ollama")

        Returns:
            Normalized finish reason string

        Example:
            >>> FinishReason.normalize("end_turn", "anthropic")
            "stop"
            >>> FinishReason.normalize("max_tokens", "anthropic")
            "length"
        """
        # OpenAI mapping
        if provider == "openai":
            mapping = {
                "stop": cls.STOP.value,
                "length": cls.LENGTH.value,
                "tool_calls": cls.TOOL_CALLS.value,
                "content_filter": cls.CONTENT_FILTER.value,
            }
            return mapping.get(provider_reason, cls.STOP.value)

        # Anthropic mapping
        elif provider == "anthropic":
            mapping = {
                "end_turn": cls.STOP.value,
                "max_tokens": cls.LENGTH.value,
                "tool_use": cls.TOOL_CALLS.value,
            }
            return mapping.get(provider_reason, cls.STOP.value)

        # Ollama mapping
        elif provider == "ollama":
            mapping = {
                "stop": cls.STOP.value,
                "length": cls.LENGTH.value,
            }
            return mapping.get(provider_reason, cls.STOP.value)

        # Default fallback for unknown providers
        return cls.STOP.value

