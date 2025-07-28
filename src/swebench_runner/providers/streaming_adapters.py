"""Streaming response adapters for provider abstraction layer.

This module provides unified streaming interfaces that can handle different
streaming formats used by various providers:
- OpenAI & Anthropic: Server-Sent Events (SSE) with "data:" prefix
- Ollama: JSON Lines format (one JSON object per line)
- Generic: Plain text streaming

The adapters convert provider-specific streaming formats into a unified
StreamChunk interface for consistent handling across all providers.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class StreamingFormat(Enum):
    """Types of streaming response formats supported by different providers."""
    SSE = "sse"              # Server-Sent Events (OpenAI, Anthropic)
    JSON_LINES = "json_lines" # JSON Lines format (Ollama)
    PLAIN_TEXT = "plain_text" # Simple text streaming (fallback)


@dataclass
class StreamChunk:
    """A chunk of streaming response data in unified format.

    This provides a consistent interface for streaming data regardless
    of the underlying provider's format. All streaming adapters convert
    their provider-specific format into this unified format.

    Attributes:
        content: Complete accumulated content up to this point
        delta: Incremental content in this specific chunk
        done: Whether this is the final chunk in the stream
        metadata: Provider-specific metadata (finish reasons, timing, etc.)
        raw_chunk: Original provider-specific chunk data for debugging
    """
    content: str
    delta: str | None = None
    done: bool = False
    metadata: dict[str, Any] | None = None
    raw_chunk: dict[str, Any] | None = None


class StreamingAdapter(ABC):
    """Abstract base class for streaming response adapters.

    Each provider can have its own streaming format, so we need adapters
    to convert from provider-specific formats to our unified StreamChunk format.
    This enables consistent streaming handling across all providers.
    """

    @abstractmethod
    def stream_response(
        self, response_stream: Any
    ) -> AsyncGenerator[StreamChunk, None]:
        """Convert provider-specific stream to unified StreamChunk format.

        Args:
            response_stream: The raw streaming response from the provider

        Yields:
            StreamChunk objects in unified format
        """
        pass

    @abstractmethod
    def parse_chunk(self, raw_chunk: str) -> StreamChunk | None:
        """Parse a raw chunk into a StreamChunk.

        Args:
            raw_chunk: Raw chunk data as string

        Returns:
            StreamChunk if parseable, None if should be skipped
        """
        pass

    @property
    @abstractmethod
    def format_type(self) -> StreamingFormat:
        """Get the streaming format type this adapter handles."""
        pass


class SSEAdapter(StreamingAdapter):
    """Adapter for Server-Sent Events format (OpenAI, Anthropic).

    Server-Sent Events format uses lines like:
    data: {"choices": [{"delta": {"content": "text"}}]}
    data: [DONE]

    Different providers have different SSE data structures:
    - OpenAI: Uses choices array with delta.content
    - Anthropic: Uses event types with delta.text
    """

    def __init__(self, provider: str = "unknown"):
        """Initialize SSE adapter for specific provider.

        Args:
            provider: Provider name for format-specific parsing
        """
        self.provider = provider.lower()
        self._current_content = ""

    async def stream_response(
        self, response_stream: Any
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream SSE responses and yield unified chunks.

        Args:
            response_stream: AsyncIterator of SSE lines

        Yields:
            StreamChunk objects parsed from SSE data
        """
        try:
            async for line in response_stream:
                # Handle bytes or string input
                if isinstance(line, bytes):
                    line = line.decode('utf-8')

                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith(':'):
                    continue

                chunk = self.parse_chunk(line)
                if chunk:
                    yield chunk

                    # Stop processing if stream is done
                    if chunk.done:
                        break
        except Exception as e:
            logger.error(f"Error in SSE stream processing: {e}")
            # Yield final chunk with error metadata
            yield StreamChunk(
                content=self._current_content,
                done=True,
                metadata={"error": str(e), "end_reason": "error"}
            )

    def parse_chunk(self, raw_chunk: str) -> StreamChunk | None:
        """Parse SSE chunk into StreamChunk.

        Args:
            raw_chunk: Raw SSE line (e.g., "data: {json}")

        Returns:
            StreamChunk if parseable, None if should be skipped
        """
        # SSE format: "data: {json_data}"
        if not raw_chunk.startswith('data: '):
            return None

        data_content = raw_chunk[6:].strip()  # Remove "data: " prefix

        # Handle end of stream
        if data_content == '[DONE]':
            return StreamChunk(
                content=self._current_content,
                done=True,
                metadata={"end_reason": "completed"}
            )

        try:
            data = json.loads(data_content)

            # Provider-specific parsing
            if self.provider == "openai":
                return self._parse_openai_chunk(data)
            elif self.provider == "anthropic":
                return self._parse_anthropic_chunk(data)
            else:
                # Generic SSE parsing
                return self._parse_generic_chunk(data)

        except json.JSONDecodeError as e:
            logger.warning(
                f"Failed to parse SSE JSON data: {data_content[:100]}... Error: {e}"
            )
            return None

    def _parse_openai_chunk(self, data: dict[str, Any]) -> StreamChunk | None:
        """Parse OpenAI-specific SSE chunk.

        OpenAI format:
        {
            "choices": [{"delta": {"content": "text"}, "finish_reason": null}],
            "model": "gpt-4",
            "id": "chatcmpl-123"
        }

        Args:
            data: Parsed JSON data from SSE chunk

        Returns:
            StreamChunk with OpenAI-specific metadata
        """
        choices = data.get("choices", [])
        if not choices:
            return None

        choice = choices[0]
        delta = choice.get("delta", {})
        content = delta.get("content", "")
        finish_reason = choice.get("finish_reason")

        # Update cumulative content
        if content:
            self._current_content += content

        return StreamChunk(
            content=self._current_content,
            delta=content,
            done=finish_reason is not None,
            metadata={
                "finish_reason": finish_reason,
                "choice_index": choice.get("index", 0),
                "model": data.get("model"),
                "id": data.get("id"),
                "provider": "openai"
            },
            raw_chunk=data
        )

    def _parse_anthropic_chunk(self, data: dict[str, Any]) -> StreamChunk | None:
        """Parse Anthropic-specific SSE chunk.

        Anthropic format varies by event type:
        - content_block_delta: {"type": "content_block_delta", "delta": {"text": "..."}}
        - message_stop: {"type": "message_stop"}
        - ping: {"type": "ping"}

        Args:
            data: Parsed JSON data from SSE chunk

        Returns:
            StreamChunk with Anthropic-specific metadata
        """
        event_type = data.get("type")

        if event_type == "content_block_delta":
            delta_data = data.get("delta", {})
            content = delta_data.get("text", "")

            if content:
                self._current_content += content

            return StreamChunk(
                content=self._current_content,
                delta=content,
                done=False,
                metadata={
                    "event_type": event_type,
                    "index": data.get("index", 0),
                    "provider": "anthropic"
                },
                raw_chunk=data
            )

        elif event_type == "message_stop":
            return StreamChunk(
                content=self._current_content,
                done=True,
                metadata={
                    "event_type": event_type,
                    "stop_reason": "end_turn",
                    "provider": "anthropic"
                },
                raw_chunk=data
            )

        elif event_type == "ping":
            # Keep-alive event, don't yield anything
            logger.debug("Received Anthropic ping event")
            return None

        elif event_type == "message_start":
            # Message start event, initialize but don't yield
            logger.debug("Received Anthropic message_start event")
            return None

        elif event_type == "content_block_start":
            # Content block start, initialize but don't yield
            logger.debug("Received Anthropic content_block_start event")
            return None

        else:
            # Unknown event type, log and skip
            logger.debug(f"Unknown Anthropic event type: {event_type}")
            return None

    def _parse_generic_chunk(self, data: dict[str, Any]) -> StreamChunk | None:
        """Parse generic SSE chunk for unknown providers.

        Args:
            data: Parsed JSON data from SSE chunk

        Returns:
            StreamChunk with generic content extraction
        """
        # Look for common content fields
        content = ""
        for field in ["content", "text", "message", "response"]:
            if field in data:
                content = str(data[field])
                break

        if content:
            self._current_content += content

        done = data.get("done", data.get("finished", False))

        return StreamChunk(
            content=self._current_content,
            delta=content,
            done=done,
            metadata={**data, "provider": "generic"},
            raw_chunk=data
        )

    @property
    def format_type(self) -> StreamingFormat:
        """Get the streaming format type."""
        return StreamingFormat.SSE


class JSONLinesAdapter(StreamingAdapter):
    """Adapter for JSON Lines format (Ollama).

    JSON Lines format sends one JSON object per line:
    {"response": "text", "done": false}
    {"response": "", "done": true, "total_duration": 123456}

    This is simpler than SSE but requires line-by-line JSON parsing.
    """

    def __init__(self) -> None:
        """Initialize JSON Lines adapter."""
        self._current_content = ""

    async def stream_response(
        self, response_stream: Any
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream JSON Lines responses and yield unified chunks.

        Args:
            response_stream: AsyncIterator of JSON Lines

        Yields:
            StreamChunk objects parsed from JSON Lines
        """
        try:
            async for line in response_stream:
                # Handle bytes or string input
                if isinstance(line, bytes):
                    line = line.decode('utf-8')

                line = line.strip()
                if not line:
                    continue

                chunk = self.parse_chunk(line)
                if chunk:
                    yield chunk

                    # Stop processing if stream is done
                    if chunk.done:
                        break
        except Exception as e:
            logger.error(f"Error in JSON Lines stream processing: {e}")
            # Yield final chunk with error metadata
            yield StreamChunk(
                content=self._current_content,
                done=True,
                metadata={"error": str(e), "end_reason": "error"}
            )

    def parse_chunk(self, raw_chunk: str) -> StreamChunk | None:
        """Parse JSON Lines chunk into StreamChunk.

        Args:
            raw_chunk: Single line of JSON data

        Returns:
            StreamChunk if parseable, None if should be skipped
        """
        try:
            data = json.loads(raw_chunk)

            # Ollama response format
            response_text = data.get("response", "")
            done = data.get("done", False)

            # Update cumulative content
            if response_text:
                self._current_content += response_text

            metadata = {
                "model": data.get("model"),
                "created_at": data.get("created_at"),
                "done_reason": data.get("done_reason"),
                "provider": "ollama"
            }

            # Add timing information if available
            if done:
                timing_fields = [
                    "total_duration", "load_duration", "prompt_eval_count",
                    "prompt_eval_duration", "eval_count", "eval_duration"
                ]
                for field in timing_fields:
                    if field in data:
                        metadata[field] = data[field]

            return StreamChunk(
                content=self._current_content,
                delta=response_text,
                done=done,
                metadata=metadata,
                raw_chunk=data
            )

        except json.JSONDecodeError as e:
            logger.warning(
                f"Failed to parse JSON Lines data: {raw_chunk[:100]}... Error: {e}"
            )
            return None

    @property
    def format_type(self) -> StreamingFormat:
        """Get the streaming format type."""
        return StreamingFormat.JSON_LINES


class PlainTextAdapter(StreamingAdapter):
    """Adapter for plain text streaming (fallback).

    This is a simple fallback adapter for providers that stream
    plain text without any structured format. Each chunk is treated
    as additional content to accumulate.
    """

    def __init__(self) -> None:
        """Initialize plain text adapter."""
        self._current_content = ""

    async def stream_response(
        self, response_stream: Any
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream plain text responses and yield unified chunks.

        Args:
            response_stream: AsyncIterator of text chunks

        Yields:
            StreamChunk objects from plain text
        """
        try:
            async for chunk in response_stream:
                # Handle bytes or string input
                if isinstance(chunk, bytes):
                    chunk = chunk.decode('utf-8')

                if chunk:
                    self._current_content += chunk
                    yield StreamChunk(
                        content=self._current_content,
                        delta=chunk,
                        done=False,
                        metadata={"provider": "plain_text"}
                    )
        except Exception as e:
            logger.error(f"Error in plain text stream processing: {e}")
            # Yield final chunk with error metadata
            yield StreamChunk(
                content=self._current_content,
                done=True,
                metadata={"error": str(e), "end_reason": "error"}
            )

    def parse_chunk(self, raw_chunk: str) -> StreamChunk | None:
        """Parse plain text chunk.

        Args:
            raw_chunk: Plain text content

        Returns:
            StreamChunk with accumulated content
        """
        if raw_chunk:
            self._current_content += raw_chunk
            return StreamChunk(
                content=self._current_content,
                delta=raw_chunk,
                done=False,
                metadata={"provider": "plain_text"}
            )
        return None

    @property
    def format_type(self) -> StreamingFormat:
        """Get the streaming format type."""
        return StreamingFormat.PLAIN_TEXT


class StreamingManager:
    """Manager for selecting and using appropriate streaming adapters.

    This provides a high-level interface for working with different
    streaming formats. It can automatically select the right adapter
    based on the provider or format type.
    """

    def __init__(self) -> None:
        """Initialize streaming manager with all available adapters."""
        self._adapters = {
            StreamingFormat.SSE: SSEAdapter,
            StreamingFormat.JSON_LINES: JSONLinesAdapter,
            StreamingFormat.PLAIN_TEXT: PlainTextAdapter,
        }

    def get_adapter(
        self, format_type: StreamingFormat, **kwargs: Any
    ) -> StreamingAdapter:
        """Get adapter for the specified format.

        Args:
            format_type: The streaming format type
            **kwargs: Additional arguments for adapter initialization

        Returns:
            Configured streaming adapter

        Raises:
            ValueError: If format type is not supported
        """
        adapter_class = self._adapters.get(format_type)
        if not adapter_class:
            raise ValueError(f"Unsupported streaming format: {format_type}")

        return adapter_class(**kwargs)  # type: ignore

    def get_adapter_for_provider(self, provider: str) -> StreamingAdapter:
        """Get appropriate adapter for the given provider.

        Args:
            provider: Provider name (case-insensitive)

        Returns:
            Streaming adapter configured for the provider
        """
        provider = provider.lower()

        if provider in ["openai"]:
            return SSEAdapter(provider="openai")
        elif provider in ["anthropic"]:
            return SSEAdapter(provider="anthropic")
        elif provider in ["ollama"]:
            return JSONLinesAdapter()
        else:
            # Default to plain text for unknown providers
            logger.warning(f"Unknown provider '{provider}', using plain text adapter")
            return PlainTextAdapter()

    async def stream_with_adapter(
        self,
        response_stream: Any,
        adapter: StreamingAdapter
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream response using the specified adapter.

        Args:
            response_stream: The raw streaming response
            adapter: Configured streaming adapter

        Yields:
            StreamChunk objects in unified format
        """
        async for chunk in adapter.stream_response(response_stream):
            yield chunk


# Helper functions for easy provider-specific usage

async def stream_openai_response(
    response_stream: Any,
) -> AsyncGenerator[StreamChunk, None]:
    """Stream OpenAI SSE response using the appropriate adapter.

    Args:
        response_stream: OpenAI streaming response

    Yields:
        StreamChunk objects with OpenAI-specific parsing
    """
    adapter = SSEAdapter(provider="openai")
    async for chunk in adapter.stream_response(response_stream):
        yield chunk


async def stream_anthropic_response(
    response_stream: Any,
) -> AsyncGenerator[StreamChunk, None]:
    """Stream Anthropic SSE response using the appropriate adapter.

    Args:
        response_stream: Anthropic streaming response

    Yields:
        StreamChunk objects with Anthropic-specific parsing
    """
    adapter = SSEAdapter(provider="anthropic")
    async for chunk in adapter.stream_response(response_stream):
        yield chunk


async def stream_ollama_response(
    response_stream: Any,
) -> AsyncGenerator[StreamChunk, None]:
    """Stream Ollama JSON Lines response using the appropriate adapter.

    Args:
        response_stream: Ollama streaming response

    Yields:
        StreamChunk objects with Ollama-specific parsing
    """
    adapter = JSONLinesAdapter()
    async for chunk in adapter.stream_response(response_stream):
        yield chunk


def create_streaming_manager() -> StreamingManager:
    """Create a streaming manager with all adapters registered.

    Returns:
        Configured StreamingManager instance
    """
    return StreamingManager()

