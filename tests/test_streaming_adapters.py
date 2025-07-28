"""Tests for streaming response adapters.

This module tests the unified streaming interface and provider-specific
adapters for OpenAI (SSE), Anthropic (SSE), Ollama (JSON Lines), and
plain text streaming formats.
"""

import json
from collections.abc import AsyncIterator

import pytest

from swebench_runner.providers.streaming_adapters import (
    JSONLinesAdapter,
    PlainTextAdapter,
    SSEAdapter,
    StreamChunk,
    StreamingFormat,
    StreamingManager,
    create_streaming_manager,
    stream_anthropic_response,
    stream_ollama_response,
    stream_openai_response,
)

# Test data fixtures

OPENAI_SSE_CHUNKS = [
    'data: {"choices":[{"delta":{"content":"Hello"},"index":0}],"model":"gpt-4","id":"chatcmpl-123"}',
    'data: {"choices":[{"delta":{"content":" world"},"index":0}],"model":"gpt-4","id":"chatcmpl-123"}',
    'data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}],"model":"gpt-4","id":"chatcmpl-123"}',
    'data: [DONE]'
]

ANTHROPIC_SSE_CHUNKS = [
    'data: {"type":"message_start","message":{"id":"msg_123","type":"message"}}',
    'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}',
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}',
    'data: {"type":"content_block_stop","index":0}',
    'data: {"type":"message_stop"}',
    'data: {"type":"ping"}'  # Should be ignored
]

OLLAMA_JSON_LINES = [
    '{"model":"llama2","response":"Hello","done":false,"created_at":"2024-01-01T00:00:00Z"}',
    '{"model":"llama2","response":" world","done":false,"created_at":"2024-01-01T00:00:01Z"}',
    '{"model":"llama2","response":"","done":true,"total_duration":123456,"eval_count":2}'
]

PLAIN_TEXT_CHUNKS = [
    "Hello",
    " world",
    "!"
]


async def async_list_to_stream(items: list[str]) -> AsyncIterator[str]:
    """Convert a list to an async iterator for testing."""
    for item in items:
        yield item


class TestStreamChunk:
    """Test the StreamChunk dataclass."""

    def test_basic_creation(self):
        """Test basic StreamChunk creation."""
        chunk = StreamChunk(content="Hello", delta="Hello", done=False)
        assert chunk.content == "Hello"
        assert chunk.delta == "Hello"
        assert chunk.done is False
        assert chunk.metadata is None
        assert chunk.raw_chunk is None

    def test_with_metadata(self):
        """Test StreamChunk with metadata."""
        metadata = {"provider": "openai", "model": "gpt-4"}
        raw_chunk = {"choices": [{"delta": {"content": "test"}}]}

        chunk = StreamChunk(
            content="test",
            delta="test",
            done=True,
            metadata=metadata,
            raw_chunk=raw_chunk
        )

        assert chunk.metadata == metadata
        assert chunk.raw_chunk == raw_chunk


class TestSSEAdapter:
    """Test the SSE (Server-Sent Events) adapter."""

    @pytest.mark.asyncio
    async def test_openai_streaming(self):
        """Test OpenAI SSE format streaming."""
        adapter = SSEAdapter(provider="openai")
        stream = async_list_to_stream(OPENAI_SSE_CHUNKS)

        chunks = []
        async for chunk in adapter.stream_response(stream):
            chunks.append(chunk)

        # Should have 3 chunks: 2 content chunks + 1 finish chunk
        assert len(chunks) == 3

        # First chunk
        assert chunks[0].content == "Hello"
        assert chunks[0].delta == "Hello"
        assert chunks[0].done is False
        assert chunks[0].metadata["provider"] == "openai"
        assert chunks[0].metadata["model"] == "gpt-4"

        # Second chunk
        assert chunks[1].content == "Hello world"
        assert chunks[1].delta == " world"
        assert chunks[1].done is False

        # Final chunk
        assert chunks[2].content == "Hello world"
        assert chunks[2].delta == ""
        assert chunks[2].done is True
        assert chunks[2].metadata["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_anthropic_streaming(self):
        """Test Anthropic SSE format streaming."""
        adapter = SSEAdapter(provider="anthropic")
        stream = async_list_to_stream(ANTHROPIC_SSE_CHUNKS)

        chunks = []
        async for chunk in adapter.stream_response(stream):
            chunks.append(chunk)

        # Should have 3 chunks: 2 content deltas + 1 stop
        # (start events and ping are filtered out)
        assert len(chunks) == 3

        # First content chunk
        assert chunks[0].content == "Hello"
        assert chunks[0].delta == "Hello"
        assert chunks[0].done is False
        assert chunks[0].metadata["event_type"] == "content_block_delta"
        assert chunks[0].metadata["provider"] == "anthropic"

        # Second content chunk
        assert chunks[1].content == "Hello world"
        assert chunks[1].delta == " world"
        assert chunks[1].done is False

        # Stop chunk
        assert chunks[2].content == "Hello world"
        assert chunks[2].done is True
        assert chunks[2].metadata["event_type"] == "message_stop"

    @pytest.mark.asyncio
    async def test_generic_sse(self):
        """Test generic SSE format for unknown providers."""
        adapter = SSEAdapter(provider="unknown")
        sse_chunks = [
            'data: {"content":"Hello","done":false}',
            'data: {"content":" world","done":false}',
            'data: {"content":"","done":true}'
        ]
        stream = async_list_to_stream(sse_chunks)

        chunks = []
        async for chunk in adapter.stream_response(stream):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0].content == "Hello"
        assert chunks[1].content == "Hello world"
        assert chunks[2].done is True

    def test_parse_invalid_json(self):
        """Test handling of invalid JSON in SSE data."""
        adapter = SSEAdapter(provider="openai")
        result = adapter.parse_chunk('data: {invalid json}')
        assert result is None

    def test_parse_non_data_line(self):
        """Test handling of non-data SSE lines."""
        adapter = SSEAdapter(provider="openai")

        # Comment line
        result = adapter.parse_chunk(': this is a comment')
        assert result is None

        # Empty line
        result = adapter.parse_chunk('')
        assert result is None

        # Non-data event
        result = adapter.parse_chunk('event: ping')
        assert result is None

    def test_done_marker(self):
        """Test handling of [DONE] marker."""
        adapter = SSEAdapter(provider="openai")
        result = adapter.parse_chunk('data: [DONE]')

        assert result is not None
        assert result.done is True
        assert result.metadata["end_reason"] == "completed"

    def test_format_type(self):
        """Test format type property."""
        adapter = SSEAdapter()
        assert adapter.format_type == StreamingFormat.SSE


class TestJSONLinesAdapter:
    """Test the JSON Lines adapter."""

    @pytest.mark.asyncio
    async def test_ollama_streaming(self):
        """Test Ollama JSON Lines format streaming."""
        adapter = JSONLinesAdapter()
        stream = async_list_to_stream(OLLAMA_JSON_LINES)

        chunks = []
        async for chunk in adapter.stream_response(stream):
            chunks.append(chunk)

        # Should have 3 chunks: 2 content + 1 final
        assert len(chunks) == 3

        # First chunk
        assert chunks[0].content == "Hello"
        assert chunks[0].delta == "Hello"
        assert chunks[0].done is False
        assert chunks[0].metadata["model"] == "llama2"
        assert chunks[0].metadata["provider"] == "ollama"

        # Second chunk
        assert chunks[1].content == "Hello world"
        assert chunks[1].delta == " world"
        assert chunks[1].done is False

        # Final chunk
        assert chunks[2].content == "Hello world"
        assert chunks[2].delta == ""
        assert chunks[2].done is True
        assert "total_duration" in chunks[2].metadata
        assert "eval_count" in chunks[2].metadata

    def test_parse_invalid_json(self):
        """Test handling of invalid JSON in JSON Lines."""
        adapter = JSONLinesAdapter()
        result = adapter.parse_chunk('{invalid json}')
        assert result is None

    def test_parse_empty_line(self):
        """Test handling of empty lines."""
        adapter = JSONLinesAdapter()
        result = adapter.parse_chunk('')
        assert result is None

    def test_format_type(self):
        """Test format type property."""
        adapter = JSONLinesAdapter()
        assert adapter.format_type == StreamingFormat.JSON_LINES


class TestPlainTextAdapter:
    """Test the plain text adapter."""

    @pytest.mark.asyncio
    async def test_plain_text_streaming(self):
        """Test plain text format streaming."""
        adapter = PlainTextAdapter()
        stream = async_list_to_stream(PLAIN_TEXT_CHUNKS)

        chunks = []
        async for chunk in adapter.stream_response(stream):
            chunks.append(chunk)

        # Should have 3 chunks
        assert len(chunks) == 3

        # First chunk
        assert chunks[0].content == "Hello"
        assert chunks[0].delta == "Hello"
        assert chunks[0].done is False
        assert chunks[0].metadata["provider"] == "plain_text"

        # Second chunk
        assert chunks[1].content == "Hello world"
        assert chunks[1].delta == " world"
        assert chunks[1].done is False

        # Third chunk
        assert chunks[2].content == "Hello world!"
        assert chunks[2].delta == "!"
        assert chunks[2].done is False

    @pytest.mark.asyncio
    async def test_bytes_input(self):
        """Test handling of bytes input."""
        adapter = PlainTextAdapter()
        byte_chunks = [b"Hello", b" world"]

        chunks = []
        async for chunk in adapter.stream_response(async_list_to_stream(byte_chunks)):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0].content == "Hello"
        assert chunks[1].content == "Hello world"

    def test_parse_chunk(self):
        """Test parse_chunk method."""
        adapter = PlainTextAdapter()

        result = adapter.parse_chunk("Hello")
        assert result is not None
        assert result.content == "Hello"
        assert result.delta == "Hello"

        # Empty chunk
        result = adapter.parse_chunk("")
        assert result is None

    def test_format_type(self):
        """Test format type property."""
        adapter = PlainTextAdapter()
        assert adapter.format_type == StreamingFormat.PLAIN_TEXT


class TestStreamingManager:
    """Test the streaming manager."""

    def test_get_adapter(self):
        """Test getting adapters by format type."""
        manager = StreamingManager()

        # SSE adapter
        adapter = manager.get_adapter(StreamingFormat.SSE, provider="openai")
        assert isinstance(adapter, SSEAdapter)
        assert adapter.provider == "openai"

        # JSON Lines adapter
        adapter = manager.get_adapter(StreamingFormat.JSON_LINES)
        assert isinstance(adapter, JSONLinesAdapter)

        # Plain text adapter
        adapter = manager.get_adapter(StreamingFormat.PLAIN_TEXT)
        assert isinstance(adapter, PlainTextAdapter)

    def test_get_adapter_invalid_format(self):
        """Test getting adapter for invalid format."""
        manager = StreamingManager()

        with pytest.raises(ValueError, match="Unsupported streaming format"):
            manager.get_adapter("invalid_format")

    def test_get_adapter_for_provider(self):
        """Test getting adapters by provider name."""
        manager = StreamingManager()

        # OpenAI
        adapter = manager.get_adapter_for_provider("openai")
        assert isinstance(adapter, SSEAdapter)
        assert adapter.provider == "openai"

        # Anthropic
        adapter = manager.get_adapter_for_provider("anthropic")
        assert isinstance(adapter, SSEAdapter)
        assert adapter.provider == "anthropic"

        # Ollama
        adapter = manager.get_adapter_for_provider("ollama")
        assert isinstance(adapter, JSONLinesAdapter)

        # Unknown provider (should default to plain text)
        adapter = manager.get_adapter_for_provider("unknown")
        assert isinstance(adapter, PlainTextAdapter)

    @pytest.mark.asyncio
    async def test_stream_with_adapter(self):
        """Test streaming with a specific adapter."""
        manager = StreamingManager()
        adapter = SSEAdapter(provider="openai")
        stream = async_list_to_stream(OPENAI_SSE_CHUNKS)

        chunks = []
        async for chunk in manager.stream_with_adapter(stream, adapter):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[-1].done is True


class TestHelperFunctions:
    """Test the helper functions."""

    @pytest.mark.asyncio
    async def test_stream_openai_response(self):
        """Test OpenAI helper function."""
        stream = async_list_to_stream(OPENAI_SSE_CHUNKS)

        chunks = []
        async for chunk in stream_openai_response(stream):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0].metadata["provider"] == "openai"

    @pytest.mark.asyncio
    async def test_stream_anthropic_response(self):
        """Test Anthropic helper function."""
        stream = async_list_to_stream(ANTHROPIC_SSE_CHUNKS)

        chunks = []
        async for chunk in stream_anthropic_response(stream):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0].metadata["provider"] == "anthropic"

    @pytest.mark.asyncio
    async def test_stream_ollama_response(self):
        """Test Ollama helper function."""
        stream = async_list_to_stream(OLLAMA_JSON_LINES)

        chunks = []
        async for chunk in stream_ollama_response(stream):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0].metadata["provider"] == "ollama"

    def test_create_streaming_manager(self):
        """Test create_streaming_manager function."""
        manager = create_streaming_manager()
        assert isinstance(manager, StreamingManager)


class TestErrorHandling:
    """Test error handling in streaming adapters."""

    @pytest.mark.asyncio
    async def test_sse_stream_error(self):
        """Test SSE adapter handling of stream errors."""
        adapter = SSEAdapter(provider="openai")

        # Create a stream that raises an error
        async def error_stream():
            yield 'data: {"choices":[{"delta":{"content":"Hello"}}]}'
            raise ValueError("Stream error")

        chunks = []
        async for chunk in adapter.stream_response(error_stream()):
            chunks.append(chunk)

        # Should have initial chunk + error chunk
        assert len(chunks) == 2
        assert chunks[0].content == "Hello"
        assert chunks[1].done is True
        assert "error" in chunks[1].metadata

    @pytest.mark.asyncio
    async def test_json_lines_stream_error(self):
        """Test JSON Lines adapter handling of stream errors."""
        adapter = JSONLinesAdapter()

        # Create a stream that raises an error
        async def error_stream():
            yield '{"response":"Hello","done":false}'
            raise ConnectionError("Network error")

        chunks = []
        async for chunk in adapter.stream_response(error_stream()):
            chunks.append(chunk)

        # Should have initial chunk + error chunk
        assert len(chunks) == 2
        assert chunks[0].content == "Hello"
        assert chunks[1].done is True
        assert "error" in chunks[1].metadata

    @pytest.mark.asyncio
    async def test_plain_text_stream_error(self):
        """Test plain text adapter handling of stream errors."""
        adapter = PlainTextAdapter()

        # Create a stream that raises an error
        async def error_stream():
            yield "Hello"
            raise RuntimeError("Processing error")

        chunks = []
        async for chunk in adapter.stream_response(error_stream()):
            chunks.append(chunk)

        # Should have initial chunk + error chunk
        assert len(chunks) == 2
        assert chunks[0].content == "Hello"
        assert chunks[1].done is True
        assert "error" in chunks[1].metadata


class TestEdgeCases:
    """Test edge cases and unusual inputs."""

    @pytest.mark.asyncio
    async def test_empty_stream(self):
        """Test handling of empty streams."""
        adapters = [
            SSEAdapter(provider="openai"),
            JSONLinesAdapter(),
            PlainTextAdapter()
        ]

        for adapter in adapters:
            chunks = []
            async for chunk in adapter.stream_response(async_list_to_stream([])):
                chunks.append(chunk)

            # Empty stream should produce no chunks
            assert len(chunks) == 0

    @pytest.mark.asyncio
    async def test_whitespace_only_chunks(self):
        """Test handling of whitespace-only chunks."""
        # SSE with whitespace
        adapter = SSEAdapter(provider="openai")
        whitespace_chunks = ["   ", "\n", "\t", "   \n   "]

        chunks = []
        async for chunk in adapter.stream_response(async_list_to_stream(whitespace_chunks)):
            chunks.append(chunk)

        assert len(chunks) == 0  # Whitespace should be filtered out

    def test_sse_malformed_data_prefix(self):
        """Test SSE lines without proper data: prefix."""
        adapter = SSEAdapter(provider="openai")

        # Missing space after colon
        result = adapter.parse_chunk('data:{"test":"value"}')
        assert result is None

        # Wrong prefix
        result = adapter.parse_chunk('event: message')
        assert result is None

    @pytest.mark.asyncio
    async def test_mixed_string_bytes_input(self):
        """Test handling of mixed string and bytes input."""
        adapter = PlainTextAdapter()
        mixed_chunks = ["Hello", b" world", "!"]

        chunks = []
        async for chunk in adapter.stream_response(async_list_to_stream(mixed_chunks)):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0].content == "Hello"
        assert chunks[1].content == "Hello world"
        assert chunks[2].content == "Hello world!"

    def test_openai_empty_choices(self):
        """Test OpenAI SSE with empty choices array."""
        adapter = SSEAdapter(provider="openai")
        result = adapter.parse_chunk('data: {"choices":[],"model":"gpt-4"}')
        assert result is None

    def test_anthropic_unknown_event_type(self):
        """Test Anthropic SSE with unknown event type."""
        adapter = SSEAdapter(provider="anthropic")
        result = adapter.parse_chunk('data: {"type":"unknown_event","data":"test"}')
        assert result is None


class TestProviderSpecificBehavior:
    """Test provider-specific parsing behavior."""

    def test_openai_tool_calls(self):
        """Test OpenAI SSE with tool calls."""
        adapter = SSEAdapter(provider="openai")
        chunk_data = {
            "choices": [{
                "delta": {"tool_calls": [{"function": {"name": "test_func"}}]},
                "finish_reason": "tool_calls",
                "index": 0
            }],
            "model": "gpt-4",
            "id": "chatcmpl-123"
        }

        result = adapter.parse_chunk(f'data: {json.dumps(chunk_data)}')
        assert result is not None
        assert result.done is True
        assert result.metadata["finish_reason"] == "tool_calls"

    def test_anthropic_content_block_events(self):
        """Test Anthropic content block start/stop events."""
        adapter = SSEAdapter(provider="anthropic")

        # Content block start (should be ignored)
        start_event = {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""}
        }
        result = adapter.parse_chunk(f'data: {json.dumps(start_event)}')
        assert result is None

        # Content block stop (should be ignored)
        stop_event = {
            "type": "content_block_stop",
            "index": 0
        }
        result = adapter.parse_chunk(f'data: {json.dumps(stop_event)}')
        assert result is None

    def test_ollama_timing_metadata(self):
        """Test Ollama timing metadata extraction."""
        adapter = JSONLinesAdapter()
        ollama_final = {
            "model": "llama2",
            "response": "",
            "done": True,
            "total_duration": 123456789,
            "load_duration": 12345,
            "prompt_eval_count": 10,
            "prompt_eval_duration": 54321,
            "eval_count": 5,
            "eval_duration": 98765
        }

        result = adapter.parse_chunk(json.dumps(ollama_final))
        assert result is not None
        assert result.done is True
        assert result.metadata["total_duration"] == 123456789
        assert result.metadata["eval_count"] == 5
