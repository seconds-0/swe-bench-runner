"""Test OpenAI-specific error handling and classification."""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from typing import Any, Dict, Union

import aiohttp

from swebench_runner.providers.openai_errors import (
    OpenAIAuthenticationError,
    OpenAIContentFilterError,
    OpenAIContextLengthError,
    OpenAIErrorHandler,
    OpenAIModelNotFoundError,
    OpenAIRateLimitError,
    OpenAIServerError,
)
from swebench_runner.providers.exceptions import (
    ProviderConnectionError,
    ProviderError,
    ProviderTimeoutError,
)


@pytest.fixture
def error_handler():
    """Create an OpenAI error handler for testing."""
    return OpenAIErrorHandler(max_retries=3, base_delay=1.0)


class MockResponse:
    """Mock aiohttp response for testing."""
    
    def __init__(self, status: int, data: Union[Dict[str, Any], None] = None, headers: Union[Dict[str, str], None] = None):
        self.status = status
        self._data = data or {}
        self.headers = headers or {}
        self.request_info = Mock()
        self.history = []
    
    async def json(self):
        return self._data
    
    async def text(self):
        return json.dumps(self._data)


class TestOpenAIErrorClassification:
    """Test OpenAI error classification."""
    
    @pytest.mark.asyncio
    async def test_authentication_error_invalid_key(self, error_handler):
        """Test authentication error with invalid API key."""
        response = MockResponse(
            status=401,
            data={"error": {"message": "Invalid API key", "type": "invalid_api_key"}}
        )
        
        error = await error_handler.classify_response_error(response)
        
        assert isinstance(error, OpenAIAuthenticationError)
        assert "Invalid OpenAI API key" in str(error)
        assert "OPENAI_API_KEY environment variable" in str(error)
        assert "https://platform.openai.com/api-keys" in str(error)
    
    @pytest.mark.asyncio
    async def test_authentication_error_unauthorized(self, error_handler):
        """Test authentication error with unauthorized access."""
        response = MockResponse(
            status=401,
            data={"error": {"message": "Unauthorized", "code": "unauthorized"}}
        )
        
        error = await error_handler.classify_response_error(response)
        
        assert isinstance(error, OpenAIAuthenticationError)
        assert error.error_code == "unauthorized"
        assert "Invalid OpenAI API key" in str(error) or "Authentication failed" in str(error)
    
    @pytest.mark.asyncio
    async def test_rate_limit_error_requests(self, error_handler):
        """Test rate limit error for requests."""
        response = MockResponse(
            status=429,
            data={"error": {"message": "Too many requests", "type": "requests"}},
            headers={"retry-after": "60", "x-ratelimit-reset-requests": "2025-01-01T00:00:00Z"}
        )
        
        error = await error_handler.classify_response_error(response)
        
        assert isinstance(error, OpenAIRateLimitError)
        assert error.retry_after == 60
        assert error.limit_type == "requests"
        assert error.daily_limit_reset == "2025-01-01T00:00:00Z"
        assert "Request rate limit exceeded" in str(error)
        assert "reduce your request frequency" in str(error)
    
    @pytest.mark.asyncio
    async def test_rate_limit_error_tokens(self, error_handler):
        """Test rate limit error for tokens."""
        response = MockResponse(
            status=429,
            data={"error": {"message": "Token limit exceeded", "type": "tokens"}},
            headers={"retry-after": "30"}
        )
        
        error = await error_handler.classify_response_error(response)
        
        assert isinstance(error, OpenAIRateLimitError)
        assert error.retry_after == 30
        assert error.limit_type == "tokens"
        assert "Token rate limit exceeded" in str(error)
        assert "reduce your request size" in str(error)
    
    @pytest.mark.asyncio
    async def test_model_not_found_error(self, error_handler):
        """Test model not found error."""
        response = MockResponse(
            status=404,
            data={"error": {"message": "Model 'gpt-5' not found", "code": "model_not_found"}}
        )
        
        error = await error_handler.classify_response_error(response)
        
        assert isinstance(error, OpenAIModelNotFoundError)
        assert error.model == "gpt-5"
        # The error handler generates user-friendly messages
        assert "gpt-5" in str(error)
        assert "gpt-4o" in str(error)
        assert "https://platform.openai.com/docs/models" in str(error)
    
    @pytest.mark.asyncio
    async def test_context_length_error(self, error_handler):
        """Test context length exceeded error."""
        response = MockResponse(
            status=400,
            data={
                "error": {
                    "message": "This model's maximum context length is 8192 tokens. However, you requested 10000 tokens",
                    "type": "context_length_exceeded"
                }
            }
        )
        
        error = await error_handler.classify_response_error(response)
        
        assert isinstance(error, OpenAIContextLengthError)
        assert error.token_count == 10000
        assert error.limit == 8192
        assert error.suggested_max_tokens is not None
        assert error.suggested_max_tokens > 0
        assert "Input too long" in str(error)
        assert "smaller chunks" in str(error)
    
    @pytest.mark.asyncio
    async def test_content_filter_error(self, error_handler):
        """Test content policy violation error."""
        response = MockResponse(
            status=400,
            data={
                "error": {
                    "message": "Content violates usage policies",
                    "type": "content_policy_violation"
                }
            }
        )
        
        error = await error_handler.classify_response_error(response)
        
        assert isinstance(error, OpenAIContentFilterError)
        assert error.filter_type == "content_policy"
        assert "Content policy violation" in str(error)
        assert "https://platform.openai.com/docs/usage-policies" in str(error)
    
    @pytest.mark.asyncio
    async def test_server_error_500(self, error_handler):
        """Test server error 500."""
        response = MockResponse(
            status=500,
            data={"error": {"message": "Internal server error", "type": "server_error"}}
        )
        
        error = await error_handler.classify_response_error(response)
        
        assert isinstance(error, OpenAIServerError)
        assert error.status_code == 500
        assert error.server_error_type == "internal_error"
        assert "OpenAI server error (500)" in str(error)
        assert "temporary issue" in str(error)
    
    @pytest.mark.asyncio
    async def test_server_error_503(self, error_handler):
        """Test service unavailable error."""
        response = MockResponse(
            status=503,
            data={"error": {"message": "Service temporarily unavailable"}}
        )
        
        error = await error_handler.classify_response_error(response)
        
        assert isinstance(error, OpenAIServerError)
        assert error.status_code == 503
        assert error.server_error_type == "service_unavailable"
    
    @pytest.mark.asyncio
    async def test_permission_error_billing(self, error_handler):
        """Test permission error related to billing."""
        response = MockResponse(
            status=403,
            data={"error": {"message": "Insufficient quota. Please check your billing details."}}
        )
        
        error = await error_handler.classify_response_error(response)
        
        assert "Billing or quota issue" in str(error)
        assert "https://platform.openai.com/usage" in str(error)
    
    def test_timeout_error_classification(self, error_handler):
        """Test timeout error classification."""
        timeout_error = asyncio.TimeoutError()
        
        error = error_handler.classify_error(timeout_error)
        
        assert isinstance(error, ProviderTimeoutError)
        assert "network issues or API server overload" in str(error)
    
    def test_connection_error_classification(self, error_handler):
        """Test connection error classification."""
        conn_error = aiohttp.ClientConnectionError("Connection failed")
        
        error = error_handler.classify_error(conn_error)
        
        assert isinstance(error, ProviderConnectionError)
        assert "Failed to connect to OpenAI API" in str(error)


class TestOpenAIRetryLogic:
    """Test OpenAI retry logic and backoff."""
    
    def test_should_retry_authentication_error(self, error_handler):
        """Test that authentication errors are not retried."""
        error = OpenAIAuthenticationError("Invalid API key")
        
        assert not error_handler.should_retry(error)
    
    def test_should_retry_content_filter_error(self, error_handler):
        """Test that content filter errors are not retried."""
        error = OpenAIContentFilterError("Content violation")
        
        assert not error_handler.should_retry(error)
    
    def test_should_retry_context_length_error(self, error_handler):
        """Test that context length errors are not retried."""
        error = OpenAIContextLengthError("Input too long")
        
        assert not error_handler.should_retry(error)
    
    def test_should_retry_model_not_found_error(self, error_handler):
        """Test that model not found errors are not retried."""
        error = OpenAIModelNotFoundError("Model not found")
        
        assert not error_handler.should_retry(error)
    
    def test_should_retry_rate_limit_error(self, error_handler):
        """Test that rate limit errors are retried."""
        error = OpenAIRateLimitError("Rate limited", retry_after=60)
        
        assert error_handler.should_retry(error)
    
    def test_should_retry_server_error(self, error_handler):
        """Test that server errors are retried."""
        error = OpenAIServerError("Server error", status_code=500)
        
        assert error_handler.should_retry(error)
    
    def test_should_retry_timeout_error(self, error_handler):
        """Test that timeout errors are retried."""
        error = ProviderTimeoutError("Timeout")
        
        assert error_handler.should_retry(error)
    
    def test_retry_delay_rate_limit_with_retry_after(self, error_handler):
        """Test retry delay for rate limits with retry-after."""
        error = OpenAIRateLimitError("Rate limited", retry_after=60)
        
        delay = error_handler.get_retry_delay(error, attempt=0)
        
        # Should be approximately retry_after with small jitter
        assert 60 <= delay <= 66  # 60 + 10% jitter
    
    def test_retry_delay_rate_limit_without_retry_after(self, error_handler):
        """Test retry delay for rate limits without retry-after."""
        error = OpenAIRateLimitError("Rate limited")  # No retry_after
        
        delay = error_handler.get_retry_delay(error, attempt=1)
        
        # Should use exponential backoff: base_delay * (3 ** attempt)
        expected = min(60.0, 1.0 * (3 ** 1))  # 3 seconds
        assert delay == expected
    
    def test_retry_delay_server_error(self, error_handler):
        """Test retry delay for server errors."""
        error = OpenAIServerError("Server error")
        
        delay = error_handler.get_retry_delay(error, attempt=1)
        
        # Should use exponential backoff: base_delay * (2 ** attempt)
        expected = min(30.0, 1.0 * (2 ** 1))  # 2 seconds
        assert delay == expected
    
    def test_retry_delay_standard_backoff(self, error_handler):
        """Test standard exponential backoff."""
        error = ProviderTimeoutError("Timeout")
        
        delay = error_handler.get_retry_delay(error, attempt=2)
        
        # Should use exponential backoff: base_delay * (2 ** attempt)
        expected = min(16.0, 1.0 * (2 ** 2))  # 4 seconds
        assert delay == expected


class TestOpenAIUserMessages:
    """Test user-friendly error messages."""
    
    def test_user_message_authentication_error(self, error_handler):
        """Test user message for authentication error."""
        error = OpenAIAuthenticationError("Invalid API key", "invalid_key")
        
        message = error_handler.get_user_message(error)
        
        assert "Invalid API key" in message
        assert message == str(error)  # OpenAI errors have user-friendly messages
    
    def test_user_message_rate_limit_error(self, error_handler):
        """Test user message for rate limit error."""
        error = OpenAIRateLimitError(
            "Token rate limit exceeded: Too many tokens. Wait 30s before retrying, or reduce your request size.",
            retry_after=30, 
            limit_type="tokens"
        )
        
        message = error_handler.get_user_message(error)
        
        assert "Token rate limit exceeded" in message
        assert "30s" in message
    
    def test_user_message_context_length_error(self, error_handler):
        """Test user message for context length error."""
        error = OpenAIContextLengthError(
            "Input too long: This model's maximum context length is 8192 tokens. Try reducing your prompt length or set max_tokens to 1000 or less.",
            token_count=10000, 
            limit=8192, 
            suggested_max_tokens=1000
        )
        
        message = error_handler.get_user_message(error)
        
        assert "Input too long" in message
        assert "max_tokens to 1000" in message
    
    def test_user_message_generic_timeout(self, error_handler):
        """Test user message for generic timeout."""
        error = ProviderTimeoutError("Request timed out after 30s")
        
        message = error_handler.get_user_message(error)
        
        assert "Request timed out after 30s" in message
        assert "network issues or high API load" in message
    
    def test_user_message_generic_connection(self, error_handler):
        """Test user message for generic connection error."""
        error = ProviderConnectionError("Connection failed")
        
        message = error_handler.get_user_message(error)
        
        assert "Connection failed" in message
        assert "https://status.openai.com" in message


class TestOpenAISpecificFeatures:
    """Test OpenAI-specific error handling features."""
    
    def test_extract_retry_after_standard_header(self, error_handler):
        """Test extracting retry-after from standard header."""
        headers = {"retry-after": "120"}
        
        retry_after = error_handler._extract_retry_after(headers)
        
        assert retry_after == 120
    
    def test_extract_retry_after_openai_headers(self, error_handler):
        """Test extracting retry-after from OpenAI-specific headers."""
        import time
        future_time = time.time() + 180
        headers = {"x-ratelimit-reset-requests": str(future_time)}
        
        retry_after = error_handler._extract_retry_after(headers)
        
        # Should be approximately 180 seconds (within 5 seconds tolerance)
        assert 175 <= retry_after <= 185
    
    def test_extract_retry_after_fallback(self, error_handler):
        """Test retry-after fallback when no headers present."""
        headers = {}
        
        retry_after = error_handler._extract_retry_after(headers)
        
        assert retry_after == 60  # Default fallback
    
    def test_extract_retry_after_cap(self, error_handler):
        """Test retry-after is capped at maximum."""
        import time
        future_time = time.time() + 600  # 10 minutes in future
        headers = {"x-ratelimit-reset-tokens": str(future_time)}
        
        retry_after = error_handler._extract_retry_after(headers)
        
        assert retry_after == 300  # Capped at 5 minutes


class TestOpenAIErrorIntegration:
    """Test integration of OpenAI error handling with provider."""
    
    @pytest.mark.asyncio
    async def test_safe_json_valid_response(self, error_handler):
        """Test safe JSON parsing with valid response."""
        response = MockResponse(200, {"message": "success"})
        
        data = await error_handler._safe_json(response)
        
        assert data == {"message": "success"}
    
    @pytest.mark.asyncio
    async def test_safe_json_invalid_response(self, error_handler):
        """Test safe JSON parsing with invalid response."""
        response = MockResponse(500)
        response.json = AsyncMock(side_effect=Exception("Invalid JSON"))
        response.text = AsyncMock(return_value="Server Error")
        
        data = await error_handler._safe_json(response)
        
        assert data == {"error": {"message": "Server Error"}}
    
    @pytest.mark.asyncio
    async def test_safe_json_total_failure(self, error_handler):
        """Test safe JSON parsing when everything fails."""
        response = MockResponse(500)
        response.json = AsyncMock(side_effect=Exception("Invalid JSON"))
        response.text = AsyncMock(side_effect=Exception("Cannot read text"))
        
        data = await error_handler._safe_json(response)
        
        assert data == {"error": {"message": "HTTP 500"}}