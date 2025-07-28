"""Tests for token counting unification system."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from dataclasses import asdict

from swebench_runner.providers.token_counters import (
    TokenCounter,
    TokenCounterType,
    TokenCountRequest,
    TokenCountResult,
    TiktokenCounter,
    AnthropicAPICounter,
    MetadataTokenCounter,
    UnifiedTokenCounter,
    create_unified_counter,
)


class TestTokenCountRequest:
    """Test TokenCountRequest dataclass"""
    
    def test_basic_request(self):
        """Test basic request creation"""
        request = TokenCountRequest(
            text="Hello world",
            model="gpt-4"
        )
        assert request.text == "Hello world"
        assert request.model == "gpt-4"
        assert request.include_system is True
        assert request.system_message is None
    
    def test_request_with_system_message(self):
        """Test request with system message"""
        request = TokenCountRequest(
            text="Hello world",
            model="gpt-4",
            system_message="You are a helpful assistant",
            include_system=True
        )
        assert request.system_message == "You are a helpful assistant"
        assert request.include_system is True


class TestTokenCountResult:
    """Test TokenCountResult dataclass"""
    
    def test_basic_result(self):
        """Test basic result creation"""
        result = TokenCountResult(
            token_count=10,
            method=TokenCounterType.TIKTOKEN,
            model="gpt-4"
        )
        assert result.token_count == 10
        assert result.method == TokenCounterType.TIKTOKEN
        assert result.model == "gpt-4"
        assert result.estimated is False
        assert result.details is None
    
    def test_result_with_details(self):
        """Test result with details"""
        details = {"encoding": "cl100k_base"}
        result = TokenCountResult(
            token_count=15,
            method=TokenCounterType.TIKTOKEN,
            model="gpt-4",
            estimated=False,
            details=details
        )
        assert result.details == details


class TestTiktokenCounter:
    """Test TiktokenCounter implementation"""
    
    def test_counter_type(self):
        """Test counter type property"""
        counter = TiktokenCounter()
        assert counter.counter_type == TokenCounterType.TIKTOKEN
    
    def test_supports_model(self):
        """Test model support checking"""
        counter = TiktokenCounter()
        assert counter.supports_model("gpt-4") is True
        assert counter.supports_model("gpt-4o") is True
        assert counter.supports_model("claude-3") is False
        assert counter.supports_model("unknown-model") is False
    
    @pytest.mark.asyncio
    async def test_count_tokens_without_tiktoken(self):
        """Test token counting when tiktoken is not available"""
        counter = TiktokenCounter()
        request = TokenCountRequest(text="Hello world", model="gpt-4")
        
        with patch("builtins.__import__", side_effect=ImportError):
            with pytest.raises(ImportError, match="tiktoken library required"):
                await counter.count_tokens(request)
    
    @pytest.mark.asyncio
    async def test_count_tokens_basic(self):
        """Test basic token counting with tiktoken"""
        counter = TiktokenCounter()
        request = TokenCountRequest(text="Hello world", model="gpt-4")
        
        # Mock tiktoken
        mock_encoding = Mock()
        mock_encoding.encode.return_value = [1, 2, 3, 4]  # 4 tokens
        
        mock_tiktoken = Mock()
        mock_tiktoken.get_encoding.return_value = mock_encoding
        
        with patch.dict("sys.modules", {"tiktoken": mock_tiktoken}):
            result = await counter.count_tokens(request)
        
        assert result.token_count == 7  # 4 + 3 for message structure
        assert result.method == TokenCounterType.TIKTOKEN
        assert result.model == "gpt-4"
        assert result.estimated is False
        assert result.details["encoding"] == "cl100k_base"
    
    @pytest.mark.asyncio
    async def test_count_tokens_with_system_message(self):
        """Test token counting with system message"""
        counter = TiktokenCounter()
        request = TokenCountRequest(
            text="Hello world",
            model="gpt-4",
            system_message="You are helpful",
            include_system=True
        )
        
        # Mock tiktoken
        mock_encoding = Mock()
        mock_encoding.encode.side_effect = [
            [1, 2, 3, 4],  # User message: 4 tokens
            [5, 6, 7]      # System message: 3 tokens
        ]
        
        mock_tiktoken = Mock()
        mock_tiktoken.get_encoding.return_value = mock_encoding
        
        with patch.dict("sys.modules", {"tiktoken": mock_tiktoken}):
            result = await counter.count_tokens(request)
        
        # 4 (user) + 3 (system) + 3 (user structure) + 3 (system structure) = 13
        assert result.token_count == 13
        assert result.details["include_system"] is True


class TestAnthropicAPICounter:
    """Test AnthropicAPICounter implementation"""
    
    def test_counter_type(self):
        """Test counter type property"""
        counter = AnthropicAPICounter()
        assert counter.counter_type == TokenCounterType.API
    
    def test_supports_model(self):
        """Test model support checking"""
        counter = AnthropicAPICounter()
        assert counter.supports_model("claude-sonnet-4-20250514") is True
        assert counter.supports_model("claude-opus-4-20250514") is True
        assert counter.supports_model("gpt-4") is False
        assert counter.supports_model("unknown-model") is False
    
    @pytest.mark.asyncio
    async def test_count_tokens_no_client(self):
        """Test token counting without API client"""
        counter = AnthropicAPICounter()
        request = TokenCountRequest(text="Hello", model="claude-sonnet-4-20250514")
        
        with pytest.raises(ValueError, match="Anthropic API client required"):
            await counter.count_tokens(request)
    
    @pytest.mark.asyncio
    async def test_count_tokens_api_success(self):
        """Test successful API token counting"""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.json.return_value = {"input_tokens": 42}
        mock_client.post.return_value = mock_response
        
        counter = AnthropicAPICounter(api_client=mock_client)
        request = TokenCountRequest(text="Hello", model="claude-sonnet-4-20250514")
        
        result = await counter.count_tokens(request)
        
        assert result.token_count == 42
        assert result.method == TokenCounterType.API
        assert result.estimated is False
        assert result.details["api_response"]["input_tokens"] == 42
    
    @pytest.mark.asyncio
    async def test_count_tokens_api_failure_fallback(self):
        """Test fallback to estimation when API fails"""
        mock_client = AsyncMock()
        mock_client.post.side_effect = Exception("API error")
        
        counter = AnthropicAPICounter(api_client=mock_client)
        request = TokenCountRequest(text="Hello world", model="claude-sonnet-4-20250514")
        
        result = await counter.count_tokens(request)
        
        assert result.method == TokenCounterType.ESTIMATION
        assert result.estimated is True
        assert result.token_count == max(1, len("Hello world") // 4)
        assert result.details["fallback_reason"] == "API unavailable"


class TestMetadataTokenCounter:
    """Test MetadataTokenCounter implementation"""
    
    def test_counter_type(self):
        """Test counter type property"""
        counter = MetadataTokenCounter()
        assert counter.counter_type == TokenCounterType.METADATA
    
    def test_supports_model(self):
        """Test model support checking"""
        counter = MetadataTokenCounter()
        assert counter.supports_model("llama3.3") is True
        assert counter.supports_model("mistral") is True
        assert counter.supports_model("gpt-4") is False
        assert counter.supports_model("claude-3") is False
    
    @pytest.mark.asyncio
    async def test_count_tokens_estimation(self):
        """Test pre-generation token estimation"""
        counter = MetadataTokenCounter()
        request = TokenCountRequest(text="Hello world", model="llama3.3")
        
        result = await counter.count_tokens(request)
        
        assert result.method == TokenCounterType.ESTIMATION
        assert result.estimated is True
        assert result.token_count == max(1, len("Hello world") // 4)
        assert result.details["note"] == "Pre-generation estimation"
    
    def test_count_from_response(self):
        """Test token counting from response metadata"""
        counter = MetadataTokenCounter()
        response_data = {
            "prompt_eval_count": 20,
            "eval_count": 30,
            "total_duration": 1000
        }
        
        result = counter.count_from_response(response_data, "llama3.3")
        
        assert result.token_count == 50  # 20 + 30
        assert result.method == TokenCounterType.METADATA
        assert result.estimated is False
        assert result.details["prompt_tokens"] == 20
        assert result.details["completion_tokens"] == 30
        assert result.details["total_duration"] == 1000


class TestUnifiedTokenCounter:
    """Test UnifiedTokenCounter implementation"""
    
    def test_initialization(self):
        """Test unified counter initialization"""
        counter = UnifiedTokenCounter()
        
        # Should have default counters
        assert "tiktoken" in counter._counters
        assert "metadata" in counter._counters
        assert len(counter._counters) == 2
    
    def test_add_counter(self):
        """Test adding custom counter"""
        counter = UnifiedTokenCounter()
        custom_counter = Mock(spec=TokenCounter)
        
        counter.add_counter("custom", custom_counter)
        
        assert "custom" in counter._counters
        assert counter._counters["custom"] == custom_counter
    
    def test_get_counter_for_model(self):
        """Test getting counter for specific model"""
        counter = UnifiedTokenCounter()
        
        # Should find tiktoken counter for GPT models
        tiktoken_counter = counter.get_counter_for_model("gpt-4")
        assert isinstance(tiktoken_counter, TiktokenCounter)
        
        # Should find metadata counter for Ollama models
        metadata_counter = counter.get_counter_for_model("llama3.3")
        assert isinstance(metadata_counter, MetadataTokenCounter)
        
        # Should return None for unsupported models
        assert counter.get_counter_for_model("unknown-model") is None
    
    @pytest.mark.asyncio
    async def test_count_tokens_success(self):
        """Test successful token counting through unified interface"""
        counter = UnifiedTokenCounter()
        
        # Mock tiktoken counter to succeed
        mock_tiktoken = Mock(spec=TiktokenCounter)
        mock_tiktoken.supports_model.return_value = True
        mock_tiktoken.count_tokens.return_value = TokenCountResult(
            token_count=15,
            method=TokenCounterType.TIKTOKEN,
            model="gpt-4"
        )
        counter._counters["tiktoken"] = mock_tiktoken
        
        result = await counter.count_tokens("Hello", "gpt-4")
        
        assert result.token_count == 15
        assert result.method == TokenCounterType.TIKTOKEN
    
    @pytest.mark.asyncio
    async def test_count_tokens_fallback(self):
        """Test fallback to estimation when no counter works"""
        counter = UnifiedTokenCounter()
        
        # Mock all counters to not support the model
        for counter_instance in counter._counters.values():
            counter_instance.supports_model = Mock(return_value=False)
        
        result = await counter.count_tokens("Hello world", "unknown-model")
        
        assert result.method == TokenCounterType.ESTIMATION
        assert result.estimated is True
        assert result.token_count == max(1, len("Hello world") // 4)
        assert result.details["fallback_reason"] == "No counter available"
    
    @pytest.mark.asyncio
    async def test_count_tokens_with_system_message(self):
        """Test token counting with system message"""
        counter = UnifiedTokenCounter()
        
        # Mock tiktoken counter
        mock_tiktoken = Mock(spec=TiktokenCounter)
        mock_tiktoken.supports_model.return_value = True
        mock_tiktoken.count_tokens.return_value = TokenCountResult(
            token_count=25,
            method=TokenCounterType.TIKTOKEN,
            model="gpt-4"
        )
        counter._counters["tiktoken"] = mock_tiktoken
        
        result = await counter.count_tokens(
            "Hello", 
            "gpt-4", 
            system_message="You are helpful"
        )
        
        # Verify request was created correctly
        call_args = mock_tiktoken.count_tokens.call_args[0][0]
        assert call_args.text == "Hello"
        assert call_args.system_message == "You are helpful"
        assert call_args.include_system is True


class TestCreateUnifiedCounter:
    """Test factory function for creating unified counter"""
    
    def test_create_without_anthropic_client(self):
        """Test creating counter without Anthropic client"""
        counter = create_unified_counter()
        
        # Should have default counters only
        assert "tiktoken" in counter._counters
        assert "metadata" in counter._counters
        assert "anthropic_api" not in counter._counters
    
    def test_create_with_anthropic_client(self):
        """Test creating counter with Anthropic client"""
        mock_client = Mock()
        counter = create_unified_counter(anthropic_client=mock_client)
        
        # Should have all counters including Anthropic
        assert "tiktoken" in counter._counters
        assert "metadata" in counter._counters
        assert "anthropic_api" in counter._counters
        assert isinstance(counter._counters["anthropic_api"], AnthropicAPICounter)


class TestTokenCounterIntegration:
    """Integration tests for token counting system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_token_counting(self):
        """Test complete token counting workflow"""
        # Create unified counter
        counter = create_unified_counter()
        
        # Test with OpenAI model (should use tiktoken if available)
        with patch.dict("sys.modules", {"tiktoken": Mock()}):
            mock_encoding = Mock()
            mock_encoding.encode.return_value = [1, 2, 3]
            
            mock_tiktoken = Mock()
            mock_tiktoken.get_encoding.return_value = mock_encoding
            
            with patch.dict("sys.modules", {"tiktoken": mock_tiktoken}):
                result = await counter.count_tokens("Test", "gpt-4")
                
                assert result.method == TokenCounterType.TIKTOKEN
                assert result.token_count > 0
    
    @pytest.mark.asyncio
    async def test_model_specific_routing(self):
        """Test that different models route to appropriate counters"""
        counter = create_unified_counter()
        
        # Mock counters
        mock_tiktoken = Mock(spec=TiktokenCounter)
        mock_tiktoken.supports_model.side_effect = lambda m: m.startswith("gpt")
        mock_tiktoken.count_tokens.return_value = TokenCountResult(
            token_count=10, method=TokenCounterType.TIKTOKEN, model="gpt-4"
        )
        
        mock_metadata = Mock(spec=MetadataTokenCounter)
        mock_metadata.supports_model.side_effect = lambda m: m.startswith("llama")
        mock_metadata.count_tokens.return_value = TokenCountResult(
            token_count=15, method=TokenCounterType.METADATA, model="llama3.3"
        )
        
        counter._counters["tiktoken"] = mock_tiktoken
        counter._counters["metadata"] = mock_metadata
        
        # Test OpenAI model routing
        result1 = await counter.count_tokens("Test", "gpt-4")
        assert result1.method == TokenCounterType.TIKTOKEN
        
        # Test Ollama model routing
        result2 = await counter.count_tokens("Test", "llama3.3")
        assert result2.method == TokenCounterType.METADATA