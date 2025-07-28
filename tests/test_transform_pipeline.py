"""Tests for the request/response transform pipeline."""

import pytest
from typing import Dict, Any

from swebench_runner.providers.transform_pipeline import (
    TransformPipeline,
    TransformPipelineConfig,
    OpenAIRequestTransformer,
    OpenAIResponseParser,
    AnthropicRequestTransformer,
    AnthropicResponseParser,
    OllamaRequestTransformer,
    OllamaResponseParser,
)
from swebench_runner.providers.unified_models import (
    UnifiedRequest,
    UnifiedResponse,
    TokenUsage,
    FinishReason,
)


class TestOpenAITransformers:
    """Test OpenAI request transformer and response parser."""
    
    def test_request_transformer_basic(self):
        """Test basic request transformation."""
        transformer = OpenAIRequestTransformer()
        request = UnifiedRequest(
            prompt="Fix this bug",
            model="gpt-4o"
        )
        
        result = transformer.transform(request)
        
        assert result["model"] == "gpt-4o"
        assert result["messages"] == [{"role": "user", "content": "Fix this bug"}]
        assert result["temperature"] == 0.7
        assert result["stream"] is False
    
    def test_request_transformer_with_system_message(self):
        """Test transformation with system message."""
        transformer = OpenAIRequestTransformer()
        request = UnifiedRequest(
            prompt="Fix this bug",
            system_message="You are a Python expert",
            model="gpt-4"
        )
        
        result = transformer.transform(request)
        
        expected_messages = [
            {"role": "system", "content": "You are a Python expert"},
            {"role": "user", "content": "Fix this bug"}
        ]
        assert result["messages"] == expected_messages
    
    def test_request_transformer_with_all_options(self):
        """Test transformation with all optional parameters."""
        transformer = OpenAIRequestTransformer()
        request = UnifiedRequest(
            prompt="Fix this bug",
            system_message="You are a Python expert",
            model="gpt-4",
            max_tokens=1000,
            temperature=0.5,
            stream=True,
            stop_sequences=["END", "STOP"]
        )
        
        result = transformer.transform(request)
        
        assert result["max_tokens"] == 1000
        assert result["temperature"] == 0.5
        assert result["stream"] is True
        assert result["stop"] == ["END", "STOP"]
    
    def test_default_model(self):
        """Test default model retrieval."""
        transformer = OpenAIRequestTransformer()
        assert transformer.get_default_model() == "gpt-4o"
    
    def test_model_validation(self):
        """Test model validation."""
        transformer = OpenAIRequestTransformer()
        
        # Valid models
        assert transformer.validate_model("gpt-4o") is True
        assert transformer.validate_model("gpt-4") is True
        assert transformer.validate_model("gpt-3.5-turbo") is True
        
        # Invalid models
        assert transformer.validate_model("invalid-model") is False
        assert transformer.validate_model("claude-3") is False
    
    def test_response_parser_basic(self):
        """Test basic response parsing."""
        parser = OpenAIResponseParser()
        raw_response = {
            "choices": [
                {
                    "message": {"content": "Fixed the bug!"},
                    "finish_reason": "stop"
                }
            ],
            "model": "gpt-4o",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }
        
        request = UnifiedRequest(prompt="Fix this bug", model="gpt-4o")
        result = parser.parse(raw_response, request, 150)
        
        assert result.content == "Fixed the bug!"
        assert result.model == "gpt-4o"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5
        assert result.usage.total_tokens == 15
        assert result.latency_ms == 150
        assert result.finish_reason == "stop"
        assert result.provider == "openai"
        assert result.raw_response == raw_response
    
    def test_response_parser_without_usage(self):
        """Test response parsing without usage information."""
        parser = OpenAIResponseParser()
        raw_response = {
            "choices": [
                {
                    "message": {"content": "Fixed the bug!"},
                    "finish_reason": "stop"
                }
            ],
            "model": "gpt-4o"
        }
        
        request = UnifiedRequest(prompt="Fix this bug", model="gpt-4o")
        result = parser.parse(raw_response, request, 150)
        
        assert result.usage.prompt_tokens == 0
        assert result.usage.completion_tokens == 0
        assert result.usage.total_tokens == 0
    
    def test_error_parsing(self):
        """Test error response parsing."""
        parser = OpenAIResponseParser()
        error_response = {
            "error": {
                "message": "Invalid API key",
                "type": "authentication_error"
            }
        }
        
        result = parser.parse_error(error_response)
        assert result == "OpenAI authentication_error: Invalid API key"
    
    def test_error_parsing_minimal(self):
        """Test error parsing with minimal information."""
        parser = OpenAIResponseParser()
        error_response = {
            "error": {
                "message": "Something went wrong"
            }
        }
        
        result = parser.parse_error(error_response)
        assert result == "OpenAI unknown: Something went wrong"


class TestAnthropicTransformers:
    """Test Anthropic request transformer and response parser."""
    
    def test_request_transformer_basic(self):
        """Test basic request transformation."""
        transformer = AnthropicRequestTransformer()
        request = UnifiedRequest(
            prompt="Fix this bug",
            model="claude-sonnet-4-20250514"
        )
        
        result = transformer.transform(request)
        
        assert result["model"] == "claude-sonnet-4-20250514"
        assert result["messages"] == [{"role": "user", "content": "Fix this bug"}]
        assert result["max_tokens"] == 4000
        assert result["temperature"] == 0.7
        assert result["stream"] is False
    
    def test_request_transformer_with_system_message(self):
        """Test transformation with system message."""
        transformer = AnthropicRequestTransformer()
        request = UnifiedRequest(
            prompt="Fix this bug",
            system_message="You are a Python expert",
            model="claude-sonnet-4-20250514"
        )
        
        result = transformer.transform(request)
        
        assert result["system"] == "You are a Python expert"
        assert result["messages"] == [{"role": "user", "content": "Fix this bug"}]
    
    def test_request_transformer_with_all_options(self):
        """Test transformation with all optional parameters."""
        transformer = AnthropicRequestTransformer()
        request = UnifiedRequest(
            prompt="Fix this bug",
            max_tokens=2000,
            temperature=0.3,
            stream=True,
            stop_sequences=["END", "STOP"],
            model="claude-sonnet-4-20250514"
        )
        
        result = transformer.transform(request)
        
        assert result["max_tokens"] == 2000
        assert result["temperature"] == 0.3
        assert result["stream"] is True
        assert result["stop_sequences"] == ["END", "STOP"]
    
    def test_default_model(self):
        """Test default model retrieval."""
        transformer = AnthropicRequestTransformer()
        assert transformer.get_default_model() == "claude-sonnet-4-20250514"
    
    def test_model_validation(self):
        """Test model validation."""
        transformer = AnthropicRequestTransformer()
        
        # Valid models
        assert transformer.validate_model("claude-sonnet-4-20250514") is True
        assert transformer.validate_model("claude-opus-4-20250514") is True
        assert transformer.validate_model("claude-haiku-3-5-20241022") is True
        
        # Invalid models
        assert transformer.validate_model("gpt-4") is False
        assert transformer.validate_model("invalid-model") is False
    
    def test_response_parser_basic(self):
        """Test basic response parsing."""
        parser = AnthropicResponseParser()
        raw_response = {
            "content": [{"text": "Fixed the bug!"}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5
            }
        }
        
        request = UnifiedRequest(prompt="Fix this bug", model="claude-sonnet-4-20250514")
        result = parser.parse(raw_response, request, 200)
        
        assert result.content == "Fixed the bug!"
        assert result.model == "claude-sonnet-4-20250514"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5
        assert result.usage.total_tokens == 15
        assert result.latency_ms == 200
        assert result.finish_reason == "stop"  # Normalized from end_turn
        assert result.provider == "anthropic"
    
    def test_error_parsing(self):
        """Test error response parsing."""
        parser = AnthropicResponseParser()
        error_response = {
            "error": {
                "message": "Rate limit exceeded",
                "type": "rate_limit_error"
            }
        }
        
        result = parser.parse_error(error_response)
        assert result == "Anthropic rate_limit_error: Rate limit exceeded"


class TestOllamaTransformers:
    """Test Ollama request transformer and response parser."""
    
    def test_request_transformer_basic(self):
        """Test basic request transformation."""
        transformer = OllamaRequestTransformer()
        request = UnifiedRequest(
            prompt="Fix this bug",
            model="llama3.3"
        )
        
        result = transformer.transform(request)
        
        assert result["model"] == "llama3.3"
        assert result["prompt"] == "Fix this bug"
        assert result["stream"] is False
        assert result["options"]["temperature"] == 0.7
        assert result["options"]["num_ctx"] == 4096
    
    def test_request_transformer_with_system_message(self):
        """Test transformation with system message."""
        transformer = OllamaRequestTransformer()
        request = UnifiedRequest(
            prompt="Fix this bug",
            system_message="You are a Python expert",
            model="llama3.3"
        )
        
        result = transformer.transform(request)
        
        assert result["system"] == "You are a Python expert"
        assert result["prompt"] == "Fix this bug"
    
    def test_request_transformer_with_all_options(self):
        """Test transformation with all optional parameters."""
        transformer = OllamaRequestTransformer()
        request = UnifiedRequest(
            prompt="Fix this bug",
            max_tokens=1000,
            temperature=0.5,
            stream=True,
            stop_sequences=["END", "STOP"],
            model="llama3.3"
        )
        
        result = transformer.transform(request)
        
        assert result["options"]["num_predict"] == 1000
        assert result["options"]["temperature"] == 0.5
        assert result["stream"] is True
        assert result["options"]["stop"] == ["END", "STOP"]
    
    def test_default_model(self):
        """Test default model retrieval."""
        transformer = OllamaRequestTransformer()
        assert transformer.get_default_model() == "llama3.3"
    
    def test_model_validation(self):
        """Test model validation."""
        transformer = OllamaRequestTransformer()
        
        # Valid models
        assert transformer.validate_model("llama3.3") is True
        assert transformer.validate_model("mistral") is True
        assert transformer.validate_model("codellama") is True
        
        # Invalid models
        assert transformer.validate_model("gpt-4") is False
        assert transformer.validate_model("claude-3") is False
    
    def test_response_parser_basic(self):
        """Test basic response parsing."""
        parser = OllamaResponseParser()
        raw_response = {
            "model": "llama3.3",
            "response": "Fixed the bug!",
            "done_reason": "stop",
            "prompt_eval_count": 10,
            "eval_count": 5,
            "total_duration": 150_000_000  # 150ms in nanoseconds
        }
        
        request = UnifiedRequest(prompt="Fix this bug", model="llama3.3")
        result = parser.parse(raw_response, request, 100)
        
        assert result.content == "Fixed the bug!"
        assert result.model == "llama3.3"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5
        assert result.usage.total_tokens == 15
        assert result.latency_ms == 150  # Uses response duration over provided
        assert result.finish_reason == "stop"
        assert result.provider == "ollama"
    
    def test_response_parser_no_duration(self):
        """Test response parsing without duration information."""
        parser = OllamaResponseParser()
        raw_response = {
            "model": "llama3.3",
            "response": "Fixed the bug!",
            "done_reason": "stop",
            "prompt_eval_count": 10,
            "eval_count": 5
        }
        
        request = UnifiedRequest(prompt="Fix this bug", model="llama3.3")
        result = parser.parse(raw_response, request, 100)
        
        assert result.latency_ms == 100  # Uses provided latency
    
    def test_error_parsing(self):
        """Test error response parsing."""
        parser = OllamaResponseParser()
        error_response = {
            "error": {
                "message": "Model not found"
            }
        }
        
        result = parser.parse_error(error_response)
        assert result == "Ollama error: Model not found"


class TestTransformPipeline:
    """Test the complete transform pipeline."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = TransformPipelineConfig(
            provider_name="test-openai",
            default_model="gpt-4o",
            supported_models=["gpt-4o", "gpt-4", "gpt-3.5-turbo"],
            max_tokens_limit=8000,
            temperature_range=(0.0, 2.0)
        )
        
        self.transformer = OpenAIRequestTransformer()
        self.parser = OpenAIResponseParser()
        self.pipeline = TransformPipeline(self.transformer, self.parser, self.config)
    
    def test_process_request_basic(self):
        """Test basic request processing."""
        request = UnifiedRequest(
            prompt="Fix this bug",
            model="gpt-4o"
        )
        
        result = self.pipeline.process_request(request)
        
        assert result["model"] == "gpt-4o"
        assert result["messages"] == [{"role": "user", "content": "Fix this bug"}]
    
    def test_process_request_default_model(self):
        """Test request processing with default model."""
        request = UnifiedRequest(prompt="Fix this bug")  # No model specified
        
        result = self.pipeline.process_request(request)
        
        assert result["model"] == "gpt-4o"  # Should use default
    
    def test_process_request_invalid_model(self):
        """Test request processing with invalid model."""
        request = UnifiedRequest(
            prompt="Fix this bug",
            model="invalid-model"
        )
        
        with pytest.raises(ValueError, match="Model 'invalid-model' not supported"):
            self.pipeline.process_request(request)
    
    def test_process_request_invalid_temperature(self):
        """Test request processing with invalid temperature."""
        # Create request with valid temperature first
        request = UnifiedRequest(
            prompt="Fix this bug",
            model="gpt-4o",
            temperature=0.5
        )
        # Then manually set invalid temperature to test pipeline validation
        request.temperature = 3.0
        
        with pytest.raises(ValueError, match="Temperature 3.0 outside range"):
            self.pipeline.process_request(request)
    
    def test_process_request_max_tokens_exceeded(self):
        """Test request processing with max_tokens exceeded."""
        request = UnifiedRequest(
            prompt="Fix this bug",
            model="gpt-4o",
            max_tokens=10000  # Exceeds limit
        )
        
        with pytest.raises(ValueError, match="max_tokens 10000 exceeds limit"):
            self.pipeline.process_request(request)
    
    def test_process_response_basic(self):
        """Test basic response processing."""
        raw_response = {
            "choices": [
                {
                    "message": {"content": "Fixed!"},
                    "finish_reason": "stop"
                }
            ],
            "model": "gpt-4o",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }
        
        request = UnifiedRequest(prompt="Fix this bug", model="gpt-4o")
        result = self.pipeline.process_response(raw_response, request, 150)
        
        assert result.content == "Fixed!"
        assert result.provider == "test-openai"  # Should be set by pipeline
        assert result.latency_ms == 150
    
    def test_process_error_basic(self):
        """Test basic error processing."""
        error_response = {
            "error": {
                "message": "Invalid API key",
                "type": "authentication_error"
            }
        }
        
        result = self.pipeline.process_error(error_response)
        assert result == "OpenAI authentication_error: Invalid API key"
    
    def test_process_error_fallback(self):
        """Test error processing fallback."""
        # Create a mock parser that will raise an exception
        class FailingParser(OpenAIResponseParser):
            def parse_error(self, error_response):
                raise ValueError("Parser failed")
        
        # Replace parser temporarily
        original_parser = self.pipeline.parser
        self.pipeline.parser = FailingParser()
        
        try:
            error_response = {"unexpected": "format"}
            result = self.pipeline.process_error(error_response)
            assert result.startswith("Error from test-openai:")
            assert "unexpected" in result
        finally:
            # Restore original parser
            self.pipeline.parser = original_parser


class TestTransformPipelineConfig:
    """Test transform pipeline configuration."""
    
    def test_config_creation(self):
        """Test basic configuration creation."""
        config = TransformPipelineConfig(
            provider_name="test-provider",
            default_model="test-model",
            supported_models=["model1", "model2"]
        )
        
        assert config.provider_name == "test-provider"
        assert config.default_model == "test-model"
        assert config.supported_models == ["model1", "model2"]
        assert config.max_tokens_limit is None
        assert config.temperature_range == (0.0, 2.0)
    
    def test_config_with_limits(self):
        """Test configuration with custom limits."""
        config = TransformPipelineConfig(
            provider_name="test-provider",
            default_model="test-model",
            supported_models=["model1"],
            max_tokens_limit=5000,
            temperature_range=(0.1, 1.5)
        )
        
        assert config.max_tokens_limit == 5000
        assert config.temperature_range == (0.1, 1.5)


class TestProviderSpecificBehavior:
    """Test provider-specific behavior and edge cases."""
    
    def test_openai_messages_structure(self):
        """Test OpenAI messages structure is correct."""
        transformer = OpenAIRequestTransformer()
        
        # Test with system message
        request_with_system = UnifiedRequest(
            prompt="Fix this",
            system_message="You are helpful",
            model="gpt-4"
        )
        result = transformer.transform(request_with_system)
        
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][1]["role"] == "user"
        
        # Test without system message
        request_no_system = UnifiedRequest(prompt="Fix this", model="gpt-4")
        result = transformer.transform(request_no_system)
        
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"
    
    def test_anthropic_system_field(self):
        """Test Anthropic uses separate system field."""
        transformer = AnthropicRequestTransformer()
        request = UnifiedRequest(
            prompt="Fix this",
            system_message="You are helpful",
            model="claude-sonnet-4-20250514"
        )
        
        result = transformer.transform(request)
        
        # Should have system field, not in messages
        assert "system" in result
        assert result["system"] == "You are helpful"
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"
    
    def test_anthropic_default_max_tokens(self):
        """Test Anthropic sets default max_tokens."""
        transformer = AnthropicRequestTransformer()
        request = UnifiedRequest(
            prompt="Fix this",
            model="claude-sonnet-4-20250514"
        )
        
        result = transformer.transform(request)
        assert result["max_tokens"] == 4000  # Default value
    
    def test_ollama_options_structure(self):
        """Test Ollama uses options structure."""
        transformer = OllamaRequestTransformer()
        request = UnifiedRequest(
            prompt="Fix this",
            model="llama3.3",
            temperature=0.5,
            max_tokens=1000
        )
        
        result = transformer.transform(request)
        
        assert "options" in result
        assert result["options"]["temperature"] == 0.5
        assert result["options"]["num_predict"] == 1000
        assert result["options"]["num_ctx"] == 4096  # Default context window
    
    def test_finish_reason_normalization(self):
        """Test finish reason normalization across providers."""
        # Test OpenAI normalization
        assert FinishReason.normalize("stop", "openai") == "stop"
        assert FinishReason.normalize("length", "openai") == "length"
        
        # Test Anthropic normalization
        assert FinishReason.normalize("end_turn", "anthropic") == "stop"
        assert FinishReason.normalize("max_tokens", "anthropic") == "length"
        
        # Test Ollama normalization
        assert FinishReason.normalize("stop", "ollama") == "stop"
        assert FinishReason.normalize("length", "ollama") == "length"
        
        # Test unknown provider/reason defaults to stop
        assert FinishReason.normalize("unknown", "unknown") == "stop"