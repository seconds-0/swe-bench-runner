# Provider API Contracts Analysis

This document provides comprehensive analysis of the three provider APIs we will implement: OpenAI, Anthropic Claude, and Ollama. This analysis forms the foundation for our unified provider abstraction layer.

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [OpenAI API Contract](#openai-api-contract)
3. [Anthropic Claude API Contract](#anthropic-claude-api-contract)
4. [Ollama API Contract](#ollama-api-contract)
5. [API Comparison Matrix](#api-comparison-matrix)
6. [Unified Abstraction Requirements](#unified-abstraction-requirements)
7. [Implementation Challenges](#implementation-challenges)

---

## Executive Summary

### Provider Classification

**API Providers (Cloud-based)**:
- **OpenAI**: Industry standard, token-based pricing, tiktoken counting
- **Anthropic**: Similar paradigm but different auth, pricing, token counting

**Local Execution**:
- **Ollama**: Local model execution, no auth, hardware-constrained

### Key Abstraction Challenges

1. **Authentication**: Bearer tokens vs API keys vs none
2. **Request Formats**: Similar but incompatible JSON schemas
3. **Response Formats**: Different field names and structures
4. **Token Counting**: tiktoken vs API endpoint vs response metadata
5. **Rate Limiting**: Different headers, algorithms, and resource types
6. **Error Handling**: Different error codes, types, and recovery strategies
7. **Streaming**: SSE vs JSON Lines formats
8. **Resource Management**: API limits vs local hardware constraints

---

## OpenAI API Contract

### Base Information
- **Documentation**: https://platform.openai.com/docs/api-reference
- **Base URL**: `https://api.openai.com/v1`
- **Primary Endpoint**: `/chat/completions`
- **Authentication**: Bearer token in Authorization header
- **Content Type**: `application/json`

### Authentication Schema
```bash
Authorization: Bearer YOUR_API_KEY
Content-Type: application/json
OpenAI-Organization: YOUR_ORG_ID  # Optional
OpenAI-Project: YOUR_PROJECT_ID   # Optional
```

### Request Schema
```json
{
  "model": "gpt-4o",                    // Required: Model identifier
  "messages": [                         // Required: Conversation array
    {
      "role": "system|user|assistant",  // Required: Message role
      "content": "string"               // Required: Message content
    }
  ],
  "temperature": 0.7,                   // Optional: 0-2, controls randomness
  "max_tokens": 150,                    // Optional: Max response tokens
  "top_p": 1.0,                        // Optional: Nucleus sampling
  "frequency_penalty": 0.0,             // Optional: -2.0 to 2.0
  "presence_penalty": 0.0,              // Optional: -2.0 to 2.0
  "stream": false,                      // Optional: Enable streaming
  "response_format": {                  // Optional: JSON mode
    "type": "json_object"
  },
  "tools": [],                          // Optional: Function calling
  "tool_choice": "auto"                 // Optional: Tool selection
}
```

### Response Schema
```json
{
  "id": "chatcmpl-123456",             // Unique completion ID
  "object": "chat.completion",          // Response type
  "created": 1677652288,               // Unix timestamp
  "model": "gpt-4o",                   // Model used
  "choices": [
    {
      "index": 0,                      // Choice index
      "message": {
        "role": "assistant",            // Response role
        "content": "Response text"     // Generated content
      },
      "finish_reason": "stop|length|tool_calls|content_filter"  // Completion reason
    }
  ],
  "usage": {
    "prompt_tokens": 12,               // Input token count
    "completion_tokens": 18,           // Output token count
    "total_tokens": 30                 // Total tokens
  }
}
```

### Available Models (2025)
- `gpt-4.1`: 1M context, 32K output tokens
- `gpt-4.1-mini`: Fast, cheap, 1M context
- `gpt-4o`: Multimodal, 128K context
- `gpt-4o-mini`: Cost-optimized
- `gpt-4`: Standard, 128K context
- `gpt-3.5-turbo`: Fast and economical

### Rate Limiting
**Headers**:
```
x-ratelimit-limit-requests: 3000
x-ratelimit-remaining-requests: 2999
x-ratelimit-reset-requests: 60
x-ratelimit-limit-tokens: 40000
x-ratelimit-remaining-tokens: 39950
x-ratelimit-reset-tokens: 60
```

**Limits**: Requests per minute (RPM) + Tokens per minute (TPM)

### Token Counting
**Library**: `tiktoken`
```python
import tiktoken
encoding = tiktoken.encoding_for_model("gpt-4o")
token_count = len(encoding.encode(text))
```

**Encodings**:
- `o200k_base`: GPT-4o models
- `cl100k_base`: GPT-4, GPT-3.5-turbo

### Pricing (2025)
**GPT-4o**: $5 input, $20 output per 1M tokens
**GPT-4**: $10 input, $30 output per 1M tokens
**GPT-3.5-turbo**: $1.50 input, $2.00 output per 1M tokens

### Error Responses
```json
{
  "error": {
    "message": "Detailed error description",
    "type": "invalid_request_error|authentication_error|rate_limit_error|server_error",
    "param": "parameter_name",
    "code": "error_code"
  }
}
```

**Status Codes**: 400, 401, 403, 404, 429, 500, 503

### Streaming Format
**Server-Sent Events** with `data:` prefix:
```
data: {"id":"chatcmpl-123","choices":[{"delta":{"content":"Hello"}}]}
data: {"id":"chatcmpl-123","choices":[{"delta":{"content":" world"}}]}
data: [DONE]
```

---

## Anthropic Claude API Contract

### Base Information
- **Documentation**: https://docs.anthropic.com/en/api/messages
- **Base URL**: `https://api.anthropic.com/v1`
- **Primary Endpoint**: `/messages`
- **Authentication**: API key in x-api-key header
- **Content Type**: `application/json`

### Authentication Schema
```bash
x-api-key: $ANTHROPIC_API_KEY      # Required
anthropic-version: 2023-06-01      # Required
content-type: application/json     # Required
anthropic-beta: comma-separated    # Optional: Beta features
```

### Request Schema
```json
{
  "model": "claude-sonnet-4-20250514",  // Required: Model identifier
  "messages": [                         // Required: Conversation array
    {
      "role": "user|assistant",         // Required: Message role (no system)
      "content": "string"               // Required: Message content
    }
  ],
  "max_tokens": 1024,                   // Required: Max tokens to generate
  "system": "You are helpful",          // Optional: System message (separate)
  "temperature": 0.7,                   // Optional: 0.0-1.0
  "top_p": 0.99,                       // Optional: 0.95-1.0
  "stream": false,                      // Optional: Enable streaming
  "stop_sequences": ["string"],         // Optional: Stop sequences
  "tools": []                           // Optional: Tool definitions
}
```

### Response Schema
```json
{
  "id": "msg_01XFDUDYJgAACzvnptvVoYEL",  // Unique message ID
  "type": "message",                     // Response type
  "role": "assistant",                   // Response role
  "content": [                           // Content array
    {
      "type": "text",                    // Content type
      "text": "Hello! How can I help?"    // Generated text
    }
  ],
  "model": "claude-sonnet-4-20250514",   // Model used
  "stop_reason": "end_turn",             // Completion reason
  "stop_sequence": null,                 // Stop sequence hit
  "usage": {
    "input_tokens": 12,                  // Input token count
    "output_tokens": 15                  // Output token count
  }
}
```

### Available Models (2025)
- `claude-opus-4-20250514`: Most capable, expensive
- `claude-sonnet-4-20250514`: Balanced performance
- `claude-haiku-3-5-20241022`: Fast, cost-effective

**Context**: 200K tokens (Claude 4 models)

### Rate Limiting
**Headers**:
```
anthropic-ratelimit-requests-remaining: 1000
anthropic-ratelimit-tokens-limit: 100000
anthropic-ratelimit-input-tokens-reset: 2024-01-01T00:00:00Z
retry-after: 60
```

**Algorithm**: Token bucket with continuous replenishment

### Token Counting
**API Endpoint**: `/v1/messages/count_tokens`
```json
{
  "model": "claude-sonnet-4-20250514",
  "messages": [...]
}
```

Response:
```json
{
  "input_tokens": 12
}
```

### Pricing (2025)
**Claude Opus 4**: $15 input, $75 output per 1M tokens
**Claude Sonnet 4**: $3 input, $15 output per 1M tokens
**Claude Haiku 3.5**: $0.80 input, $4 output per 1M tokens

**Additional**: Prompt caching discounts up to 90%

### Error Responses
```json
{
  "type": "error",
  "error": {
    "type": "rate_limit_error|authentication_error|invalid_request_error",
    "message": "Request rate too high. Please slow down and try again."
  }
}
```

**Status Codes**: 400, 401, 403, 404, 413, 429, 500, 529

### Streaming Format
**Server-Sent Events** with multiple event types:
- `message_start`: Stream initialization
- `content_block_delta`: Incremental content
- `message_stop`: Stream completion
- `ping`: Keep-alive events

---

## Ollama API Contract

### Base Information
- **Documentation**: https://github.com/ollama/ollama/blob/main/docs/api.md
- **Base URL**: `http://localhost:11434`
- **Primary Endpoint**: `/api/generate`
- **Authentication**: None (local execution)
- **Content Type**: `application/json`

### Request Schema
```json
{
  "model": "llama3.2",                  // Required: Model name
  "prompt": "string",                   // Required: Input prompt
  "suffix": "string",                   // Optional: Text to append
  "system": "string",                   // Optional: System message
  "context": [int],                     // Optional: Conversation context
  "stream": true,                       // Optional: Stream response (default: true)
  "raw": false,                         // Optional: Bypass prompt template
  "format": {},                         // Optional: JSON schema for structured output
  "keep_alive": "5m",                   // Optional: Model memory duration
  "images": ["base64"],                 // Optional: Images for vision models
  "options": {                          // Optional: Model parameters
    "num_ctx": 2048,                    // Context window size
    "temperature": 0.7,                 // Randomness 0.0-1.0
    "top_p": 0.9,                      // Nucleus sampling
    "top_k": 40,                       // Top-k sampling
    "repeat_penalty": 1.1,             // Repetition penalty
    "seed": 42,                        // Reproducible seed
    "num_predict": 128,                // Max tokens (-1=infinite, -2=fill)
    "stop": ["string"]                 // Stop sequences
  }
}
```

### Response Schema
```json
{
  "model": "llama3.2",                  // Model name used
  "created_at": "2025-01-27T...",       // ISO timestamp
  "response": "Generated text",          // Response content
  "done": true,                         // Completion status
  "done_reason": "stop",                // Completion reason
  "context": [int],                     // Context for continuation
  "total_duration": 302810709,          // Total time (nanoseconds)
  "load_duration": 13315375,            // Model load time (nanoseconds)
  "prompt_eval_count": 47,              // Input token count
  "prompt_eval_duration": 132000000,    // Prompt processing time (nanoseconds)
  "eval_count": 30,                     // Output token count
  "eval_duration": 156000000            // Generation time (nanoseconds)
}
```

### Available Models (2025)
**Popular Models**:
- `deepseek-r1`: Advanced reasoning
- `llama3.3`: Meta's latest (8B, 70B variants)
- `phi-4`: Microsoft 14B parameter
- `qwen2.5`: Multilingual, 128K context
- `mistral-small`: Efficient reasoning
- `codellama:34b`: Coding specialist

**Model Management**:
- Download: `ollama pull model-name`
- List: `GET /api/tags`
- Delete: `DELETE /api/delete`

### Rate Limiting & Resource Management
**Configuration Variables**:
- `OLLAMA_MAX_QUEUE`: Max queued requests (default: 512)
- `OLLAMA_NUM_PARALLEL`: Parallel requests per model (default: 1)
- `OLLAMA_MAX_LOADED_MODELS`: Max concurrent models (default: 3 * GPUs)

**No built-in rate limiting** - constrained by local hardware

### Token Counting
**Included in response metadata**:
- `prompt_eval_count`: Input tokens
- `eval_count`: Output tokens
- Control via `num_predict` option

### Resource Costs
**Free to use** but limited by:
- Available RAM/VRAM
- CPU/GPU processing power
- Local storage for models (1GB-70GB+ per model)

### Error Responses
```json
{
  "error": {
    "message": "unexpected server status: 1",
    "type": "api_error",
    "param": null,
    "code": null
  }
}
```

**Common Errors**:
- 404: Model not found (need to pull first)
- 500: Insufficient memory
- Connection refused: Service not running

### Streaming Format
**JSON Lines** (not SSE) over HTTP:
```json
{"model":"llama3.2","response":"The","done":false}
{"model":"llama3.2","response":" sky","done":false}
{"model":"llama3.2","response":"","done":true,"total_duration":302810709,...}
```

---

## API Comparison Matrix

| Feature | OpenAI | Anthropic | Ollama |
|---------|--------|-----------|--------|
| **Authentication** | Bearer token | API key header | None |
| **Base URL** | api.openai.com | api.anthropic.com | localhost:11434 |
| **Endpoint** | /chat/completions | /messages | /api/generate |
| **System Messages** | In messages array | Separate field | In prompt/options |
| **Required Fields** | model, messages | model, messages, max_tokens | model, prompt |
| **Token Counting** | tiktoken library | API endpoint | Response metadata |
| **Rate Limiting** | RPM + TPM | Token bucket | Hardware constrained |
| **Streaming Format** | Server-Sent Events | Server-Sent Events | JSON Lines |
| **Error Format** | Nested error object | Type + error object | Simple error object |
| **Pricing Model** | Per-token usage | Per-token usage | Free (hardware cost) |
| **Context Windows** | 128K-1M tokens | 200K tokens | Model dependent |
| **Model Management** | Cloud-hosted | Cloud-hosted | Local download required |

---

## Unified Abstraction Requirements

### Core Interface
Our unified provider interface must abstract these differences:

```python
@dataclass
class ModelRequest:
    """Unified request format across all providers"""
    prompt: str
    system_message: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    stream: bool = False
    model: Optional[str] = None

@dataclass
class ModelResponse:
    """Unified response format across all providers"""
    content: str
    model: str
    usage: TokenUsage
    latency_ms: int
    finish_reason: str
    provider: str
    raw_response: dict

@dataclass
class TokenUsage:
    """Unified token usage information"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
```

### Provider Configuration
```python
@dataclass
class ProviderConfig:
    """Provider-specific configuration"""
    name: str
    auth_config: AuthConfig
    rate_limits: RateLimitConfig
    model_config: ModelConfig
    endpoint_config: EndpointConfig

@dataclass
class AuthConfig:
    """Authentication configuration"""
    auth_type: AuthType  # bearer_token, api_key, none
    credentials: dict
    headers: dict

@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    requests_per_minute: Optional[int]
    tokens_per_minute: Optional[int]
    concurrent_requests: Optional[int]
    burst_allowance: Optional[int]
```

### Provider Interface
```python
class ModelProvider(ABC):
    """Abstract base class for all providers"""

    @abstractmethod
    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate a response for the given request"""

    @abstractmethod
    def estimate_cost(self, request: ModelRequest) -> float:
        """Estimate the cost for this request"""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in the given text"""

    @abstractmethod
    async def validate_config(self) -> bool:
        """Validate provider configuration"""

    @property
    @abstractmethod
    def available_models(self) -> List[str]:
        """List of available models"""
```

---

## Implementation Challenges

### 1. Authentication Abstraction
**Challenge**: Three different auth patterns
**Solution**: Pluggable auth strategy pattern
```python
class AuthStrategy(ABC):
    @abstractmethod
    def prepare_headers(self, base_headers: dict) -> dict:
        pass

class BearerTokenAuth(AuthStrategy):
    def prepare_headers(self, base_headers: dict) -> dict:
        return {**base_headers, "Authorization": f"Bearer {self.token}"}

class ApiKeyAuth(AuthStrategy):
    def prepare_headers(self, base_headers: dict) -> dict:
        return {**base_headers, "x-api-key": self.api_key}
```

### 2. Request Format Transformation
**Challenge**: Incompatible JSON schemas
**Solution**: Provider-specific request builders
```python
class RequestBuilder(ABC):
    @abstractmethod
    def build_request(self, unified_request: ModelRequest) -> dict:
        pass

class OpenAIRequestBuilder(RequestBuilder):
    def build_request(self, request: ModelRequest) -> dict:
        return {
            "model": request.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "temperature": request.temperature,
            "stream": request.stream
        }
```

### 3. Response Format Normalization
**Challenge**: Different response structures
**Solution**: Provider-specific response parsers
```python
class ResponseParser(ABC):
    @abstractmethod
    def parse_response(self, raw_response: dict) -> ModelResponse:
        pass

class OpenAIResponseParser(ResponseParser):
    def parse_response(self, raw_response: dict) -> ModelResponse:
        choice = raw_response["choices"][0]
        return ModelResponse(
            content=choice["message"]["content"],
            model=raw_response["model"],
            usage=self._parse_usage(raw_response["usage"]),
            finish_reason=choice["finish_reason"],
            provider="openai",
            raw_response=raw_response
        )
```

### 4. Token Counting Unification
**Challenge**: Different token counting methods
**Solution**: Provider-specific token counters
```python
class TokenCounter(ABC):
    @abstractmethod
    def count_tokens(self, text: str, model: str) -> int:
        pass

class OpenAITokenCounter(TokenCounter):
    def count_tokens(self, text: str, model: str) -> int:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))

class AnthropicTokenCounter(TokenCounter):
    async def count_tokens(self, text: str, model: str) -> int:
        # Use API endpoint
        response = await self.client.count_tokens(...)
        return response["input_tokens"]
```

### 5. Rate Limiting Abstraction
**Challenge**: Different rate limiting patterns
**Solution**: Unified rate limiter with provider adapters
```python
class RateLimiter:
    def __init__(self, config: RateLimitConfig):
        self.requests_bucket = TokenBucket(config.requests_per_minute)
        self.tokens_bucket = TokenBucket(config.tokens_per_minute)

    async def acquire(self, estimated_tokens: int) -> bool:
        request_ok = await self.requests_bucket.acquire(1)
        tokens_ok = await self.tokens_bucket.acquire(estimated_tokens)
        return request_ok and tokens_ok
```

### 6. Streaming Response Handling
**Challenge**: SSE vs JSON Lines formats
**Solution**: Streaming response adapters
```python
class StreamingAdapter(ABC):
    @abstractmethod
    async def stream_response(self, response_stream) -> AsyncIterator[str]:
        pass

class SSEAdapter(StreamingAdapter):
    async def stream_response(self, response_stream) -> AsyncIterator[str]:
        async for line in response_stream:
            if line.startswith("data: "):
                data = json.loads(line[6:])
                if content := self._extract_content(data):
                    yield content

class JSONLinesAdapter(StreamingAdapter):
    async def stream_response(self, response_stream) -> AsyncIterator[str]:
        async for line in response_stream:
            data = json.loads(line)
            if not data.get("done") and (content := data.get("response")):
                yield content
```

### 7. Error Handling Unification
**Challenge**: Different error formats and codes
**Solution**: Unified error classification system
```python
class ProviderError(Exception):
    def __init__(self, category: ErrorCategory, message: str,
                 retry_strategy: RetryStrategy, original_error: Exception):
        self.category = category
        self.message = message
        self.retry_strategy = retry_strategy
        self.original_error = original_error

class ErrorClassifier:
    def classify_error(self, error: Exception, provider: str) -> ProviderError:
        # Provider-specific error classification logic
        pass
```

This comprehensive analysis provides the foundation for implementing a robust, unified provider abstraction layer that can seamlessly handle the differences between OpenAI, Anthropic, and Ollama APIs while maintaining a clean, consistent interface for the rest of the application.
