# Master Implementation Plan: Model Provider System with Unified Abstraction Layer

**Task ID**: MASTER-ProviderImplementation
**Status**: Not Started
**Created**: 2025-01-27
**Priority**: High

## Executive Summary

This is the master implementation plan for building three model providers (OpenAI, Anthropic, Ollama) with a unified abstraction layer. The plan leverages extensive API research documented in `Documentation/API-Contracts-Analysis.md` and builds upon the completed Phase 1 infrastructure.

**Goal**: Transform user experience from "write integration code" to "run one command with any model provider"

## Foundation Analysis

### Current State Assessment ✅
- **Phase 1 Infrastructure**: Completed (FEAT-ModelProviders-Phase1.md)
  - Base provider system with capabilities ✅
  - Thread-safe registry with validation ✅
  - Circuit breaker implementation ✅
  - Async/sync bridge ✅
  - Multi-source configuration management ✅
- **Generation System**: Existing robust framework ✅
- **Dataset Integration**: Existing DatasetManager ✅
- **CLI Framework**: Existing Click-based CLI ✅

### Research Foundation ✅
Based on `Documentation/API-Contracts-Analysis.md`, key abstraction challenges identified:
1. **Authentication**: Bearer tokens vs API keys vs none
2. **Request Formats**: Similar but incompatible JSON schemas
3. **Response Formats**: Different field names and structures
4. **Token Counting**: tiktoken vs API endpoint vs response metadata
5. **Rate Limiting**: Different headers, algorithms, and resource types
6. **Error Handling**: Different error codes, types, and recovery strategies
7. **Streaming**: SSE vs JSON Lines formats
8. **Resource Management**: API limits vs local hardware constraints

## Architecture Overview

### Unified Abstraction Layer Design

```
┌─────────────────────────────────────────────────────────────┐
│                    SWE-bench Runner CLI                      │
├─────────────────────────────────────────────────────────────┤
│                 Generation Integration                       │
│  - Unified patch generation workflow                         │
│  - Cost estimation and batch processing                      │
│  - Error handling and retry logic                            │
├─────────────────────────────────────────────────────────────┤
│                   Abstraction Layer                          │
│  - Unified ModelRequest/ModelResponse                        │
│  - Authentication Strategy Pattern                           │
│  - Request/Response Transform Pipeline                       │
│  - Token Counting Unification                               │
│  - Rate Limiting Coordination                               │
│  - Streaming Response Adapters                              │
├─────────────────┬───────────────┬───────────────────────────┤
│   OpenAI        │   Anthropic   │      Ollama               │
│   Provider      │   Provider    │      Provider             │
│  - API Client   │  - API Client │   - Local Client          │
│  - tiktoken     │  - Count API  │   - Response metadata     │
│  - SSE Stream   │  - SSE Stream │   - JSON Lines            │
├─────────────────┴───────────────┴───────────────────────────┤
│              Foundation Infrastructure (✅ Completed)        │
│  - Provider Registry  - Circuit Breaker  - Async Bridge     │
│  - Config Manager    - Exception System  - Base Classes     │
└─────────────────────────────────────────────────────────────┘
```

### Execution Paradigms Support

1. **API Providers** (OpenAI, Anthropic): Cloud-based, authenticated, rate-limited
2. **Local Execution** (Ollama): Hardware-constrained, no auth, direct HTTP
3. **Future Extensions**: HuggingFace Hub, Vertex AI, Azure OpenAI

## Implementation Phases

### Phase 2A: Unified Abstraction Layer (Foundation)
**Objective**: Build core abstractions that unify all provider differences

#### 2A.1 Core Data Models
```python
# Enhanced unified request/response models
@dataclass
class UnifiedRequest:
    """Provider-agnostic request format"""
    prompt: str
    system_message: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    stream: bool = False
    model: Optional[str] = None
    stop_sequences: Optional[List[str]] = None

@dataclass
class UnifiedResponse:
    """Provider-agnostic response format"""
    content: str
    model: str
    usage: TokenUsage
    latency_ms: int
    finish_reason: str
    provider: str
    cost: Optional[float] = None
    raw_response: dict = field(default_factory=dict)

@dataclass
class TokenUsage:
    """Unified token usage information"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
```

#### 2A.2 Authentication Strategy Pattern
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

class NoAuth(AuthStrategy):
    def prepare_headers(self, base_headers: dict) -> dict:
        return base_headers
```

#### 2A.3 Request/Response Transform Pipeline
```python
class RequestTransformer(ABC):
    @abstractmethod
    def transform(self, unified_request: UnifiedRequest) -> dict:
        pass

class ResponseParser(ABC):
    @abstractmethod
    def parse(self, raw_response: dict) -> UnifiedResponse:
        pass

class TransformPipeline:
    def __init__(self, transformer: RequestTransformer, parser: ResponseParser):
        self.transformer = transformer
        self.parser = parser

    def process_request(self, request: UnifiedRequest) -> dict:
        return self.transformer.transform(request)

    def process_response(self, raw_response: dict) -> UnifiedResponse:
        return self.parser.parse(raw_response)
```

#### 2A.4 Token Counting Unification
```python
class TokenCounter(ABC):
    @abstractmethod
    async def count_tokens(self, text: str, model: str) -> int:
        pass

class TiktokenCounter(TokenCounter):
    def count_tokens(self, text: str, model: str) -> int:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))

class APITokenCounter(TokenCounter):
    async def count_tokens(self, text: str, model: str) -> int:
        # Use provider's API endpoint
        response = await self.client.count_tokens(text, model)
        return response["input_tokens"]

class MetadataTokenCounter(TokenCounter):
    def count_tokens(self, text: str, model: str) -> int:
        # Extract from response metadata (Ollama)
        return self.last_response_metadata.get("prompt_eval_count", 0)
```

#### 2A.5 Streaming Response Adapters
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

#### 2A.6 Rate Limiting Coordinator
```python
class RateLimitCoordinator:
    def __init__(self, provider_config: dict):
        self.limiters = {}
        for provider, limits in provider_config.items():
            self.limiters[provider] = self._create_limiter(limits)

    async def acquire(self, provider: str, estimated_tokens: int) -> bool:
        limiter = self.limiters.get(provider)
        if not limiter:
            return True  # No limits configured

        return await limiter.acquire(estimated_tokens)

    def _create_limiter(self, limits: dict):
        # Create token bucket or sliding window limiter
        return TokenBucketLimiter(
            requests_per_minute=limits.get("requests_per_minute"),
            tokens_per_minute=limits.get("tokens_per_minute")
        )
```

### Phase 2B: OpenAI Provider Implementation
**Objective**: Complete, production-ready OpenAI provider with all edge cases handled

#### 2B.1 Enhanced OpenAI Provider
- Build upon existing `src/swebench_runner/providers/openai.py`
- Add comprehensive model support (GPT-4.1, GPT-4o, GPT-3.5-turbo variants)
- Implement proper token counting with tiktoken
- Add streaming support with SSE parsing
- Enhanced error handling with specific provider errors
- Cost calculation with latest pricing

#### 2B.2 OpenAI-Compatible Endpoints
- Support for vLLM endpoints
- Support for FastChat endpoints
- Support for llama.cpp endpoints
- Endpoint discovery and validation

#### 2B.3 OpenAI Request/Response Transformers
```python
class OpenAIRequestTransformer(RequestTransformer):
    def transform(self, request: UnifiedRequest) -> dict:
        openai_request = {
            "model": request.model,
            "messages": self._build_messages(request),
            "temperature": request.temperature,
            "stream": request.stream
        }

        if request.max_tokens:
            openai_request["max_tokens"] = request.max_tokens
        if request.stop_sequences:
            openai_request["stop"] = request.stop_sequences

        return openai_request

    def _build_messages(self, request: UnifiedRequest) -> List[dict]:
        messages = []
        if request.system_message:
            messages.append({"role": "system", "content": request.system_message})
        messages.append({"role": "user", "content": request.prompt})
        return messages

class OpenAIResponseParser(ResponseParser):
    def parse(self, raw_response: dict) -> UnifiedResponse:
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
            finish_reason=choice.get("finish_reason", "stop"),
            provider="openai",
            raw_response=raw_response
        )
```

### Phase 2C: Anthropic Provider Implementation
**Objective**: Native Anthropic Claude support with optimal API usage

#### 2C.1 Anthropic Provider Features
- Support for Claude 4 models (Opus, Sonnet, Haiku)
- Native Anthropic API client with proper headers
- Token counting via Anthropic API endpoint
- Prompt caching optimization
- System message handling (separate field vs messages)

#### 2C.2 Anthropic Request/Response Transformers
```python
class AnthropicRequestTransformer(RequestTransformer):
    def transform(self, request: UnifiedRequest) -> dict:
        anthropic_request = {
            "model": request.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": request.max_tokens or 4000,
            "temperature": request.temperature,
            "stream": request.stream
        }

        if request.system_message:
            anthropic_request["system"] = request.system_message
        if request.stop_sequences:
            anthropic_request["stop_sequences"] = request.stop_sequences

        return anthropic_request

class AnthropicResponseParser(ResponseParser):
    def parse(self, raw_response: dict) -> UnifiedResponse:
        content = raw_response["content"][0]["text"]
        usage = raw_response.get("usage", {})

        return UnifiedResponse(
            content=content,
            model=raw_response["model"],
            usage=TokenUsage(
                prompt_tokens=usage.get("input_tokens", 0),
                completion_tokens=usage.get("output_tokens", 0),
                total_tokens=usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            ),
            finish_reason=raw_response.get("stop_reason", "end_turn"),
            provider="anthropic",
            raw_response=raw_response
        )
```

#### 2C.3 Anthropic-Specific Features
- Implement prompt caching for cost optimization
- Handle rate limiting with token bucket algorithm
- Support for beta features via headers
- Proper streaming with multiple event types

### Phase 2D: Ollama Provider Implementation
**Objective**: Local model execution with resource management

#### 2D.1 Ollama Provider Features
- Local HTTP client for http://localhost:11434
- Model management (pull, list, delete)
- Resource monitoring and limits
- JSON Lines streaming support
- No authentication required

#### 2D.2 Ollama Request/Response Transformers
```python
class OllamaRequestTransformer(RequestTransformer):
    def transform(self, request: UnifiedRequest) -> dict:
        ollama_request = {
            "model": request.model,
            "prompt": request.prompt,
            "stream": request.stream,
            "options": {
                "temperature": request.temperature,
                "num_ctx": 4096,  # Context window
            }
        }

        if request.system_message:
            ollama_request["system"] = request.system_message
        if request.max_tokens:
            ollama_request["options"]["num_predict"] = request.max_tokens
        if request.stop_sequences:
            ollama_request["options"]["stop"] = request.stop_sequences

        return ollama_request

class OllamaResponseParser(ResponseParser):
    def parse(self, raw_response: dict) -> UnifiedResponse:
        return UnifiedResponse(
            content=raw_response["response"],
            model=raw_response["model"],
            usage=TokenUsage(
                prompt_tokens=raw_response.get("prompt_eval_count", 0),
                completion_tokens=raw_response.get("eval_count", 0),
                total_tokens=raw_response.get("prompt_eval_count", 0) + raw_response.get("eval_count", 0)
            ),
            finish_reason=raw_response.get("done_reason", "stop"),
            provider="ollama",
            latency_ms=int(raw_response.get("total_duration", 0) / 1_000_000),  # ns to ms
            raw_response=raw_response
        )
```

#### 2D.3 Ollama-Specific Features
- Model availability checking
- Resource usage monitoring
- Queue management for concurrent requests
- Graceful degradation when service unavailable

### Phase 3: Integration & CLI Enhancement
**Objective**: Seamless integration with existing generation system and enhanced CLI

#### 3.1 Enhanced Generation Integration
- Update `GenerationIntegration` class to use unified providers
- Add provider selection and model validation
- Implement cost estimation across all providers
- Enhanced batch processing with provider-specific optimizations

#### 3.2 CLI Provider Management
```bash
# Provider management commands
swebench provider list                    # List available providers
swebench provider init openai            # Initialize OpenAI provider
swebench provider test openai            # Test provider connection
swebench provider models openai          # List available models

# Generation with providers
swebench generate --provider openai --model gpt-4o --count 5
swebench generate --provider anthropic --model claude-sonnet-4 --count 5
swebench generate --provider ollama --model llama3.3 --count 5

# Comparison and evaluation
swebench compare --providers openai,anthropic --count 10
```

#### 3.3 Configuration Wizard
```python
# Interactive setup wizard
@cli.command()
def init():
    """Interactive provider setup wizard."""
    registry = get_registry()
    providers = registry.list_providers()

    # Show available providers
    selected = prompt_provider_selection(providers)

    # Provider-specific setup
    provider_class = registry.get_provider_class(selected)

    if provider_class.requires_api_key:
        api_key = click.prompt("API Key", hide_input=True)

    # Model selection
    if hasattr(provider_class, 'supported_models'):
        model = prompt_model_selection(provider_class.supported_models)

    # Test and save
    config = ProviderConfig(name=selected, api_key=api_key, model=model)
    test_provider_connection(provider_class, config)
    save_provider_config(config)
```

### Phase 4: Advanced Features
**Objective**: Production-ready features for enterprise use

#### 4.1 Multi-Provider Batch Processing
- Intelligent provider selection based on cost/speed tradeoffs
- Fallback between providers on failures
- Load balancing across multiple API keys
- Concurrent processing with rate limit coordination

#### 4.2 Cost Optimization
- Real-time cost tracking across providers
- Budget limits and warnings
- Cost-optimal model selection
- Batch size optimization for cost efficiency

#### 4.3 Performance Monitoring
- Latency tracking per provider
- Success rate monitoring
- Circuit breaker statistics
- Provider health dashboards

#### 4.4 Advanced Error Recovery
- Smart retry strategies per error type
- Automatic model fallback (GPT-4 → GPT-3.5)
- Checkpoint-based resume for long-running batches
- Partial result preservation

### Phase 5: Testing & Quality Assurance
**Objective**: Comprehensive testing strategy for production reliability

#### 5.1 Unit Testing Strategy
```python
# Test each abstraction layer component
class TestUnifiedAbstraction:
    def test_request_transformation(self):
        # Test OpenAI transformer
        transformer = OpenAIRequestTransformer()
        request = UnifiedRequest(prompt="test", system_message="system")
        result = transformer.transform(request)
        assert result["messages"][0]["role"] == "system"

    def test_response_parsing(self):
        # Test response parsing consistency
        parser = OpenAIResponseParser()
        mock_response = {"choices": [{"message": {"content": "test"}}]}
        result = parser.parse(mock_response)
        assert isinstance(result, UnifiedResponse)

    def test_token_counting(self):
        # Test token counting accuracy
        counter = TiktokenCounter()
        count = counter.count_tokens("Hello world", "gpt-4")
        assert count > 0
```

#### 5.2 Integration Testing
```python
class TestProviderIntegration:
    @pytest.mark.integration
    async def test_openai_end_to_end(self):
        provider = OpenAIProvider(config)
        request = UnifiedRequest(prompt="Say hello")
        response = await provider.generate_unified(request)
        assert response.content
        assert response.provider == "openai"

    @pytest.mark.integration
    async def test_provider_fallback(self):
        # Test fallback when primary provider fails
        coordinator = ProviderCoordinator(["openai", "anthropic"])
        response = await coordinator.generate_with_fallback(request)
        assert response.content
```

#### 5.3 Performance Testing
```python
class TestPerformance:
    @pytest.mark.benchmark
    async def test_concurrent_generation(self):
        # Test 10 concurrent generations
        tasks = [provider.generate_unified(request) for _ in range(10)]
        results = await asyncio.gather(*tasks)
        assert len(results) == 10
        assert all(r.content for r in results)

    @pytest.mark.benchmark
    def test_token_counting_performance(self):
        # Test token counting speed
        large_text = "word " * 10000
        start = time.time()
        count = counter.count_tokens(large_text, "gpt-4")
        duration = time.time() - start
        assert duration < 1.0  # Should be under 1 second
```

### Phase 6: Documentation & Examples
**Objective**: Comprehensive documentation for users and developers

#### 6.1 User Documentation
- Getting started guide with provider setup
- CLI reference with all commands and options
- Configuration guide for different environments
- Troubleshooting guide for common issues

#### 6.2 Developer Documentation
- Provider development guide for adding new providers
- API reference for all classes and methods
- Architecture overview with diagrams
- Extension points and customization guide

#### 6.3 Example Implementations
```python
# examples/basic_usage.py
from swebench_runner.providers import get_registry

async def basic_example():
    registry = get_registry()
    provider = registry.get_provider("openai")

    request = UnifiedRequest(
        prompt="Fix this Python bug: ...",
        system_message="You are a expert Python developer",
        max_tokens=1000
    )

    response = await provider.generate_unified(request)
    print(f"Generated patch: {response.content}")
    print(f"Cost: ${response.cost:.4f}")
    print(f"Tokens: {response.usage.total_tokens}")

# examples/batch_processing.py
async def batch_example():
    coordinator = ProviderCoordinator(["openai", "anthropic"])

    instances = load_swe_bench_instances("lite", limit=10)
    results = await coordinator.process_batch(instances)

    for result in results:
        print(f"Instance {result.instance_id}: {result.success}")
```

## Task Decomposition

### Phase 2A Tasks (Unified Abstraction Layer)
1. **Create unified data models** (UnifiedRequest, UnifiedResponse, TokenUsage)
2. **Implement authentication strategies** (Bearer, API Key, None)
3. **Build request/response transform pipeline** (base classes and interfaces)
4. **Create token counting unification** (tiktoken, API, metadata counters)
5. **Implement streaming adapters** (SSE, JSON Lines)
6. **Build rate limiting coordinator** (token bucket, sliding window)

### Phase 2B Tasks (OpenAI Provider)
7. **Enhance existing OpenAI provider** with unified interface
8. **Add OpenAI request transformer** (UnifiedRequest → OpenAI API format)
9. **Add OpenAI response parser** (OpenAI API → UnifiedResponse)
10. **Implement tiktoken integration** for accurate token counting
11. **Add streaming support** with SSE adapter
12. **Add comprehensive error handling** with OpenAI-specific errors

### Phase 2C Tasks (Anthropic Provider)
13. **Create Anthropic provider class** following provider interface
14. **Add Anthropic request transformer** (handle system messages properly)
15. **Add Anthropic response parser** (content array → unified format)
16. **Implement API token counting** via /v1/messages/count_tokens
17. **Add streaming support** with multi-event SSE handling
18. **Add rate limiting** with token bucket algorithm

### Phase 2D Tasks (Ollama Provider)
19. **Create Ollama provider class** for local execution
20. **Add Ollama request transformer** (prompt-based format)
21. **Add Ollama response parser** (JSON response → unified format)
22. **Implement metadata token counting** from response
23. **Add JSON Lines streaming** support
24. **Add resource monitoring** and health checks

### Phase 3 Tasks (Integration & CLI)
25. **Update GenerationIntegration** to use unified providers
26. **Add provider selection logic** and validation
27. **Implement cost estimation** across all providers
28. **Add CLI provider commands** (list, init, test, models)
29. **Add generation commands** with provider flags
30. **Create configuration wizard** for interactive setup

### Phase 4 Tasks (Advanced Features)
31. **Implement multi-provider coordinator** with fallback
32. **Add cost optimization** features and budgeting
33. **Add performance monitoring** and health checks
34. **Implement advanced error recovery** with smart retries

### Phase 5 Tasks (Testing)
35. **Create unit tests** for all abstraction components
36. **Add integration tests** for each provider
37. **Implement performance benchmarks** for concurrent usage
38. **Add end-to-end tests** with real API integration

### Phase 6 Tasks (Documentation)
39. **Write user documentation** and getting started guide
40. **Create developer documentation** and API reference
41. **Add example implementations** and tutorials
42. **Create troubleshooting guide** for common issues

## Success Criteria

### Technical Criteria
- [ ] All three providers (OpenAI, Anthropic, Ollama) working with unified interface
- [ ] Token counting accuracy within 1% across providers
- [ ] Rate limiting working correctly for all providers
- [ ] Streaming support functional for all providers
- [ ] Error handling comprehensive with proper recovery
- [ ] Performance: <100ms overhead for unified abstraction layer
- [ ] Memory usage: <50MB additional for provider system
- [ ] Test coverage: >95% for all provider code

### User Experience Criteria
- [ ] Single command setup: `swebench provider init openai`
- [ ] Single command generation: `swebench generate --provider openai --count 5`
- [ ] Zero-config operation with environment variables
- [ ] Clear error messages with actionable fix instructions
- [ ] Cost estimation before expensive operations
- [ ] Progress indication for long-running batches
- [ ] Graceful fallback between providers

### Integration Criteria
- [ ] Full compatibility with existing DatasetManager
- [ ] Seamless integration with existing generation system
- [ ] CLI commands work with all existing functionality
- [ ] No breaking changes to existing user workflows
- [ ] Support for all SWE-bench datasets (lite, verified, full)

## Risk Mitigation

### Technical Risks
1. **API Changes**: Abstract API details behind stable interfaces
2. **Rate Limiting**: Implement adaptive rate limiting with provider feedback
3. **Cost Overruns**: Add mandatory cost estimation and user confirmations
4. **Provider Outages**: Implement circuit breakers and provider fallback

### Implementation Risks
1. **Complexity**: Start with MVP and iterate based on feedback
2. **Performance**: Profile early and optimize hot paths
3. **Testing**: Use mocks for development, real APIs for integration tests
4. **Documentation**: Write docs alongside code to prevent drift

### User Experience Risks
1. **Setup Complexity**: Provide interactive wizard and clear documentation
2. **Error Messages**: Test error scenarios and improve messaging iteratively
3. **Configuration**: Support multiple configuration methods with precedence
4. **Migration**: Provide clear migration path from existing scripts

## Dependencies

### External Dependencies
```toml
# Required packages
providers = [
    "anthropic>=0.18.0",        # Anthropic Claude API
    "openai>=1.12.0",          # OpenAI API
    "tiktoken>=0.5.0",         # OpenAI token counting
    "aiohttp>=3.9.0",          # Async HTTP for Ollama
    "aiolimiter>=1.1.0",       # Rate limiting
]
```

### Internal Dependencies
- Existing provider infrastructure (✅ Phase 1 completed)
- DatasetManager for instance loading
- GenerationIntegration for batch processing
- CLI framework for command integration
- Configuration system for provider setup

## Decision Authority

You have full authority to make architectural decisions for this implementation, including:
- **Provider interface design**: Balancing consistency with provider-specific features
- **Error handling strategy**: Choosing between fail-fast vs graceful degradation
- **Configuration approach**: File-based vs environment vs interactive setup
- **Performance optimizations**: Caching, connection pooling, batching strategies
- **Testing strategy**: Unit vs integration vs end-to-end test balance

## Acceptance Criteria

### Minimum Viable Magic (MVP)
- OpenAI provider working with gpt-4 and gpt-3.5-turbo
- Anthropic provider working with Claude Sonnet and Haiku
- Ollama provider working with llama3 and mistral
- CLI commands: `provider init`, `provider test`, `generate --provider`
- Cost estimation and basic error handling
- Documentation for setup and basic usage

### Production Ready
- All providers with comprehensive model support
- Advanced features: streaming, cost optimization, monitoring
- Robust error handling with circuit breakers and fallback
- Comprehensive testing with >95% coverage
- Complete documentation with examples and troubleshooting

### Enterprise Features (Future)
- Multi-provider load balancing and coordination
- Advanced cost optimization and budgeting
- Performance monitoring and alerting
- Provider plugin system for easy extensions

## Implementation Timeline

**Phase 2A** (Unified Abstraction): 2-3 days
**Phase 2B** (OpenAI Provider): 2-3 days
**Phase 2C** (Anthropic Provider): 2-3 days
**Phase 2D** (Ollama Provider): 2-3 days
**Phase 3** (Integration & CLI): 2-3 days
**Phase 4** (Advanced Features): 3-4 days
**Phase 5** (Testing): 2-3 days
**Phase 6** (Documentation): 1-2 days

**Total Estimated Duration**: 16-24 days

## Notes

This plan leverages the extensive research in `API-Contracts-Analysis.md` and builds upon the solid foundation from Phase 1. The unified abstraction layer design ensures that adding new providers in the future will be straightforward, while the three initial providers (OpenAI, Anthropic, Ollama) provide comprehensive coverage of the main execution paradigms.

The emphasis on user experience ("just works") and production reliability aligns with the project's goal of making SWE-bench evaluation accessible to researchers and developers without requiring them to write integration code.
