# Work Plan: Model Provider Integration - Making SWE-bench Magically Easy

**Task ID**: FEAT-ModelProviders
**Status**: Not Started
**Last Updated**: 2025-01-25 - Balanced robustness with practical implementation for V1

## Problem Statement

Current SWE-bench runner requires users to write Python scripts to generate patches from their models. This is a significant barrier to entry. Users want to test their models (private, fine-tuned, or via various providers) with a single command, not write integration code.

The goal: Transform the experience from "write code to test your model" to "just run `swebench test` and it works with ANY model."

## Proposed Solution

Create a plugin-based provider system that abstracts away all model communication complexity. Users should be able to connect to any model (API, local, or cloud) through a simple interactive setup or command-line flags.

### Core Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SWE-bench Runner CLI                      │
├─────────────────────────────────────────────────────────────┤
│                    Provider Registry                         │
│  - Explicit provider registration                            │
│  - Configuration management (env vars → keyring → file)     │
│  - Provider lifecycle management                             │
├─────────────────────┬───────────────┬───────────────────────┤
│   OpenAI Provider   │  OpenRouter   │   Future Providers    │
│   (+ Compatible)    │   Provider    │   (Plugin Ready)      │
├─────────────────────┴───────────────┴───────────────────────┤
│              Robust Patch Generation Engine                  │
│  - Smart prompt templates                                    │
│  - Multi-strategy response parsing                           │
│  - Intelligent retry with backoff                            │
│  - Circuit breaker protection                                │
└─────────────────────────────────────────────────────────────┘
```

### Provider Interface (Production-Ready)

```python
# src/swebench_runner/providers/base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelResponse:
    """Standard response from any model provider."""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    cost: Optional[float] = None
    latency_ms: Optional[int] = None
    provider: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ProviderConfig:
    """Configuration for a provider."""
    name: str
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    model: Optional[str] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.extra_params is None:
            self.extra_params = {}

@dataclass
class ProviderCapabilities:
    """Declare provider capabilities for smart routing."""
    max_context_length: int = 4096
    supports_streaming: bool = False
    supports_json_mode: bool = False
    rate_limits: Optional[Dict[str, int]] = None
    supported_models: List[str] = field(default_factory=list)

class ModelProvider(ABC):
    """Base class for all model providers with production features."""

    # Provider metadata
    name: str = ""
    description: str = ""
    api_version: str = "1.0"
    requires_api_key: bool = True

    def __init__(self, config: ProviderConfig):
        self.config = config
        self.capabilities = self._init_capabilities()
        self._initialized = False
        self._health_status = "unknown"

    @abstractmethod
    def _init_capabilities(self) -> ProviderCapabilities:
        """Initialize provider-specific capabilities."""
        pass

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate a response from the model."""
        pass

    async def validate_connection(self) -> bool:
        """Test if the provider is properly configured and accessible."""
        try:
            # Default implementation - try a minimal generation
            test_response = await self.generate(
                "Respond with 'OK' if you receive this.",
                max_tokens=10
            )
            self._health_status = "healthy"
            return bool(test_response.content)
        except Exception as e:
            logger.warning(f"Provider {self.name} validation failed: {e}")
            self._health_status = "unhealthy"
            return False

    @abstractmethod
    def estimate_cost(self, prompt_tokens: int, max_tokens: int) -> float:
        """Estimate the cost of a generation."""
        pass

    def get_generation_params(self) -> Dict[str, Any]:
        """Get provider-specific generation parameters."""
        return {
            "temperature": 0,
            "max_tokens": 4000,
        }

    async def health_check(self) -> Dict[str, Any]:
        """Get current health status."""
        return {
            "status": self._health_status,
            "provider": self.name,
            "model": self.config.model,
            "timestamp": datetime.utcnow().isoformat()
        }
```

### Provider Manager

```python
# src/swebench_runner/providers/manager.py
class ProviderManager:
    """Manages all available providers and configurations."""

    def __init__(self):
        self.providers: Dict[str, Type[ModelProvider]] = {}
        self.configs: Dict[str, ProviderConfig] = {}
        self._load_providers()
        self._load_configs()

    def _load_providers(self):
        """Dynamically load all provider plugins."""
        # Auto-discover providers in providers/ directory

    def register_provider(self, provider_class: Type[ModelProvider]):
        """Register a new provider."""
        self.providers[provider_class.name] = provider_class

    def get_provider(self, name: str) -> ModelProvider:
        """Get an initialized provider by name."""
        if name not in self.configs:
            raise ValueError(f"Provider {name} not configured. Run 'swebench init'")

        provider_class = self.providers[name]
        return provider_class(self.configs[name])

    def list_providers(self) -> List[Dict[str, str]]:
        """List all available providers."""
        return [
            {
                "name": p.name,
                "description": p.description,
                "configured": p.name in self.configs
            }
            for p in self.providers.values()
        ]
```

## Implementation Phases

### Phase 0: Core Infrastructure (Foundation)

**Objective**: Build the plugin system and core abstractions.

**Components**:
1. **Provider Plugin System**
   - Base `ModelProvider` abstract class
   - Plugin discovery mechanism
   - Configuration management (encrypted storage)
   - Provider manager for plugin lifecycle

2. **Patch Generation Engine**
   ```python
   class PatchGenerator:
       def __init__(self, provider: ModelProvider):
           self.provider = provider
           self.prompt_template = self._load_template()

       async def generate_patch(self, issue: Dict) -> str:
           # Construct prompt
           prompt = self.prompt_template.format(**issue)

           # Generate with retry logic
           for attempt in range(3):
               response = await self.provider.generate(prompt)
               patch = response.content

               if self._validate_patch_format(patch):
                   return patch

               # Retry with format correction prompt
               prompt = self._correction_prompt(patch)

           raise ValueError("Failed to generate valid patch")
   ```

3. **CLI Integration**
   ```python
   # src/swebench_runner/cli.py
   @cli.command()
   def init():
       """Interactive setup wizard."""
       manager = ProviderManager()
       providers = manager.list_providers()

       # Show TUI for provider selection
       selected = prompt_provider_selection(providers)

       # Run provider-specific setup
       provider_class = manager.providers[selected]
       config = provider_class.interactive_setup()

       # Save configuration
       manager.save_config(selected, config)

   @cli.command()
   @click.option('--provider', help='Model provider to use')
   @click.option('--model', help='Model name')
   @click.option('--count', default=5, help='Number of issues to test')
   def test(provider, model, count):
       """Run SWE-bench evaluation."""
       manager = ProviderManager()

       if not provider:
           provider = manager.get_default_provider()

       provider_instance = manager.get_provider(provider)
       generator = PatchGenerator(provider_instance)

       # Run evaluation with progress bar
       run_evaluation(generator, count)
   ```

4. **Configuration Storage**
   ```python
   # ~/.swebench/config.json (encrypted)
   {
       "default_provider": "openai",
       "providers": {
           "openai": {
               "api_key": "encrypted:...",
               "model": "gpt-4-turbo",
               "endpoint": "https://api.openai.com/v1"
           }
       }
   }
   ```

**Success Criteria**:
- Plugin system can dynamically load providers
- Configuration is securely stored and retrieved
- Basic CLI commands work with mock provider

### Phase 1: OpenAI-Compatible Provider (Most Common)

**Objective**: Support OpenAI and all compatible APIs (vLLM, FastChat, etc.)

**Implementation**:
```python
# src/swebench_runner/providers/openai_compatible.py
class OpenAICompatibleProvider(ModelProvider):
    name = "openai"
    description = "OpenAI and compatible APIs (vLLM, FastChat, llama.cpp)"

    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 4000,
            **kwargs
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.config.endpoint}/chat/completions",
                headers=headers,
                json=data
            ) as response:
                result = await response.json()

                return ModelResponse(
                    content=result["choices"][0]["message"]["content"],
                    model=result["model"],
                    usage=result.get("usage"),
                    cost=self._calculate_cost(result.get("usage"))
                )
**Features**:
- Full error handling with specific messages
- Rate limit detection and retry hints
- Cost calculation and estimation
- Support for OpenAI-compatible endpoints (vLLM, etc.)
- Organization ID support
- Health check via models endpoint
```

**Supported Endpoints**:
- OpenAI: `https://api.openai.com/v1`
- Local vLLM: `http://localhost:8000/v1`
- FastChat: `http://localhost:7860/v1`
- llama.cpp: `http://localhost:8080/v1`

### Phase 2: Anthropic Provider

**Objective**: Native support for Claude models.

**Implementation**:
```python
# src/swebench_runner/providers/anthropic.py
class AnthropicProvider(ModelProvider):
    name = "anthropic"
    description = "Anthropic Claude models"
    supported_models = [
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
        "claude-3-5-sonnet-20241022"
    ]

    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        headers = {
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        data = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4000,
            "temperature": 0
        }

        # Anthropic-specific handling
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data
            ) as response:
                result = await response.json()

                return ModelResponse(
                    content=result["content"][0]["text"],
                    model=result["model"],
                    usage=result["usage"],
                    cost=self._calculate_anthropic_cost(result["usage"])
                )
```

### Phase 3: OpenRouter Provider (100+ Models)

**Objective**: Access to diverse models through OpenRouter.

**Features**:
- Model discovery from OpenRouter API
- Cost optimization
- Fallback handling

```python
# src/swebench_runner/providers/openrouter.py
class OpenRouterProvider(ModelProvider):
    name = "openrouter"
    description = "Access 100+ models through OpenRouter"

    async def _fetch_available_models(self) -> List[Dict]:
        """Fetch available models and their pricing."""
        async with aiohttp.ClientSession() as session:
            async with session.get("https://openrouter.ai/api/v1/models") as resp:
                data = await resp.json()
                # Filter for code-capable models
                return [
                    m for m in data["data"]
                    if any(tag in m.get("tags", []) for tag in ["code", "programming"])
                ]

    @classmethod
    def interactive_setup(cls) -> ProviderConfig:
        api_key = click.prompt("OpenRouter API Key", hide_input=True)

        # Fetch and display available models
        models = asyncio.run(cls._fetch_available_models())

        click.echo("\nPopular code models:")
        for i, model in enumerate(models[:10]):
            price = model['pricing']
            click.echo(f"  [{i+1}] {model['id']} - ${price['prompt']}/1k tokens")

        choice = click.prompt("Select model (number or name)", type=str)
        # Handle selection...
```

### Phase 4: Groq Provider (Ultra-Fast)

**Objective**: Leverage Groq's speed for rapid iteration.

**Special Features**:
- Rate limit handling
- Parallel batch optimization
- Speed metrics

```python
# src/swebench_runner/providers/groq.py
class GroqProvider(ModelProvider):
    name = "groq"
    description = "Ultra-fast inference with Groq"

    # Groq-specific rate limits
    rate_limits = {
        "mixtral-8x7b-32768": {"rpm": 30, "tpm": 30000},
        "llama3-70b-8192": {"rpm": 30, "tpm": 15000}
    }

    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        # Include rate limit handling
        await self._check_rate_limit()

        start_time = time.time()
        response = await self._make_request(prompt, **kwargs)
        latency_ms = int((time.time() - start_time) * 1000)

        return ModelResponse(
            content=response["choices"][0]["message"]["content"],
            model=response["model"],
            latency_ms=latency_ms,
            usage=response["usage"]
        )
```

### Phase 5: Google Vertex AI Provider

**Objective**: Enterprise Google Cloud integration.

**Features**:
- Automatic credential detection
- Region optimization
- Project management

```python
# src/swebench_runner/providers/vertex.py
class VertexAIProvider(ModelProvider):
    name = "vertex"
    description = "Google Vertex AI (Gemini, PaLM)"
    requires_api_key = False  # Uses gcloud auth

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        # Initialize Vertex AI SDK
        import vertexai
        vertexai.init(
            project=config.extra_params.get("project"),
            location=config.extra_params.get("region", "us-central1")
        )

    @classmethod
    def interactive_setup(cls) -> ProviderConfig:
        # Auto-detect GCP project
        try:
            import google.auth
            _, project = google.auth.default()
            detected_project = click.prompt("GCP Project", default=project)
        except:
            detected_project = click.prompt("GCP Project")

        region = click.prompt("Region", default="us-central1")

        models = ["gemini-pro", "gemini-ultra", "code-bison"]
        model = click.prompt("Model", type=click.Choice(models))

        return ProviderConfig(
            name="vertex",
            model=model,
            extra_params={"project": detected_project, "region": region}
        )
```

### Phase 6: HuggingFace Provider (Hub & Inference)

**Objective**: Support HuggingFace ecosystem.

**Features**:
- Serverless Inference API
- Dedicated Inference Endpoints
- Spaces support
- Model search/discovery



## Security Considerations

1. **API Key Storage**:
   - Use `keyring` library for secure credential storage
   - Fall back to environment variables
   - Never store in plain text files

2. **Secure Communication**:
   - All providers must use HTTPS
   - Validate SSL certificates
   - Set reasonable timeouts

3. **Input Validation**:
   ```python
   class SecurityValidator:
       """Validates provider inputs and outputs for security."""

       MAX_PROMPT_SIZE = 1024 * 1024  # 1MB
       DANGEROUS_PATTERNS = [
           r'eval\s*\(',
           r'exec\s*\(',
           r'__import__',
           r'subprocess',
           r'os\.system'
       ]

       def validate_prompt(self, prompt: str) -> None:
           """Validate prompt before sending to provider."""
           if len(prompt) > self.MAX_PROMPT_SIZE:
               raise ValueError("Prompt too large")

           # Check for potential injection attempts
           for pattern in self.DANGEROUS_PATTERNS:
               if re.search(pattern, prompt, re.IGNORECASE):
                   logger.warning(f"Potentially dangerous pattern in prompt: {pattern}")

       def validate_response(self, response: str) -> None:
           """Validate model response for safety."""
           # Check response isn't trying to execute code
           if "```python" in response and any(
               pattern in response for pattern in self.DANGEROUS_PATTERNS
           ):
               logger.warning("Model response contains potentially dangerous code")
   ```

4. **Credential Security**:
   ```python
   import keyring
   from cryptography.fernet import Fernet

   class CredentialManager:
       """Securely manages API credentials."""

       SERVICE_NAME = "swebench-runner"

       def store_api_key(self, provider: str, api_key: str) -> None:
           """Store API key securely."""
           try:
               keyring.set_password(self.SERVICE_NAME, provider, api_key)
           except keyring.errors.NoKeyringError:
               # Fall back to encrypted file storage
               self._store_encrypted(provider, api_key)

       def get_api_key(self, provider: str) -> Optional[str]:
           """Retrieve API key, checking multiple sources."""
           # 1. Environment variable
           env_var = f"{provider.upper()}_API_KEY"
           if api_key := os.getenv(env_var):
               return api_key

           # 2. Keyring
           try:
               if api_key := keyring.get_password(self.SERVICE_NAME, provider):
                   return api_key
           except keyring.errors.NoKeyringError:
               pass

           # 3. Encrypted file
           return self._get_encrypted(provider)
   ```

5. **Rate Limiting Protection**:
   - Implement per-provider rate limits
   - Exponential backoff on failures
   - Prevent DOS through batch size limits

## Documentation Requirements

1. **Provider Documentation**: Each provider needs:
   - Setup guide
   - Supported models list
   - Cost information
   - Rate limits
   - Example usage

2. **Plugin Development Guide**: How to add new providers

3. **Migration Guide**: For users with existing scripts


## Notes

This architecture makes SWE-bench truly accessible. Users focus on their models, not integration code. The magic is that it "just works" with any model, anywhere.

## Critical Review & Updated Implementation Requirements

### Major Changes from Original Plan (2025-01-25)

After reviewing the actual codebase and identifying untested assumptions:

1. **Dataset Schema Correction**:
   - ❌ Original assumed: `FAIL_TO_PASS`, `hints_text` fields
   - ✅ Actual fields: `instance_id`, `repo`, `base_commit`, `problem_statement`, `patch`

2. **Integration Points Fixed**:
   - ❌ Original: Non-existent `load_dataset_instances()` function
   - ✅ Actual: Use `DatasetManager.get_instances()` from existing codebase

3. **Async/Sync Handling**:
   - ❌ Original: Risky event loop detection
   - ✅ Fixed: Thread pool executor with dedicated event loop

4. **Token Management**:
   - ❌ Original: Hardcoded model limits
   - ✅ Fixed: Dynamic querying of provider APIs

5. **Security Enhancements**:
   - ✅ Added: Input validation, credential manager, security validator
   - ✅ Added: Keyring integration with environment variable fallback

6. **Response Parsing**:
   - ✅ Enhanced: Multiple extraction strategies for various model outputs
   - ✅ Added: Edit instruction parsing and conversion

### Research Required Before Implementation

Following the CLAUDE.md guidance on research-first methodology:

1. **Test Actual Model Outputs** (15 min):
   ```bash
   # Test what patches actually look like from different models
   curl -X POST https://api.openai.com/v1/chat/completions \
     -H "Authorization: Bearer $OPENAI_API_KEY" \
     -d '{"model": "gpt-4", "messages": [{"role": "user", "content": "Generate a git patch..."}]}'
   ```

2. **Verify Dataset Schema** (5 min):
   ```python
   # Quick test to confirm actual fields
   from datasets import load_dataset
   ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
   print(ds[0].keys())  # Verify actual fields
   ```

3. **Test Harness Integration** (10 min):
   ```bash
   # Verify harness command structure
   python -m swebench.harness.run_evaluation --help
   ```

4. **Platform Compatibility** (10 min):
   - Test keyring on Windows/Linux/macOS
   - Verify async behavior across platforms

## Detailed Implementation Plan

### Phase 1: Core Infrastructure (⚠️ Partially Complete - 2025-01-25)

#### 1.1 Completed Components ✅
- Created `src/swebench_runner/providers/` directory structure
- Implemented robust base classes with proper validation
- Built comprehensive exception hierarchy
- Created provider registry with auto-discovery
- Implemented circuit breaker with state management
- Created async/sync bridge with singleton pattern
- Built multi-source configuration management
- Added provider lifecycle basics

#### 1.2 Missing Critical Components ❌
Based on review, the following are missing or need improvement:

1. **MockProvider** - Essential for testing but not implemented
2. **CircuitBreakerProvider Wrapper** - Circuit breaker exists but no provider wrapper
3. **SyncProviderWrapper** - Clean sync interface for CLI missing
4. **Complete __init__.py** - Missing exports for public API
5. **Provider Examples** - No reference implementations
6. **Integration Hooks** - No connection to existing CLI
7. **estimate_cost() guidance** - Abstract method but no implementation pattern

### Phase 1a: Infrastructure Completion (Priority: HIGH)

**Objective**: Complete missing core infrastructure components

#### MockProvider Implementation
```python
# src/swebench_runner/providers/mock.py
from typing import Dict, Any, Optional
import asyncio
from .base import ModelProvider, ProviderConfig, ModelResponse, ProviderCapabilities
from .registry import register_provider

@register_provider
class MockProvider(ModelProvider):
    """Mock provider for testing and development."""

    name = "mock"
    description = "Mock provider for testing"
    api_version = "1.0"
    requires_api_key = False
    supported_models = ["mock-fast", "mock-slow", "mock-error"]
    default_model = "mock-fast"

    def __init__(self, config: ProviderConfig):
        # Don't validate API key for mock
        self.config = config
        self.capabilities = self._init_capabilities()
        self._initialized = True
        self._health_status = "healthy"
        self.call_count = 0
        self.responses = config.extra_params.get("responses", {})
        self.delay = config.extra_params.get("delay", 0.1)
        self.should_fail = config.extra_params.get("should_fail", False)

    def _init_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            max_context_length=100000,
            supports_streaming=False,
            supports_json_mode=True,
            rate_limits={"rpm": 1000, "tpm": 1000000},
            supported_models=self.supported_models,
            cost_per_1k_prompt_tokens=0.0001,
            cost_per_1k_completion_tokens=0.0002
        )

    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        await asyncio.sleep(self.delay)  # Simulate API latency
        self.call_count += 1

        # Simulate errors if configured
        if self.should_fail or self.config.model == "mock-error":
            if "rate_limit" in str(self.config.extra_params.get("error_type", "")):
                from .exceptions import ProviderRateLimitError
                raise ProviderRateLimitError("Mock rate limit exceeded", retry_after=60)
            else:
                from .exceptions import ProviderError
                raise ProviderError("Mock error for testing")

        # Return configured response or default
        if self.responses:
            content = self.responses.get(str(self.call_count), "Default mock response")
        else:
            content = f"Mock response to: {prompt[:50]}..."

        return ModelResponse(
            content=content,
            model=self.config.model or self.default_model,
            usage={"prompt_tokens": len(prompt.split()), "completion_tokens": len(content.split())},
            cost=0.001,
            latency_ms=int(self.delay * 1000),
            provider=self.name
        )

    def estimate_cost(self, prompt_tokens: int, max_tokens: int) -> float:
        input_cost = (prompt_tokens / 1000) * self.capabilities.cost_per_1k_prompt_tokens
        output_cost = (max_tokens / 1000) * self.capabilities.cost_per_1k_completion_tokens
        return round(input_cost + output_cost, 6)

    @classmethod
    def _config_from_env(cls, env_vars: Dict[str, str], model: Optional[str] = None) -> ProviderConfig:
        return ProviderConfig(
            name=cls.name,
            model=model or cls.default_model,
            api_key=None  # Not required for mock
        )
```

#### CircuitBreakerProvider Wrapper
```python
# src/swebench_runner/providers/circuit_breaker_provider.py
from typing import Any, Dict, Optional
from .base import ModelProvider, ModelResponse, ProviderCapabilities
from .circuit_breaker import CircuitBreaker

class CircuitBreakerProvider(ModelProvider):
    """Wraps any provider with circuit breaker protection."""

    def __init__(self, provider: ModelProvider, circuit_breaker: Optional[CircuitBreaker] = None):
        self._wrapped_provider = provider
        self.circuit_breaker = circuit_breaker or CircuitBreaker(
            name=f"{provider.name}_circuit",
            failure_threshold=5,
            recovery_timeout=60.0
        )

        # Delegate all attributes to wrapped provider
        self.name = provider.name
        self.description = f"{provider.description} (with circuit breaker)"
        self.api_version = provider.api_version
        self.requires_api_key = provider.requires_api_key
        self.supported_models = provider.supported_models
        self.default_model = provider.default_model
        self.config = provider.config
        self.capabilities = provider.capabilities
        self._initialized = provider._initialized
        self._health_status = provider._health_status

    def _init_capabilities(self) -> ProviderCapabilities:
        # Not used - we delegate to wrapped provider
        return self._wrapped_provider.capabilities

    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate with circuit breaker protection."""
        return await self.circuit_breaker.call(
            self._wrapped_provider.generate,
            prompt,
            **kwargs
        )

    async def validate_connection(self) -> bool:
        """Validate connection through circuit breaker."""
        try:
            return await self.circuit_breaker.call(
                self._wrapped_provider.validate_connection
            )
        except Exception:
            return False

    def estimate_cost(self, prompt_tokens: int, max_tokens: int) -> float:
        return self._wrapped_provider.estimate_cost(prompt_tokens, max_tokens)

    @classmethod
    def _config_from_env(cls, env_vars: Dict[str, str], model: Optional[str] = None) -> ProviderConfig:
        # Not used - we wrap existing providers
        raise NotImplementedError("CircuitBreakerProvider wraps existing providers")

    def get_circuit_state(self) -> str:
        """Get current circuit breaker state."""
        return self.circuit_breaker.state.value

    def reset_circuit(self):
        """Reset circuit breaker to closed state."""
        self.circuit_breaker.reset()
```

#### SyncProviderWrapper
```python
# src/swebench_runner/providers/sync_wrapper.py
from typing import Any, Dict, Optional
from .base import ModelProvider, ModelResponse
from .async_bridge import AsyncBridge

class SyncProviderWrapper:
    """Provides synchronous interface to async providers for CLI integration."""

    def __init__(self, provider: ModelProvider):
        self.provider = provider
        self.bridge = AsyncBridge()

        # Expose provider attributes
        self.name = provider.name
        self.config = provider.config
        self.capabilities = provider.capabilities

    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Synchronous generation method."""
        return self.bridge.run(
            self.provider.generate(prompt, **kwargs),
            timeout=kwargs.get('timeout', self.provider.config.timeout)
        )

    def validate_connection(self) -> bool:
        """Synchronous connection validation."""
        return self.bridge.run(
            self.provider.validate_connection(),
            timeout=30
        )

    def estimate_cost(self, prompt_tokens: int, max_tokens: int) -> float:
        """Estimate cost (already sync)."""
        return self.provider.estimate_cost(prompt_tokens, max_tokens)

    def health_check(self) -> Dict[str, Any]:
        """Synchronous health check."""
        return self.bridge.run(
            self.provider.health_check(),
            timeout=10
        )
```

#### Updated __init__.py
```python
# src/swebench_runner/providers/__init__.py
"""Provider system for model integrations."""

# Core components
from .base import (
    ModelProvider,
    ModelResponse,
    ProviderConfig,
    ProviderCapabilities,
)
from .registry import (
    ProviderRegistry,
    get_registry,
    register_provider,
)
from .exceptions import (
    ProviderError,
    ProviderNotFoundError,
    ProviderConnectionError,
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderConfigurationError,
    ProviderTimeoutError,
    ProviderTokenLimitError,
    CircuitBreakerError,
)
from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
)
from .async_bridge import (
    AsyncBridge,
    run_async,
    async_to_sync,
)
from .config import ProviderConfigManager

# Wrappers
from .circuit_breaker_provider import CircuitBreakerProvider
from .sync_wrapper import SyncProviderWrapper

# Built-in providers
from .mock import MockProvider

__all__ = [
    # Core
    "ModelProvider",
    "ModelResponse",
    "ProviderConfig",
    "ProviderCapabilities",
    # Registry
    "ProviderRegistry",
    "get_registry",
    "register_provider",
    # Exceptions
    "ProviderError",
    "ProviderNotFoundError",
    "ProviderConnectionError",
    "ProviderAuthenticationError",
    "ProviderRateLimitError",
    "ProviderResponseError",
    "ProviderConfigurationError",
    "ProviderTimeoutError",
    "ProviderTokenLimitError",
    "CircuitBreakerError",
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitState",
    "CircuitBreakerProvider",
    # Async Bridge
    "AsyncBridge",
    "run_async",
    "async_to_sync",
    # Config
    "ProviderConfigManager",
    # Wrappers
    "SyncProviderWrapper",
    # Providers
    "MockProvider",
]

__version__ = "0.1.0"
```

### Phase 1b: Integration with Existing CLI

**Objective**: Connect provider system to existing SWE-bench runner

#### CLI Integration Points
1. Modify existing commands to accept `--provider` flag
2. Add provider management commands
3. Create initialization wizard

#### Example Integration
```python
# Modification to existing CLI commands
# src/swebench_runner/cli.py (additions)

@cli.group()
def provider():
    """Manage model providers."""
    pass

@provider.command()
def list():
    """List available providers."""
    registry = get_registry()
    providers = registry.list_providers()

    # Display in table format
    from rich.table import Table
    from rich.console import Console

    console = Console()
    table = Table(title="Available Providers")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    table.add_column("Configured", style="green")
    table.add_column("API Key Required")

    for p in providers:
        configured = "✓" if p["configured"] else "✗"
        api_key = "Yes" if p["requires_api_key"] else "No"
        table.add_row(p["name"], p["description"], configured, api_key)

    console.print(table)

@provider.command()
@click.argument('provider_name')
def init(provider_name):
    """Initialize a provider configuration."""
    registry = get_registry()
    config_manager = ProviderConfigManager()

    try:
        provider_class = registry.get_provider_class(provider_name)
    except ProviderNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        return

    # Interactive setup
    click.echo(f"Setting up {provider_name} provider...")

    if provider_class.requires_api_key:
        api_key = click.prompt("API Key", hide_input=True)
    else:
        api_key = None

    model = click.prompt(
        "Model",
        default=provider_class.default_model or provider_class.supported_models[0]
        if provider_class.supported_models else None
    )

    config = ProviderConfig(
        name=provider_name,
        api_key=api_key,
        model=model
    )

    # Test connection
    click.echo("Testing connection...")
    try:
        provider = provider_class(config)
        sync_wrapper = SyncProviderWrapper(provider)
        if sync_wrapper.validate_connection():
            click.echo("✓ Connection successful!")
            config_manager.save_config(config)
            click.echo(f"Configuration saved for {provider_name}")
        else:
            click.echo("✗ Connection failed. Please check your settings.", err=True)
    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)

@provider.command()
@click.argument('provider_name')
@click.option('--prompt', '-p', default='Say "Hello, World!"')
def test(provider_name, prompt):
    """Test a provider with a simple prompt."""
    registry = get_registry()

    try:
        provider = registry.get_provider(provider_name)
        sync_wrapper = SyncProviderWrapper(provider)

        click.echo(f"Testing {provider_name} with prompt: {prompt}")
        response = sync_wrapper.generate(prompt, max_tokens=50)

        click.echo(f"\nResponse: {response.content}")
        click.echo(f"Model: {response.model}")
        if response.cost:
            click.echo(f"Cost: ${response.cost:.4f}")
        if response.latency_ms:
            click.echo(f"Latency: {response.latency_ms}ms")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
```

#### 1.2 Original Required Components (Moved to Phase 2)
```python
# src/swebench_runner/generate.py
class PatchGenerator:
    def __init__(self, provider: ModelProvider):
        self.provider = provider
        self.prompt_builder = PromptBuilder()
        self.response_parser = ResponseParser()
        self.token_manager = TokenManager()
        self.validator = PatchValidator()

    async def generate_patch(self, instance: Dict[str, Any]) -> str:
        # Build context from SWE-bench instance
        context = self.prompt_builder.build_context(instance)

        # Manage tokens for model limits
        prompt, truncated = self.token_manager.fit_context(
            context,
            self.provider.config.model,
            self.prompt_template
        )

        if truncated:
            logger.warning(f"Context truncated for {instance['instance_id']}")

        # Generate with retry logic
        for attempt in range(3):
            try:
                response = await self.provider.generate(prompt)
                patch = self.response_parser.extract_patch(response.content)

                if patch and self.validator.validate(patch):
                    return patch

                # Retry with clarification
                prompt = self.prompt_builder.build_retry_prompt(response.content)
            except ProviderTokenLimitError:
                # Reduce context and retry
                context = self.prompt_builder.reduce_context(context)
                prompt, _ = self.token_manager.fit_context(context, model, template)
            except ProviderRateLimitError as e:
                await asyncio.sleep(e.retry_after or 60)

        raise GenerationError(f"Failed to generate valid patch for {instance['instance_id']}")
```

**Synchronous Wrapper** (For Integration):
```python
# src/swebench_runner/providers/sync_wrapper.py
import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

class SyncProviderWrapper:
    """Wraps async providers for sync codebase using thread pool."""

    _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="provider")
    _loop = None
    _lock = Lock()

    def __init__(self, async_provider: ModelProvider):
        self.async_provider = async_provider
        self._ensure_event_loop()

    @classmethod
    def _ensure_event_loop(cls):
        """Ensure we have a dedicated event loop for async operations."""
        with cls._lock:
            if cls._loop is None:
                # Create event loop in executor thread
                future = cls._executor.submit(asyncio.new_event_loop)
                cls._loop = future.result()

                # Run the loop in the executor
                cls._executor.submit(cls._loop.run_forever)

    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Synchronous generation method using thread-safe async execution."""
        future = asyncio.run_coroutine_threadsafe(
            self.async_provider.generate(prompt, **kwargs),
            self._loop
        )
        return future.result(timeout=300)  # 5 minute timeout

    def validate_connection(self) -> bool:
        """Synchronous connection validation."""
        future = asyncio.run_coroutine_threadsafe(
            self.async_provider.validate_connection(),
            self._loop
        )
        return future.result(timeout=30)
```

**Token Management System**:
```python
# src/swebench_runner/providers/tokens.py
import aiohttp
from functools import lru_cache
from typing import Dict, Optional

class TokenManager:
    """Manages token counting and limits with dynamic provider queries."""

    # Fallback limits if API queries fail
    DEFAULT_LIMITS = {
        "gpt-4": 8192,
        "gpt-3.5-turbo": 16384,
        "claude-3": 200000,
        "default": 4096
    }

    def __init__(self):
        self._init_tokenizers()
        self._token_cache = {}
        self._limit_cache = {}
        self._provider_clients = {}

    async def get_model_limit(self, provider: str, model: str) -> int:
        """Get token limit for model, querying provider if needed."""
        cache_key = f"{provider}:{model}"

        if cache_key in self._limit_cache:
            return self._limit_cache[cache_key]

        # Try to query provider for actual limits
        try:
            if provider == "openai":
                limit = await self._query_openai_limits(model)
            elif provider == "anthropic":
                limit = await self._query_anthropic_limits(model)
            else:
                limit = None

            if limit:
                self._limit_cache[cache_key] = limit
                return limit
        except Exception as e:
            logger.warning(f"Failed to query {provider} for model limits: {e}")

        # Fallback to defaults
        for key, default_limit in self.DEFAULT_LIMITS.items():
            if key in model.lower():
                return default_limit

        return self.DEFAULT_LIMITS["default"]

    async def _query_openai_limits(self, model: str) -> Optional[int]:
        """Query OpenAI API for model context length."""
        # OpenAI models endpoint provides this info
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    for model_info in data["data"]:
                        if model_info["id"] == model:
                            return model_info.get("context_length", None)
        return None

    @lru_cache(maxsize=1000)
    def count_tokens(self, text: str, model: str) -> int:
        """Count tokens with caching."""
        # Use tiktoken for OpenAI models
        if "gpt" in model.lower() or "davinci" in model.lower():
            import tiktoken
            try:
                encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))

        # Rough approximation for other models
        # Most models average ~4 characters per token
        return len(text) // 4
```

### Phase 2: Error Handling & Recovery

**Batch Processor with Checkpoints**:
```python
# src/swebench_runner/providers/batch.py
class BatchProcessor:
    def __init__(self, provider: ModelProvider, storage_manager: StorageManager):
        self.provider = provider
        self.storage = storage_manager
        self.error_classifier = ErrorClassifier()

    async def process_batch(
        self,
        instances: List[Dict],
        generator: PatchGenerator,
        checkpoint_dir: Path
    ) -> BatchResult:
        # Load checkpoint if exists
        checkpoint = self.load_checkpoint(checkpoint_dir)
        completed_ids = checkpoint.get("completed", set())

        # Filter already completed
        remaining = [i for i in instances if i["instance_id"] not in completed_ids]

        results = BatchResult()
        with tqdm(total=len(remaining)) as pbar:
            for instance in remaining:
                result = await self._process_with_recovery(instance, generator)

                if result.success:
                    results.successful.append(result)
                    completed_ids.add(instance["instance_id"])
                    self.save_checkpoint(checkpoint_dir, completed_ids)
                else:
                    results.failed.append(result)

                pbar.update(1)

        return results
```

### Phase 3: Response Parsing

**Robust Patch Extraction**:
```python
# src/swebench_runner/providers/parser.py
class ResponseParser:
    """Robust parser for extracting patches from various model outputs."""

    def extract_patch(self, response: str) -> Optional[str]:
        """Extract patch using multiple strategies."""
        strategies = [
            self._extract_fenced_code_block,
            self._extract_diff_block,
            self._extract_inline_diff,
            self._extract_file_blocks,
            self._extract_edit_instructions,
            self._extract_replacement_blocks,
        ]

        for strategy in strategies:
            patch = strategy(response)
            if patch and self._looks_like_patch(patch):
                return self._normalize_patch(patch)

        # Last resort: try to convert edit instructions to patch
        if edits := self._parse_edit_instructions(response):
            return self._convert_edits_to_patch(edits)

        return None

    def _extract_fenced_code_block(self, response: str) -> Optional[str]:
        """Extract from markdown code blocks."""
        # Handle ```diff, ```patch, or plain ```
        patterns = [
            r'```(?:diff|patch)\n(.*?)\n```',
            r'```\n((?:diff --git|---|\+\+\+|@@).*?)\n```',
            r'<patch>\n(.*?)\n</patch>',  # XML-style tags
        ]

        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                if self._looks_like_patch(match):
                    return match

        return None

    def _extract_edit_instructions(self, response: str) -> Optional[List[Dict]]:
        """Extract structured edit instructions."""
        # Pattern: "In file.py, change line X from 'old' to 'new'"
        edit_pattern = r'In\s+(\S+),\s+(?:change|replace).*?from[:\s]+[\'"`]?(.*?)[\'"`]?\s+to[:\s]+[\'"`]?(.*?)[\'"`]?'

        edits = []
        for match in re.finditer(edit_pattern, response, re.IGNORECASE | re.DOTALL):
            edits.append({
                'file': match.group(1),
                'old': match.group(2).strip(),
                'new': match.group(3).strip()
            })

        return edits if edits else None

    def _extract_replacement_blocks(self, response: str) -> Optional[str]:
        """Extract from replacement-style blocks."""
        # Pattern: <<< old content >>> new content
        pattern = r'<<<\s*(.*?)\s*>>>\s*(.*?)\s*(?=<<<|$)'

        patches = []
        for match in re.finditer(pattern, response, re.DOTALL):
            old_content = match.group(1).strip()
            new_content = match.group(2).strip()

            # Try to infer file from content
            if file_hint := self._infer_file_from_content(old_content):
                patch = self._create_simple_patch(file_hint, old_content, new_content)
                patches.append(patch)

        return '\n'.join(patches) if patches else None

    def _convert_edits_to_patch(self, edits: List[Dict]) -> str:
        """Convert edit instructions to unified diff format."""
        patches = []

        for edit in edits:
            file_path = edit['file']
            old_lines = edit['old'].splitlines(keepends=True)
            new_lines = edit['new'].splitlines(keepends=True)

            # Create a simple unified diff
            patch = f"diff --git a/{file_path} b/{file_path}\n"
            patch += f"--- a/{file_path}\n"
            patch += f"+++ b/{file_path}\n"
            patch += "@@ -1,{len(old_lines)} +1,{len(new_lines)} @@\n"

            for line in old_lines:
                patch += f"-{line}"
            for line in new_lines:
                patch += f"+{line}"

            patches.append(patch)

        return '\n'.join(patches)

    def _looks_like_patch(self, text: str) -> bool:
        """Check if text appears to be a valid patch."""
        indicators = [
            'diff --git',
            '--- a/',
            '+++ b/',
            '@@',
            'Index:',
            '====='
        ]

        return any(indicator in text for indicator in indicators)
```

### Phase 4: Integration with Existing System

**Harness Integration**:
```python
# src/swebench_runner/providers/integration.py
from pathlib import Path
from typing import List, Dict, Any
import json
import threading

class HarnessIntegration:
    """Integrates generated patches with the SWE-bench harness."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def save_predictions_for_harness(
        self,
        patches: List[Dict[str, Any]],
        run_id: str
    ) -> Path:
        """Save patches in format expected by SWE-bench harness.

        The harness expects a JSONL file with format:
        {"instance_id": "...", "model": "...", "patch": "..."}
        """
        predictions_file = self.output_dir / f"predictions_{run_id}.jsonl"

        with self._lock:  # Thread-safe file writing
            with open(predictions_file, 'w') as f:
                for patch_data in patches:
                    # Format for harness compatibility
                    prediction = {
                        "instance_id": patch_data["instance_id"],
                        "model": patch_data.get("model", "unknown"),
                        "patch": patch_data["patch"]  # Note: harness expects "patch" not "prediction"
                    }
                    json.dump(prediction, f)
                    f.write('\n')

        return predictions_file

    def create_harness_command(
        self,
        predictions_file: Path,
        dataset_name: str,
        output_dir: Path,
        limit: Optional[int] = None,
        instance_ids: Optional[List[str]] = None
    ) -> List[str]:
        """Create command to run harness evaluation."""
        cmd = [
            sys.executable, "-m", "swebench.harness.run_evaluation",
            "--predictions_path", str(predictions_file),
            "--dataset_name", dataset_name,
            "--run_id", output_dir.name,
            "--output_dir", str(output_dir)
        ]

        if limit:
            cmd.extend(["--limit", str(limit)])

        if instance_ids:
            cmd.extend(["--instance_ids"] + instance_ids)

        return cmd
```

### Phase 5: CLI Integration

**Generate Command**:
```python
# src/swebench_runner/cli.py additions
@cli.command()
@click.option('--provider', '-p', required=True, help='Model provider')
@click.option('--model', '-m', help='Specific model to use')
@click.option('--dataset', '-d', default='lite', help='Dataset to use')
@click.option('--instances', '-i', multiple=True, help='Specific instances')
@click.option('--count', '-c', type=int, help='Number of instances')
@click.option('--output', '-o', type=click.Path(), help='Output directory')
@click.option('--resume', type=click.Path(exists=True), help='Resume from checkpoint')
def generate(provider, model, dataset, instances, count, output, resume):
    """Generate patches using a model provider."""
    try:
        # Setup
        registry = get_registry()
        provider_instance = registry.get_provider(provider)

        if model:
            provider_instance.config.model = model

        # Load instances using actual DatasetManager
        from swebench_runner.datasets import DatasetManager
        from pathlib import Path

        cache_dir = Path.home() / ".swebench"
        dataset_manager = DatasetManager(cache_dir)

        instance_list = dataset_manager.get_instances(
            dataset_name=dataset,
            instances=list(instances) if instances else None,
            limit=count
        )

        # Generate patches
        generator = PatchGenerator(provider_instance)
        processor = BatchProcessor(provider_instance)

        # Run generation
        results = asyncio.run(
            processor.process_batch(
                instance_list,
                generator,
                checkpoint_dir=Path(resume) if resume else None
            )
        )

        # Save results
        output_dir = Path(output or f"./patches_{provider}_{dataset}")
        output_dir.mkdir(parents=True, exist_ok=True)

        integration = HarnessIntegration(output_dir)

        # Convert results to format expected by harness
        patches_for_harness = []
        for result in results.successful:
            patches_for_harness.append({
                "instance_id": result.instance_id,
                "model": f"{provider}/{model or provider_instance.config.model}",
                "patch": result.patch
            })

        predictions_file = integration.save_predictions_for_harness(
            patches_for_harness,
            run_id=f"{provider}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Display summary
        click.echo(f"\n✅ Generated {len(results.successful)} patches")
        click.echo(f"📁 Saved to: {predictions_file}")

        if results.failed:
            click.echo(f"⚠️  Failed: {len(results.failed)} instances")

    except Exception as e:
        logger.exception("Generation failed")
        raise click.ClickException(str(e))
```

### Phase 6: Configuration Management

**Environment Variable Support**:
```python
# src/swebench_runner/config.py
class ProviderConfigManager:
    """Manages provider configurations from multiple sources."""

    ENV_MAPPING = {
        "openai": {
            "api_key": "OPENAI_API_KEY",
            "org_id": "OPENAI_ORG_ID",
            "base_url": "OPENAI_BASE_URL",
        },
        "anthropic": {
            "api_key": "ANTHROPIC_API_KEY",
        },
        # ... more providers
    }

    def load_config(self, provider_name: str) -> ProviderConfig:
        # 1. Check environment variables
        config = self._load_from_env(provider_name)

        # 2. Check config file
        if not config:
            config = self._load_from_file(provider_name)

        # 3. Check system keyring
        if not config:
            config = self._load_from_keyring(provider_name)

        if not config:
            raise ProviderConfigurationError(
                f"No configuration found for {provider_name}. "
                f"Set {self.ENV_MAPPING[provider_name]['api_key']} or run 'swebench init'"
            )

        return config
```

### Phase 7: Testing Strategy

**Mock Provider**:
```python
# src/swebench_runner/providers/mock.py
@register_provider
class MockProvider(ModelProvider):
    name = "mock"
    description = "Mock provider for testing"
    requires_api_key = False

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.responses = config.extra_params.get("responses", {})
        self.call_count = 0

    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        self.call_count += 1

        # Simulate latency
        await asyncio.sleep(0.1)

        # Return predetermined response
        if "error" in self.config.extra_params:
            raise ProviderError(self.config.extra_params["error"])

        content = self.responses.get(
            str(self.call_count),
            f"Mock patch for prompt {self.call_count}"
        )

        return ModelResponse(
            content=content,
            model="mock-model",
            usage={"prompt_tokens": 100, "completion_tokens": 200},
            cost=0.01,
            latency_ms=100
        )
```

## V1 Deliverables

### Core Infrastructure
- [x] Base provider system with capabilities
- [ ] Thread-safe registry with validation
- [ ] Circuit breaker implementation
- [ ] Async/sync bridge
- [ ] Multi-source configuration management

### Providers
- [ ] OpenAI provider with full error handling
- [ ] OpenRouter provider with model discovery
- [ ] Mock provider for testing

### Integration
- [ ] PatchGenerator with retry logic
- [ ] Response parser with multiple strategies
- [ ] CLI generate command
- [ ] Harness integration

### Quality
- [ ] Unit tests with mocked providers
- [ ] Integration tests with real APIs
- [ ] End-to-end test suite
- [ ] Performance benchmarks


## Missing Class Definitions

### PromptBuilder Class

```python
# src/swebench_runner/providers/prompts.py
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class PromptBuilder:
    """Builds and manages prompts for patch generation."""

    DEFAULT_TEMPLATE = """You are a software engineer fixing a bug in {repo}.

Issue: {instance_id}
Repository: {repo}
Base Commit: {base_commit}

Problem Statement:
{problem_statement}

Please provide a patch in unified diff format that fixes this issue. The patch should:
1. Be minimal and focused only on fixing the described issue
2. Follow the coding style of the repository
3. Not break existing functionality

Provide only the patch in unified diff format, starting with 'diff --git'."""

    def __init__(self):
        self.templates = {
            "default": self.DEFAULT_TEMPLATE,
            "claude": self._load_claude_template(),
            "gpt4": self._load_gpt4_template(),
        }

    def build_context(self, instance: Dict[str, Any]) -> str:
        """Build context from SWE-bench instance.

        Note: SWE-bench instances contain these fields:
        - instance_id: Unique identifier
        - repo: Repository name
        - base_commit: Git commit hash
        - problem_statement: Description of the issue
        - patch: The reference patch (not used for generation)
        """
        # Build prompt from actual SWE-bench fields
        return self.DEFAULT_TEMPLATE.format(
            repo=instance["repo"],
            instance_id=instance["instance_id"],
            base_commit=instance["base_commit"],
            problem_statement=instance["problem_statement"],
            hints_section="",  # Not present in actual dataset
            test_info_section=""  # Not present in actual dataset
        )

    def get_default_template(self) -> str:
        return self.DEFAULT_TEMPLATE

    def reduce_context(self, context: str) -> str:
        """Reduce context size while preserving key information."""
        lines = context.split('\n')

        # Keep header
        header_end = next(i for i, line in enumerate(lines) if "Problem Statement:" in line)
        reduced = lines[:header_end + 1]

        # Truncate problem statement
        problem_lines = []
        in_problem = False
        for line in lines[header_end:]:
            if "Problem Statement:" in line:
                in_problem = True
            elif line.strip() and not line.startswith(' ') and in_problem:
                break
            if in_problem:
                problem_lines.append(line)

        # Keep first 1000 chars of problem
        problem_text = '\n'.join(problem_lines)
        if len(problem_text) > 1000:
            problem_text = problem_text[:1000] + "\n[... truncated for length ...]"

        reduced.extend(problem_text.split('\n'))
        reduced.append("\nPlease provide a minimal patch to fix this issue.")

        return '\n'.join(reduced)

    def build_retry_prompt(self, previous_response: str) -> str:
        """Build retry prompt for invalid responses."""
        return f"""Your previous response did not contain a valid patch.

Previous response:
{previous_response[:500]}...

Please provide a valid patch in unified diff format. The patch must:
1. Start with 'diff --git a/... b/...'
2. Include --- and +++ file markers
3. Include @@ hunk headers
4. Show the changes with - and + prefixes

Example format:
diff --git a/file.py b/file.py
--- a/file.py
+++ b/file.py
@@ -10,7 +10,7 @@
 def function():
-    old_line
+    new_line

Please provide only the patch, no explanations."""

    def _load_claude_template(self) -> str:
        """Load Claude-specific template."""
        return """Human: I need you to fix a bug in the {repo} repository.

<issue_details>
Instance ID: {instance_id}
Repository: {repo}
Base Commit: {base_commit}

Problem Description:
{problem_statement}

{hints_section}

{test_info_section}
</issue_details>

Please analyze this issue and provide a minimal patch that fixes the problem. Return only the patch in unified diff format.


    def _load_gpt4_template(self) -> str:
        """Load GPT-4 specific template."""
        return """Fix the following bug in {repo}:

Issue ID: {instance_id}
Base Commit: {base_commit}

{problem_statement}

{hints_section}

{test_info_section}

Provide a minimal unified diff patch that fixes this issue. Output only the patch, no explanations."""
```

### PatchValidator Class

```python
# src/swebench_runner/providers/validators.py
import re
from typing import Tuple, Optional

class PatchValidator:
    """Validates patches against SWE-bench requirements."""

    def __init__(self):
        self.max_patch_size = 1024 * 1024  # 1MB
        self.required_headers = ["diff --git", "---", "+++", "@@"]

    def validate(self, patch: str) -> bool:
        """Validate patch format and content."""
        if not patch or not patch.strip():
            return False

        # Check size
        if len(patch) > self.max_patch_size:
            return False

        # Check required headers
        for header in self.required_headers:
            if header not in patch:
                return False

        # Validate structure
        try:
            self._parse_patch_structure(patch)
            return True
        except Exception:
            return False

    def _parse_patch_structure(self, patch: str):
        """Parse and validate patch structure."""
        lines = patch.strip().split("
")

        # Must start with diff --git
        if not lines[0].startswith("diff --git"):
            raise ValueError("Patch must start with diff --git")

        # Check for file markers
        has_minus = any(line.startswith("---") for line in lines)
        has_plus = any(line.startswith("+++") for line in lines)

        if not (has_minus and has_plus):
            raise ValueError("Missing --- or +++ file markers")

        # Check for hunks
        has_hunks = any(line.startswith("@@") for line in lines)
        if not has_hunks:
            raise ValueError("No @@ hunk headers found")
```


### ErrorClassifier Class

```python
# src/swebench_runner/providers/errors.py
import re
import asyncio
import aiohttp
from enum import Enum
from typing import Dict, List

class ErrorType(Enum):
    RATE_LIMIT = "rate_limit"
    TOKEN_LIMIT = "token_limit"
    NETWORK_ERROR = "network_error"
    AUTH_ERROR = "auth_error"
    UNKNOWN = "unknown"

class ErrorClassifier:
    """Classifies errors from providers for appropriate handling."""

    ERROR_PATTERNS = {
        ErrorType.RATE_LIMIT: [
            r"rate.?limit",
            r"429",
            r"too many requests",
            r"quota exceeded"
        ],
        ErrorType.TOKEN_LIMIT: [
            r"context.?length",
            r"token.?limit",
            r"maximum.?context",
            r"too.?long"
        ],
        ErrorType.NETWORK_ERROR: [
            r"timeout",
            r"connection.?error",
            r"network",
            r"unreachable"
        ],
        ErrorType.AUTH_ERROR: [
            r"unauthorized",
            r"401",
            r"invalid.?key",
            r"authentication"
        ]
    }

    def classify(self, error: Exception) -> ErrorType:
        """Classify an error for appropriate handling."""
        error_str = str(error).lower()

        # Check patterns
        for error_type, patterns in self.ERROR_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, error_str, re.IGNORECASE):
                    return error_type

        # Check exception types
        if isinstance(error, asyncio.TimeoutError):
            return ErrorType.NETWORK_ERROR
        elif isinstance(error, aiohttp.ClientError):
            return ErrorType.NETWORK_ERROR

        return ErrorType.UNKNOWN

    def get_retry_delay(self, error_type: ErrorType, attempt: int) -> float:
        """Get appropriate retry delay for error type."""
        base_delays = {
            ErrorType.RATE_LIMIT: 60.0,
            ErrorType.NETWORK_ERROR: 2.0,
            ErrorType.TOKEN_LIMIT: 0.0,  # No retry
            ErrorType.AUTH_ERROR: 0.0,    # No retry
            ErrorType.UNKNOWN: 5.0
        }

        base = base_delays.get(error_type, 5.0)
        return base * (2 ** attempt)  # Exponential backoff
```

### StorageManager Class

```python
# src/swebench_runner/providers/storage.py
import json
from pathlib import Path
from typing import Dict, Any, Optional

class StorageManager:
    """Manages persistent storage for provider operations."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.checkpoints_dir = self.base_dir / "checkpoints"
        self.patches_dir = self.base_dir / "patches"
        self.logs_dir = self.base_dir / "logs"

        for dir in [self.checkpoints_dir, self.patches_dir, self.logs_dir]:
            dir.mkdir(exist_ok=True)

    def save_checkpoint(self, run_id: str, data: Dict[str, Any]):
        """Save checkpoint data."""
        checkpoint_file = self.checkpoints_dir / f"{run_id}.json"

        # Atomic write
        temp_file = checkpoint_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(data, f, indent=2)
        temp_file.replace(checkpoint_file)

    def load_checkpoint(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint data."""
        checkpoint_file = self.checkpoints_dir / f"{run_id}.json"
        if checkpoint_file.exists():
            with open(checkpoint_file) as f:
                return json.load(f)
        return None

    def save_patch(self, instance_id: str, patch: str):
        """Save individual patch."""
        patch_file = self.patches_dir / f"{instance_id}.patch"
        patch_file.write_text(patch)

    def get_run_log_path(self, run_id: str) -> Path:
        """Get path for run logs."""
        return self.logs_dir / f"{run_id}.log"
```

### GenerationError Class

```python
# src/swebench_runner/providers/exceptions.py (addition)
class GenerationError(ProviderError):
    """Raised when patch generation fails after all retries."""
    pass
```


## Dependencies

### Required Python Packages

Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
providers = [
    # API Clients
    "anthropic>=0.18.0",
    "openai>=1.12.0",
    "google-cloud-aiplatform>=1.38.0",  # For Vertex AI
    "boto3>=1.28.0",  # For AWS Bedrock

    # Async HTTP
    "aiohttp>=3.9.0",
    "aiofiles>=23.0.0",

    # Token counting
    "tiktoken>=0.5.0",

    # Progress bars
    "tqdm>=4.65.0",

    # Secure credential storage
    "keyring>=24.0.0",

    # Rate limiting
    "aiolimiter>=1.1.0",
]

# For development
dev = [
    "pytest-asyncio>=0.21.0",
    "pytest-mock>=3.11.0",
    "aioresponses>=0.7.4",  # Mock aiohttp
]
```

## Prompt Templates

### Default Template
The default template is designed to work across all providers:

```
You are a software engineer fixing a bug in {repo}.

Issue: {instance_id}
Repository: {repo}
Base Commit: {base_commit}

Problem Statement:
{problem_statement}

{hints_section}

{test_info_section}

Please provide a patch in unified diff format that fixes this issue.
```

### Model-Specific Templates

**Claude Template**: Uses XML tags for better structure recognition
**GPT-4 Template**: More concise, direct instructions
**Llama Template**: Includes examples for better format compliance

## Error Handling Scenarios

### Rate Limit Handling

```python
# Per-provider rate limit strategies
RATE_LIMIT_STRATEGIES = {
    "openai": {
        "base_delay": 60,
        "max_retries": 3,
        "backoff": "exponential"
    },
    "anthropic": {
        "base_delay": 30,
        "max_retries": 5,
        "backoff": "linear"
    },
    "groq": {
        "base_delay": 2,
        "max_retries": 10,
        "backoff": "exponential",
        "per_minute_limit": 30
    }
}
```

### Network Timeout Strategies

```python
TIMEOUT_STRATEGIES = {
    "default": {
        "connect_timeout": 10,
        "read_timeout": 120,
        "total_timeout": 300
    },
    "slow_providers": {
        "connect_timeout": 30,
        "read_timeout": 300,
        "total_timeout": 600
    }
}
```

### Partial Response Handling

```python
class PartialResponseHandler:
    """Handles incomplete responses from providers."""

    def can_recover(self, partial: str) -> bool:
        """Check if partial response is recoverable."""
        # Has some diff content
        if "diff --git" in partial or "---" in partial:
            return True
        return False

    def complete_response(self, partial: str) -> str:
        """Attempt to complete a partial response."""
        # Add missing closing markers
        if "@@" in partial and not partial.strip().endswith(("", "+")):
            # Find last complete line
            lines = partial.split("\n")
            # Truncate to last complete change
            # ...
```

## Testing Scenarios

### Unit Test Examples

```python
# tests/test_providers.py
import pytest
from unittest.mock import Mock, patch
from swebench_runner.providers import get_registry
from swebench_runner.providers.mock import MockProvider

class TestProviderRegistry:
    def test_provider_discovery(self):
        registry = get_registry()
        providers = registry.list_providers()
        assert len(providers) > 0
        assert any(p["name"] == "mock" for p in providers)

    def test_provider_instantiation(self):
        registry = get_registry()
        config = ProviderConfig(name="mock", api_key="test")
        provider = registry.get_provider("mock", config)
        assert isinstance(provider, MockProvider)

class TestPatchGeneration:
    @pytest.mark.asyncio
    async def test_successful_generation(self):
        # Setup mock provider
        provider = MockProvider(ProviderConfig(
            name="mock",
            extra_params={"responses": {"1": VALID_PATCH}}
        ))

        generator = PatchGenerator(provider)
        instance = {
            "instance_id": "test-123",
            "repo": "test/repo",
            "base_commit": "abc123",
            "problem_statement": "Fix the bug",
            "FAIL_TO_PASS": ["test_function"]
        }

        patch = await generator.generate_patch(instance)
        assert "diff --git" in patch
        assert "@@" in patch

    @pytest.mark.asyncio
    async def test_retry_on_invalid_format(self):
        # Provider returns invalid then valid
        responses = {
            "1": "This is not a patch",
            "2": VALID_PATCH
        }

        provider = MockProvider(ProviderConfig(
            name="mock",
            extra_params={"responses": responses}
        ))

        generator = PatchGenerator(provider)
        patch = await generator.generate_patch(SAMPLE_INSTANCE)

        assert provider.call_count == 2
        assert "diff --git" in patch
```

### Integration Test Scenarios

```python
# tests/integration/test_end_to_end.py
class TestEndToEnd:
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_generation_flow(self, tmp_path):
        # Use mock provider
        os.environ["MOCK_API_KEY"] = "test"

        # Run CLI command
        runner = CliRunner()
        result = runner.invoke(cli, [
            "generate",
            "--provider", "mock",
            "--dataset", "lite",
            "--count", "1",
            "--output", str(tmp_path)
        ])

        assert result.exit_code == 0
        predictions_file = list(tmp_path.glob("predictions_*.jsonl"))[0]
        assert predictions_file.exists()

        # Verify content
        with open(predictions_file) as f:
            prediction = json.loads(f.readline())
            assert "instance_id" in prediction
            assert "model" in prediction
            assert "prediction" in prediction

    @pytest.mark.integration
    async def test_resume_from_checkpoint(self, tmp_path):
        # Create checkpoint
        checkpoint_data = {
            "completed": ["instance-1", "instance-2"],
            "timestamp": "2024-01-01T00:00:00"
        }

        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        with open(checkpoint_dir / "test-run.json", "w") as f:
            json.dump(checkpoint_data, f)

        # Resume generation
        # Should skip completed instances
```

### Rate Limit Simulation Tests

```python
# tests/test_rate_limits.py
class TestRateLimits:
    @pytest.mark.asyncio
    async def test_rate_limit_retry(self):
        # Mock provider that returns rate limit error
        provider = MockProvider(ProviderConfig(
            name="mock",
            extra_params={"error": "Rate limit exceeded"}
        ))

        with patch("asyncio.sleep") as mock_sleep:
            processor = BatchProcessor(provider)

            # Should retry with appropriate delay
            result = await processor.process_batch(
                [SAMPLE_INSTANCE],
                PatchGenerator(provider),
                None
            )

            assert mock_sleep.called
            assert mock_sleep.call_args[0][0] >= 60  # Base delay
```

### Performance Benchmarks

```python
# tests/benchmarks/test_performance.py
import time

class TestPerformance:
    @pytest.mark.benchmark
    async def test_token_counting_performance(self, benchmark):
        manager = TokenManager()
        text = "x" * 10000  # 10k characters

        result = benchmark(manager.count_tokens, text, "gpt-4")
        assert result > 0

    @pytest.mark.benchmark
    async def test_batch_processing_speed(self, benchmark):
        provider = MockProvider(ProviderConfig(name="mock"))
        processor = BatchProcessor(provider)

        instances = [SAMPLE_INSTANCE] * 100

        start = time.time()
        result = await processor.process_batch(
            instances,
            PatchGenerator(provider),
            None
        )
        duration = time.time() - start

        assert duration < 60  # Should process 100 in under a minute
        assert len(result.successful) == 100
```

## Complete Example Usage

### Basic Usage

```bash
# Set API key
export OPENAI_API_KEY="sk-..."

# Generate patches for 5 lite instances
swebench generate --provider openai --dataset lite --count 5

# Use specific model
swebench generate --provider openai --model gpt-4-turbo --dataset lite --count 5

# Resume from checkpoint
swebench generate --provider openai --dataset lite --resume ./checkpoints/run-123
```

### Advanced Usage

```bash
# Use custom output directory
swebench generate --provider claude --dataset full --instances django__django-12345 --output ./my-patches

# Compare multiple providers
swebench compare --providers openai,claude,groq --dataset lite --count 10

# Generate and immediately evaluate
swebench run --provider openai --dataset lite --count 5 --evaluate
```

### Programmatic Usage

```python
from swebench_runner.providers import get_registry
from swebench_runner.generate import PatchGenerator
from swebench_runner.datasets import load_dataset_instances

async def generate_patches():
    # Get provider
    registry = get_registry()
    provider = registry.get_provider("openai")

    # Load instances
    instances = load_dataset_instances("lite", limit=5)

    # Generate patches
    generator = PatchGenerator(provider)
    patches = []

    for instance in instances:
        patch = await generator.generate_patch(instance)
        patches.append({
            "instance_id": instance["instance_id"],
            "patch": patch
        })

    return patches
```

## Final Notes

This plan balances robustness with practical implementation for V1:

1. **Production-grade architecture** that can support many providers
2. **Focused V1 scope** with just OpenAI and OpenRouter
3. **Clear extension points** for future providers
4. **Comprehensive error handling** from day one
5. **Zero-config operation** with environment variables

The system is designed to be rock-solid for production use while maintaining the flexibility to easily add new providers as needed.
EOF < /dev/null
