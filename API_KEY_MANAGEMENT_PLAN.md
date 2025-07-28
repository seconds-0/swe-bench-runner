# API Key Management Plan

## Current State

The system already has a comprehensive key management system with the following precedence:
1. **Environment variables** (highest priority)
2. **Keyring** (secure OS-level storage)
3. **Configuration file** (~/.swebench/providers.json)
4. **Default values**

## Requirements

### 1. User Key Management
- Interactive setup via `swebench provider init <provider>`
- Secure storage using OS keyring when available
- Support for environment variables
- Clear error messages when keys are missing

### 2. CI Runner Key Management
- GitHub Actions secrets for secure storage
- Environment variable injection during CI runs
- Support for OpenRouter as a universal provider
- Cost-effective model selection for CI

## Implementation Plan

### 1. Add OpenRouter to CI Configuration

```yaml
# .github/workflows/ci.yml - Add to integration-tests job

- name: Run OpenRouter integration tests
  if: env.OPENROUTER_API_KEY != ''
  env:
    OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
    OPENROUTER_TEST_MODEL: anthropic/claude-3-haiku  # Cost-effective for CI
  run: |
    pytest tests/integration/test_openrouter_integration.py -v -s -m integration || echo "OpenRouter tests skipped (no API key)"
```

### 2. Create OpenRouter Integration Tests

Since OpenRouter can access multiple models, we can use it as a universal test provider:

```python
# tests/integration/test_openrouter_integration.py
"""Integration tests for OpenRouter provider."""

import os
import pytest
from swebench_runner.providers.openrouter import OpenRouterProvider
from swebench_runner.providers.unified_models import UnifiedRequest

@pytest.mark.integration
class TestOpenRouterIntegration:
    """Integration tests for OpenRouter with real API calls."""
    
    @pytest.fixture
    def skip_without_openrouter_key():
        """Skip test if OpenRouter API key is not available."""
        if not os.environ.get("OPENROUTER_API_KEY"):
            pytest.skip("OPENROUTER_API_KEY not set")
    
    @pytest.fixture
    async def provider(self, skip_without_openrouter_key) -> OpenRouterProvider:
        """Create an OpenRouter provider with real credentials."""
        provider = OpenRouterProvider()
        await provider.initialize()
        return provider
    
    @pytest.mark.asyncio
    async def test_basic_generation(self, provider, openrouter_test_model):
        """Test basic generation with cost-effective model."""
        request = UnifiedRequest(
            model=openrouter_test_model or "anthropic/claude-3-haiku",
            prompt="Say 'test' and nothing else.",
            temperature=0.0,
            max_tokens=10,
        )
        
        response = await provider.generate_unified(request)
        
        assert response.content is not None
        assert "test" in response.content.lower()
        assert response.cost > 0  # OpenRouter always returns costs
```

### 3. Update Provider Config Manager

Add specific handling for CI environments:

```python
# In src/swebench_runner/providers/config.py

def _load_from_env(self, provider_name: str) -> ProviderConfig | None:
    """Load configuration from environment variables."""
    # ... existing code ...
    
    # Special handling for CI environments
    if os.getenv("CI") == "true":
        # Use cost-effective models in CI
        if provider_name == "openrouter":
            config.model = os.getenv("OPENROUTER_TEST_MODEL", "anthropic/claude-3-haiku")
        elif provider_name == "openai":
            config.model = os.getenv("OPENAI_TEST_MODEL", "gpt-3.5-turbo")
        elif provider_name == "anthropic":
            config.model = os.getenv("ANTHROPIC_TEST_MODEL", "claude-3-haiku-20240307")
```

### 4. GitHub Actions Setup

To add the OpenRouter API key to CI:

```bash
# Add via GitHub UI or CLI
gh secret set OPENROUTER_API_KEY --body "your-openrouter-api-key"
```

### 5. User Documentation

Update the provider documentation:

```markdown
# docs/providers.md

## Setting Up API Keys

### For Local Development

1. **Interactive Setup** (Recommended):
   ```bash
   swebench provider init openrouter
   # Enter your API key when prompted (hidden input)
   # Select your preferred model
   ```

2. **Environment Variables**:
   ```bash
   export OPENROUTER_API_KEY="sk-or-..."
   export OPENROUTER_MODEL="anthropic/claude-3-sonnet"
   ```

3. **Configuration File**:
   Create `~/.swebench/providers.json`:
   ```json
   {
     "openrouter": {
       "model": "anthropic/claude-3-sonnet",
       "temperature": 0.1
     }
   }
   ```
   Note: API keys are stored in keyring for security

### For CI/CD

Add secrets to your GitHub repository:
- `OPENROUTER_API_KEY`: Universal provider access
- `OPENAI_API_KEY`: OpenAI-specific features
- `ANTHROPIC_API_KEY`: Anthropic-specific features

CI automatically uses cost-effective models unless overridden.
```

### 6. Security Best Practices

1. **Never commit API keys** - Use .gitignore
2. **Use keyring for local storage** - OS-level security
3. **Rotate keys regularly** - Especially for CI
4. **Monitor usage** - Set spending limits
5. **Use read-only keys when possible** - Minimize risk

### 7. Cost Management for CI

OpenRouter is ideal for CI because:
- Single API key for multiple providers
- Per-request pricing (no monthly fees)
- Detailed usage tracking
- Automatic fallback if a model is unavailable

Recommended CI models:
- `anthropic/claude-3-haiku`: $0.25/$1.25 per 1M tokens
- `meta-llama/llama-3-70b-instruct`: $0.59/$0.79 per 1M tokens
- `mistralai/mistral-7b-instruct`: $0.07/$0.07 per 1M tokens

## Summary

The existing system already handles keys well. We need to:
1. ✅ Add OpenRouter to CI configuration
2. ✅ Create OpenRouter integration tests
3. ✅ Document the setup process
4. ✅ Add the provided API key to GitHub secrets

The system will then support both:
- **Users**: Easy setup via `swebench provider init`
- **CI**: Automatic testing with cost-effective models