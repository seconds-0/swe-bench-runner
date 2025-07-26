# TokenManager Component

The TokenManager is a critical component of the SWE-bench runner's model provider integration that manages token counting and context window limits across different models and providers.

## Key Features

1. **Accurate Token Counting**
   - Uses tiktoken for OpenAI models when available
   - Falls back to character-based estimation for other models
   - Caches token counts for performance

2. **Model Context Limits**
   - Hardcoded limits for major models (GPT-4, Claude, etc.)
   - Conservative defaults for unknown models
   - Supports both direct model names and provider-prefixed names

3. **Smart Context Truncation**
   - Multiple truncation strategies:
     - **BALANCED**: Reduces all sections proportionally
     - **PRESERVE_TESTS**: Keeps test information, reduces code
     - **PRESERVE_RECENT**: Keeps recent/relevant code
     - **SUMMARY**: Summarizes removed content (future)
   - Preserves code structure (imports, signatures) when truncating
   - Adds clear truncation markers

4. **Cost Estimation**
   - Estimates generation costs based on known pricing
   - Supports input/output token pricing

## Usage Example

```python
from swebench_runner.generation import TokenManager, TruncationStrategy, PromptContext

# Create token manager
manager = TokenManager(cache_enabled=True)

# Count tokens
tokens = manager.count_tokens("Hello world", "gpt-4")

# Check model limits
limit = manager.get_model_limit("openai", "gpt-4-turbo")

# Fit context within limits
context = PromptContext(
    problem_statement="Fix the bug...",
    code_context={"main.py": "..."},
    test_context={"test_main.py": "..."}
)

prompt, truncated, stats = manager.fit_context(
    context,
    "gpt-4",
    reserve_tokens=1000,
    strategy=TruncationStrategy.PRESERVE_TESTS
)

if truncated:
    print(manager.format_truncation_summary(stats))

# Estimate costs
cost = manager.estimate_cost(
    prompt_tokens=5000,
    max_completion_tokens=2000,
    provider="openai",
    model="gpt-4-turbo"
)
```

## Dependencies

- **Required**: None (pure Python)
- **Optional**: `tiktoken` for accurate OpenAI token counting
  - Install with: `pip install swebench-runner[tokenizer]`
  - Falls back to estimation if not available

## Integration with Other Components

- Works with `PromptBuilder` to create prompts that fit within limits
- Used by `PatchGenerator` to ensure prompts don't exceed model capabilities
- Provides token counts for cost estimation and monitoring

## Configuration

The TokenManager supports several configuration options:

- `cache_enabled`: Whether to cache token counts (default: True)
- `cache_size`: Maximum cache entries (default: 10000)
- `fallback_estimation`: Use estimation when tokenizer unavailable (default: True)
- `default_chars_per_token`: Characters per token for estimation (default: 4.0)

## Model Support

As of early 2024, the following models are supported with accurate limits:

### OpenAI
- GPT-4 Turbo: 128,000 tokens
- GPT-4: 8,192 tokens
- GPT-4-32k: 32,768 tokens
- GPT-3.5 Turbo: 16,385 tokens

### Anthropic
- Claude 3 (Opus/Sonnet/Haiku): 200,000 tokens
- Claude 2.1: 200,000 tokens
- Claude 2: 100,000 tokens

### Default
- Unknown models: 4,096 tokens (conservative)

## Truncation Strategies

### BALANCED
Reduces all sections proportionally to fit within limits. Good for general use.

### PRESERVE_TESTS
Prioritizes keeping test information intact while aggressively truncating code context. Useful when test failures are the primary diagnostic tool.

### PRESERVE_RECENT
Keeps recent/relevant code while truncating older parts. Good for focusing on specific areas.

### SUMMARY (Future)
Will summarize removed content instead of just truncating. Not yet implemented.
