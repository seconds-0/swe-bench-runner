# Anthropic Provider Setup Guide

This guide covers setting up the Anthropic (Claude) provider for SWE-bench Runner.

## Prerequisites

1. Anthropic API account
2. API key with access to Claude models
3. Sufficient credits for your evaluation needs

## Setup Steps

### 1. Get Your API Key

1. Visit [console.anthropic.com](https://console.anthropic.com)
2. Navigate to API Keys section
3. Create a new API key
4. Copy the key (starts with `sk-ant-`)

### 2. Configure the Provider

#### Option A: Environment Variable (Recommended)
```bash
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
swebench provider init anthropic
```

#### Option B: Interactive Setup
```bash
swebench provider init anthropic
# You'll be prompted for your API key
```

#### Option C: Manual Configuration
```bash
swebench provider init anthropic --api-key "sk-ant-your-key-here"
```

### 3. Test Your Configuration

```bash
swebench provider test anthropic
```

Expected output:
```
 Anthropic provider configured successfully
 Authentication successful
 Model access verified: claude-3-opus-20240229
=° Account has credits available
```

## Available Models

List all available models:
```bash
swebench provider models anthropic
```

### Recommended Models for SWE-bench

| Model | Best For | Speed | Cost | Context |
|-------|----------|-------|------|---------|
| `claude-3-opus-20240229` | Best accuracy | Slow | $$$$ | 200k |
| `claude-3-sonnet-20240229` | Balance | Medium | $$ | 200k |
| `claude-3-haiku-20240307` | Quick testing | Fast | $ | 200k |

### Model Selection Examples

```bash
# Use Claude 3 Opus (best performance)
swebench run --dataset lite --provider anthropic --model claude-3-opus-20240229

# Use Claude 3 Haiku for faster, cheaper runs
swebench run --dataset lite --provider anthropic --model claude-3-haiku-20240307

# Use Claude 3 Sonnet for balanced performance
swebench run --dataset lite --provider anthropic --model claude-3-sonnet-20240229
```

## Cost Management

### Pricing (as of 2024)

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| Claude 3 Opus | $15.00 | $75.00 |
| Claude 3 Sonnet | $3.00 | $15.00 |
| Claude 3 Haiku | $0.25 | $1.25 |

### Estimate Costs Before Running

```bash
# Estimate cost for 50 instances
swebench generate --dataset lite --provider anthropic --model claude-3-haiku-20240307 --count 50 --dry-run

# Output:
# =° Estimated cost: $2.50
# =Ê Tokens: ~400k prompt, ~200k completion
```

### Set Usage Limits

```bash
# Set a maximum cost per run
export ANTHROPIC_MAX_COST=10.00

# Set token limits
export ANTHROPIC_MAX_TOKENS_PER_MIN=100000
```

### Monitor Usage

```bash
# Check current usage
swebench provider usage anthropic
```

## Performance Optimization

### 1. Batch Processing
```bash
# Process in smaller batches to manage costs
swebench run --dataset lite --provider anthropic --batch-size 3
```

### 2. System Prompts
Anthropic models work especially well with clear system prompts:

```bash
# Use a custom system prompt for better results
swebench run --dataset lite --provider anthropic \
  --system-prompt "You are an expert Python developer fixing GitHub issues."
```

### 3. Temperature Settings
```bash
# Use temperature 0 for consistency
swebench run --dataset lite --provider anthropic --temperature 0

# Use higher temperature for creative solutions
swebench run --dataset lite --provider anthropic --temperature 0.7
```

## Rate Limits

Anthropic enforces rate limits based on your tier:

| Tier | Requests/min | Tokens/min |
|------|--------------|------------|
| Free Trial | 5 | 10,000 |
| Build Tier 1 | 50 | 50,000 |
| Build Tier 2 | 1,000 | 100,000 |
| Scale | Custom | Custom |

The provider automatically handles rate limiting with exponential backoff.

### Configure Rate Limiting

```bash
# Be conservative with rate limits
export ANTHROPIC_RATE_LIMIT_BUFFER=0.8  # Use 80% of limit

# Adjust retry behavior
export ANTHROPIC_MAX_RETRIES=5
export ANTHROPIC_RETRY_DELAY=2
```

## Advanced Configuration

### Custom Base URL
```bash
# For proxies or compatible endpoints
export ANTHROPIC_BASE_URL="https://your-proxy.com/v1"
swebench provider init anthropic
```

### Timeout Settings
```bash
# Increase timeout for complex tasks (default: 600s)
export ANTHROPIC_TIMEOUT=900
```

### Beta Features
```bash
# Enable beta features (when available)
export ANTHROPIC_BETA="messages-2023-12-15"
```

## Special Features

### 1. Long Context Window
Claude models support up to 200k token context windows:

```bash
# Process large codebases
swebench run --dataset lite --provider anthropic \
  --model claude-3-sonnet-20240229 \
  --max-context 150000
```

### 2. Vision Capabilities
Claude 3 models can process images (useful for UI-related issues):

```bash
# Enable vision processing (when applicable)
swebench run --dataset lite --provider anthropic \
  --model claude-3-opus-20240229 \
  --enable-vision
```

### 3. XML Formatting
Claude models work well with structured output:

```bash
# Request structured output
swebench run --dataset lite --provider anthropic \
  --output-format xml
```

## Troubleshooting

### Authentication Errors
```
Error: Invalid API key
```
**Solution**: Ensure your API key starts with `sk-ant-` and is active

### Rate Limit Errors
```
Error: Rate limit exceeded
```
**Solution**: Reduce `--max-workers` or upgrade your tier

### Credit Errors
```
Error: Insufficient credits
```
**Solution**: Add credits to your Anthropic account

### Model Access Issues
```
Error: Model not found
```
**Solution**: Check model name spelling and availability in your region

### Timeout Issues
```
Error: Request timeout
```
**Solution**: Increase timeout or reduce prompt complexity

## Best Practices

1. **Model Selection**:
   - Use Haiku for development and testing
   - Use Sonnet for balanced performance
   - Use Opus for critical evaluations

2. **Prompt Engineering**:
   - Use clear, specific system prompts
   - Leverage Claude's 200k context window
   - Use XML tags for structured output

3. **Cost Optimization**:
   - Start with Haiku to test your workflow
   - Monitor token usage carefully
   - Use temperature 0 for consistency

4. **Safety and Compliance**:
   - Anthropic has strong safety measures
   - Some prompts may be filtered
   - Respect content policy guidelines

## Integration Tips

### Working with Large Codebases
```bash
# Claude's large context window is perfect for complex issues
swebench run --dataset lite --provider anthropic \
  --model claude-3-sonnet-20240229 \
  --include-full-context \
  --max-context 100000
```

### Combining with Other Providers
```bash
# Use Claude for analysis, GPT-4 for implementation
swebench analyze --provider anthropic --model claude-3-opus-20240229
swebench generate --provider openai --model gpt-4
```

## Security Notes

- Store API keys securely (never in code)
- Use environment variables or secure vaults
- Rotate keys regularly
- Monitor usage for anomalies
- Be aware of data retention policies

## Next Steps

- [Run your first evaluation](../getting-started.md#run-your-first-evaluation)
- [Compare Claude models](../examples/claude-comparison.md)
- [Optimize for large codebases](../examples/large-context-optimization.md)
- [Try Ollama for local models](ollama-setup.md)