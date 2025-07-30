# OpenAI Provider Setup Guide

This guide covers setting up the OpenAI provider for SWE-bench Runner.

## Prerequisites

1. OpenAI API account
2. API key with access to GPT models
3. Sufficient credits/budget for your evaluation needs

## Setup Steps

### 1. Get Your API Key

1. Visit [platform.openai.com](https://platform.openai.com)
2. Navigate to API Keys section
3. Create a new API key
4. Copy the key (starts with `sk-`)

### 2. Configure the Provider

#### Option A: Environment Variable (Recommended)
```bash
export OPENAI_API_KEY="sk-your-key-here"
swebench provider init openai
```

#### Option B: Interactive Setup
```bash
swebench provider init openai
# You'll be prompted for your API key
```

#### Option C: Manual Configuration
```bash
swebench provider init openai --api-key "sk-your-key-here"
```

### 3. Test Your Configuration

```bash
swebench provider test openai
```

Expected output:
```
✅ OpenAI provider configured successfully
✅ Authentication successful
✅ Model access verified: gpt-4o
💰 Account has credits available
```

## Available Models

List all available models:
```bash
swebench provider models openai
```

### Recommended Models for SWE-bench

| Model | Best For | Speed | Cost |
|-------|----------|-------|------|
| `gpt-4o` | Best accuracy | Fast | $$$ |
| `gpt-4-turbo` | High accuracy | Medium | $$$ |
| `gpt-3.5-turbo` | Quick testing | Very Fast | $ |

### Model Selection Examples

```bash
# Use GPT-4o (default, best performance)
swebench run --dataset lite --provider openai

# Use GPT-3.5 for faster, cheaper runs
swebench run --dataset lite --provider openai --model gpt-3.5-turbo

# Use specific GPT-4 variant
swebench run --dataset lite --provider openai --model gpt-4-turbo-2024-04-09
```

## Cost Management

### Estimate Costs Before Running

```bash
# Estimate cost for 50 instances
swebench generate --dataset lite --provider openai --count 50 --dry-run

# Output:
# 💰 Estimated cost: $12.50
# 📊 Tokens: ~500k prompt, ~250k completion
```

### Set Usage Limits

```bash
# Set a maximum cost per run
export OPENAI_MAX_COST=10.00

# Set token limits
export OPENAI_MAX_TOKENS_PER_MIN=150000
```

### Monitor Usage

```bash
# Check current month's usage
swebench provider usage openai
```

## Performance Optimization

### 1. Batch Processing
```bash
# Process in batches to avoid rate limits
swebench run --dataset lite --provider openai --batch-size 5
```

### 2. Parallel Requests
```bash
# Use multiple API keys for higher throughput
export OPENAI_API_KEYS="sk-key1,sk-key2,sk-key3"
swebench run --dataset lite --provider openai --max-workers 10
```

### 3. Retry Configuration
```bash
# Configure retry behavior
export OPENAI_MAX_RETRIES=5
export OPENAI_RETRY_DELAY=2
```

## Rate Limits

OpenAI enforces rate limits based on your tier:

| Tier | Requests/min | Tokens/min |
|------|--------------|------------|
| Free | 3 | 40,000 |
| Tier 1 | 60 | 150,000 |
| Tier 2 | 500 | 2,000,000 |

The provider automatically handles rate limiting, but you can configure behavior:

```bash
# Be conservative with rate limits
export OPENAI_RATE_LIMIT_BUFFER=0.8  # Use 80% of limit

# Or disable rate limiting (not recommended)
export OPENAI_DISABLE_RATE_LIMIT=true
```

## Advanced Configuration

### Custom Base URL (for proxies or compatible endpoints)
```bash
export OPENAI_BASE_URL="https://your-proxy.com/v1"
swebench provider init openai
```

### Organization ID
```bash
export OPENAI_ORG_ID="org-your-org-id"
swebench provider init openai
```

### Timeout Settings
```bash
# Increase timeout for slow networks (default: 60s)
export OPENAI_TIMEOUT=120
```

## Troubleshooting

### Authentication Errors
```
Error: Invalid API key
```
**Solution**: Check your API key is correct and has not been revoked

### Rate Limit Errors
```
Error: Rate limit exceeded
```
**Solution**: Reduce `--max-workers` or wait for limit reset

### Insufficient Credits
```
Error: You exceeded your current quota
```
**Solution**: Add credits to your OpenAI account

### Network Issues
```
Error: Connection timeout
```
**Solution**: Check internet connection, try increasing timeout

### Model Access Issues
```
Error: Model 'gpt-4' not found
```
**Solution**: Ensure your API key has access to the requested model

## Best Practices

1. **Start Small**: Test with 5-10 instances before large runs
2. **Use Appropriate Models**: GPT-3.5 for testing, GPT-4 for accuracy
3. **Monitor Costs**: Set budgets and check usage regularly
4. **Handle Secrets Safely**: Never commit API keys to version control
5. **Optimize Prompts**: Shorter prompts save tokens and money

## Security Notes

- Store API keys in environment variables or secure credential manager
- Use `.env` files with `.gitignore` for local development
- Rotate API keys regularly
- Use organization IDs to track usage by project

## Next Steps

- [Run your first evaluation](../getting-started.md#run-your-first-evaluation)
- [Compare with other providers](../examples/provider-comparison.md)
- [Optimize for cost and performance](../examples/cost-optimization.md)