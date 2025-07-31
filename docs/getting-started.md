# Getting Started with SWE-bench Runner

Welcome to SWE-bench Runner! This guide will help you go from installation to your first evaluation run in under 10 minutes.

## 🚀 Quick Start (5 minutes)

### 1. Install SWE-bench Runner

```bash
pip install swebench-runner
```

### 2. Configure a Model Provider

Choose your preferred model provider:

#### Option A: OpenAI (Recommended for beginners)
```bash
export OPENAI_API_KEY="your-api-key-here"
swebench provider init openai
```

#### Option B: Anthropic (Claude)
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
swebench provider init anthropic
```

#### Option C: Ollama (Local models)
```bash
# First, install and start Ollama from https://ollama.ai
ollama pull llama3.3  # or any model you prefer
swebench provider init ollama
```

### 3. Run Your First Evaluation

```bash
# Quick test with 5 instances from the lite dataset
swebench run --dataset lite --provider openai --count 5
```

That's it! The runner will:
- ✅ Auto-download the dataset
- ✅ Generate patches using your chosen model
- ✅ Apply patches and run tests in Docker
- ✅ Generate an HTML report with results

## 📋 Common Commands

### List Available Providers
```bash
swebench provider list
```

### Test Provider Connection
```bash
swebench provider test openai
```

### List Available Models for a Provider
```bash
swebench provider models openai
```

### Run on Different Datasets
```bash
# Lite dataset (300 instances, good for testing)
swebench run --dataset lite --provider openai

# Verified dataset (500 instances, human-verified)
swebench run --dataset verified --provider anthropic

# Full dataset (2,294 instances)
swebench run --dataset full --provider ollama --model llama3.3
```

### Use Existing Patches
```bash
# If you already have patches in JSONL format
swebench run --patches my_patches.jsonl

# If you have patches in a directory
swebench run --patches-dir ./patches/
```

### Generate Patches Only (Without Evaluation)
```bash
swebench generate --dataset lite --provider openai --count 10
```

## 🎯 Selecting Instances

### Run Specific Instances
```bash
# By instance ID
swebench run --dataset lite --instances "django__django-11999,astropy__astropy-14365"

# Using a pattern
swebench run --dataset lite --subset "django/*" --regex
```

### Random Sampling
```bash
# Random 20 instances
swebench run --dataset verified --count 20 --sample random

# Random 10% of dataset
swebench run --dataset lite --sample "10%"
```

## 🔧 Advanced Options

### Parallel Execution
```bash
# Use 8 parallel workers (default: CPU count)
swebench run --dataset lite --provider openai --max-workers 8
```

### Offline Mode
```bash
# Skip dataset updates, use cached data only
swebench run --dataset lite --provider ollama --offline
```

### Cost Estimation
```bash
# See cost estimate before running (OpenAI/Anthropic)
swebench generate --dataset lite --provider openai --count 50 --dry-run
```

## 💡 Tips for Success

1. **Start Small**: Use `--count 5` to test your setup before running larger evaluations

2. **Monitor Costs**: For cloud providers (OpenAI/Anthropic), use the cost estimation feature

3. **Use Lite Dataset**: Perfect for development and testing (300 instances)

4. **Check Docker**: Ensure Docker is running before starting evaluation
   ```bash
   docker info
   ```

5. **Resource Requirements**:
   - Disk: ~20GB for Docker images
   - RAM: 8GB minimum, 16GB recommended
   - Network: First run downloads ~1GB of data

## 🆘 Troubleshooting

### Docker Not Found
```bash
# macOS
brew install --cask docker
open /Applications/Docker.app

# Linux
sudo apt-get install docker.io
sudo systemctl start docker
```

### API Key Issues
```bash
# Check your configuration
swebench provider test openai

# Re-initialize if needed
swebench provider init openai --force
```

### Out of Memory
```bash
# Reduce parallel workers
swebench run --dataset lite --provider openai --max-workers 2
```

### Slow Generation
```bash
# Use a faster model
swebench run --dataset lite --provider openai --model gpt-3.5-turbo
```

## 📚 Next Steps

- [Provider Setup Guides](providers/) - Detailed setup for each provider
- [CLI Reference](cli-reference.md) - Complete command documentation
- [Examples](../examples/) - Common use cases and workflows

## 🎉 Success!

You're now ready to run SWE-bench evaluations! Start with a small test run to ensure everything works, then scale up as needed.

Remember: The goal is to evaluate how well models can fix real GitHub issues. Have fun exploring!