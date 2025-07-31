# Ollama Provider Setup Guide

This guide covers setting up the Ollama provider for running models locally with SWE-bench Runner.

## Prerequisites

1. A computer with at least 8GB RAM (16GB+ recommended)
2. Ollama installed on your system
3. Sufficient disk space for models (5-50GB per model)

## Installation

### macOS
```bash
# Install using Homebrew
brew install ollama

# Or download from ollama.ai
curl -fsSL https://ollama.ai/install.sh | sh
```

### Linux
```bash
# Install script
curl -fsSL https://ollama.ai/install.sh | sh

# Or using package manager (Ubuntu/Debian)
sudo apt-get install ollama
```

### Windows
Download the installer from [ollama.ai](https://ollama.ai/download/windows)

## Setup Steps

### 1. Start Ollama Service

```bash
# Start Ollama (if not already running)
ollama serve

# Check if Ollama is running
curl http://localhost:11434/api/tags
```

### 2. Pull Models

```bash
# Pull recommended models for SWE-bench
ollama pull llama3.2:1b     # Smallest, fastest (1.3GB)
ollama pull llama3.2:3b     # Better performance (2GB)
ollama pull mistral:7b      # Good balance (4.1GB)
ollama pull codellama:7b    # Code-specialized (3.8GB)
ollama pull gemma2:9b       # Google's model (5.5GB)
```

### 3. Configure the Provider

```bash
# Ollama requires no API key!
swebench provider init ollama

# Test the connection
swebench provider test ollama
```

Expected output:
```
 Ollama provider configured successfully
 Connection successful (http://localhost:11434)
 Found 5 models available
=€ Ready to run locally!
```

## Available Models

### List Installed Models
```bash
# Via Ollama
ollama list

# Via SWE-bench Runner
swebench provider models ollama
```

### Recommended Models for SWE-bench

| Model | Size | RAM Needed | Best For | Speed |
|-------|------|------------|----------|--------|
| `llama3.2:1b` | 1.3GB | 3GB | Quick tests | ¡¡¡¡¡ |
| `llama3.2:3b` | 2GB | 4GB | Basic tasks | ¡¡¡¡ |
| `mistral:7b` | 4.1GB | 8GB | General use | ¡¡¡ |
| `codellama:7b` | 3.8GB | 8GB | Code tasks | ¡¡¡ |
| `gemma2:9b` | 5.5GB | 10GB | Quality | ¡¡ |
| `llama3.2:70b` | 40GB | 48GB | Best quality | ¡ |

### Model Selection Examples

```bash
# Use small model for quick testing
swebench run --dataset lite --provider ollama --model llama3.2:1b

# Use code-specialized model
swebench run --dataset lite --provider ollama --model codellama:7b

# Use larger model for better results
swebench run --dataset lite --provider ollama --model gemma2:9b
```

## Performance Optimization

### 1. GPU Acceleration

#### NVIDIA GPUs
```bash
# Check if GPU is detected
ollama run llama3.2:1b --verbose

# Should show: "Using NVIDIA GPU"
```

#### Apple Silicon
```bash
# Automatically uses Metal acceleration
# No configuration needed!
```

#### AMD GPUs (Linux)
```bash
# Set environment variable
export HSA_OVERRIDE_GFX_VERSION=10.3.0
ollama serve
```

### 2. Memory Management

```bash
# Limit memory usage
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_PARALLEL=2

# Increase context window (uses more memory)
export OLLAMA_NUM_CTX=4096
```

### 3. CPU Optimization

```bash
# Set thread count
export OLLAMA_NUM_THREAD=8

# Use all CPU cores
export OLLAMA_NUM_THREAD=$(nproc)
```

## Advanced Configuration

### Custom Model Parameters

```bash
# Create custom model with specific parameters
ollama create myswebench -f - <<EOF
FROM llama3.2:3b
PARAMETER temperature 0
PARAMETER num_predict 2048
PARAMETER top_k 40
PARAMETER top_p 0.9
EOF

# Use custom model
swebench run --dataset lite --provider ollama --model myswebench
```

### Network Configuration

```bash
# Run Ollama on different port
OLLAMA_HOST=0.0.0.0:8080 ollama serve

# Configure SWE-bench Runner
export OLLAMA_BASE_URL="http://localhost:8080"
swebench provider init ollama
```

### Model Storage Location

```bash
# Change model storage directory
export OLLAMA_MODELS="/path/to/models"
ollama serve
```

## Running Evaluations

### Basic Usage

```bash
# Run with default model (llama3.2:1b)
swebench run --dataset lite --provider ollama

# Run with specific model
swebench run --dataset lite --provider ollama --model codellama:7b

# Run with custom parameters
swebench run --dataset lite --provider ollama \
  --model mistral:7b \
  --temperature 0 \
  --max-tokens 2000
```

### Batch Processing

```bash
# Process one at a time (recommended for large models)
swebench run --dataset lite --provider ollama \
  --model gemma2:9b \
  --max-workers 1

# Parallel processing (for smaller models)
swebench run --dataset lite --provider ollama \
  --model llama3.2:1b \
  --max-workers 4
```

## Cost Benefits

### Free and Private
-  No API costs
-  No usage limits
-  Complete privacy
-  No internet required

### Speed Advantages
- Faster response times (no network latency)
- Consistent performance
- No rate limiting

## Troubleshooting

### Ollama Not Running
```
Error: Connection refused to localhost:11434
```
**Solution**: Start Ollama service
```bash
ollama serve
```

### Model Not Found
```
Error: Model 'llama3.2:1b' not found
```
**Solution**: Pull the model first
```bash
ollama pull llama3.2:1b
```

### Out of Memory
```
Error: Out of memory (OOM)
```
**Solution**: Use a smaller model or increase system RAM
```bash
# Use smaller model
swebench run --dataset lite --provider ollama --model llama3.2:1b

# Or limit context
export OLLAMA_NUM_CTX=2048
```

### Slow Performance
```
Warning: Model running slowly
```
**Solution**: Check GPU acceleration or use smaller model
```bash
# Check if using GPU
ollama ps

# Use quantized model
ollama pull llama3.2:1b-q4_0
```

### Port Already in Use
```
Error: bind: address already in use
```
**Solution**: Kill existing process or use different port
```bash
# Find and kill process
lsof -i :11434
kill -9 <PID>

# Or use different port
OLLAMA_HOST=0.0.0.0:8080 ollama serve
```

## Best Practices

1. **Model Selection**:
   - Start with small models (1b, 3b) for testing
   - Use code-specialized models for better results
   - Balance model size with available RAM

2. **Resource Management**:
   - Monitor RAM usage with `htop` or Activity Monitor
   - Close other applications when using large models
   - Use GPU acceleration when available

3. **Performance Tuning**:
   - Use temperature 0 for consistency
   - Limit max tokens to control generation time
   - Adjust context window based on needs

4. **Development Workflow**:
   - Test with small models first
   - Validate results before scaling up
   - Keep multiple models for different tasks

## Integration Examples

### CI/CD Pipeline
```yaml
# GitHub Actions example
name: Local SWE-bench Evaluation
on: [push]

jobs:
  evaluate:
    runs-on: self-hosted  # Requires self-hosted runner
    steps:
      - uses: actions/checkout@v2
      
      - name: Start Ollama
        run: |
          ollama serve &
          sleep 5
          ollama pull llama3.2:3b
      
      - name: Run Evaluation
        run: |
          pip install swebench-runner
          swebench run \
            --dataset lite \
            --provider ollama \
            --model llama3.2:3b \
            --count 10
```

### Docker Deployment
```dockerfile
FROM ollama/ollama:latest

# Install models
RUN ollama pull llama3.2:3b
RUN ollama pull codellama:7b

# Install SWE-bench Runner
RUN pip install swebench-runner

# Run evaluation
CMD ["swebench", "run", "--dataset", "lite", "--provider", "ollama"]
```

## Comparing Models

### Quick Benchmark
```bash
# Compare different models on same dataset
for model in llama3.2:1b llama3.2:3b mistral:7b codellama:7b; do
  echo "Testing $model..."
  swebench run \
    --dataset lite \
    --provider ollama \
    --model $model \
    --count 5 \
    --output "results_$model.json"
done
```

## Security and Privacy

### Complete Privacy
- All processing happens locally
- No data sent to external servers
- No API keys or authentication needed
- Perfect for sensitive codebases

### Network Isolation
```bash
# Run completely offline
swebench run --dataset lite --provider ollama --offline
```

## Next Steps

- [Compare local vs cloud models](../examples/local-vs-cloud.md)
- [Optimize for your hardware](../examples/hardware-optimization.md)
- [Create custom models](../examples/custom-ollama-models.md)
- [Try other providers](../getting-started.md#using-different-model-providers)