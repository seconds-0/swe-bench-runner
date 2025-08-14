# SWE-Bench Runner

[![CI](https://github.com/seconds-0/swe-bench-runner/actions/workflows/ci.yml/badge.svg)](https://github.com/seconds-0/swe-bench-runner/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A delightfully simple CLI for running SWE-bench evaluations with any model provider

## Features

**✅ Complete SWE-bench Evaluation Pipeline**
- Run evaluations with a single command
- Automatic dataset fetching from HuggingFace
- Docker-based execution using official SWE-bench harness
- Support for Epoch AI optimized images (x86_64)
- **NEW: ARM64/Apple Silicon support with local Docker builds**
- Real-time progress tracking for long-running operations
- Comprehensive error handling and recovery

**✅ Multiple Model Providers**
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude 3)
- Ollama (local models)
- OpenRouter (multi-provider gateway)

**✅ Production-Ready**
- Thread-safe execution with resource management
- Automatic retries and circuit breakers
- Progress tracking and detailed logging
- HTML reports for easy result sharing

## Quick Start

### Installation

```bash
# From PyPI (coming soon)
pip install swebench-runner

# For development
git clone https://github.com/swebench/runner
cd runner
pip install -e .
```

### Prerequisites

- Python 3.11+
- Docker Desktop (macOS) or Docker Engine (Linux)
- 16GB+ RAM recommended
- 120GB+ free disk space for Docker images

**ARM64/Apple Silicon Users:**
- First run will build Docker images locally (30-60+ minutes per repository)
- Subsequent runs use cached images (fast)
- Automatic detection and configuration - no manual setup required!

### Basic Usage

```bash
# Run evaluation on a single patch
swebench run --patches predictions.jsonl

# Use a specific dataset (lite, verified, or full)
swebench run --patches predictions.jsonl --dataset lite

# Generate patches using a model provider
swebench generate --dataset lite --provider openai --model gpt-4

# Run evaluation with generated patches
swebench run --patches predictions.jsonl --dataset lite
```

### Model Provider Setup

```bash
# List available providers
swebench provider list

# Initialize a provider (interactive setup)
swebench provider init openai

# Test provider connectivity
swebench provider test openai

# List available models
swebench provider models openai
```

### Dataset Management

```bash
# Download and cache datasets
swebench dataset fetch lite

# View dataset information
swebench info --dataset lite

# Filter instances by pattern
swebench run --patches predictions.jsonl --filter "django__*"
```

## Architecture

SWE-Bench Runner uses a modular architecture:

```
┌─────────────────────────────────────────────────┐
│                   CLI Layer                      │
│  (Click-based commands and argument parsing)     │
└─────────────────────────────────────────────────┘
                         │
┌─────────────────────────────────────────────────┐
│              Provider Abstraction                │
│  (Unified interface for all model providers)     │
└─────────────────────────────────────────────────┘
                         │
┌─────────────────────────────────────────────────┐
│             SWE-bench Harness                    │
│  (Official harness via subprocess execution)     │
└─────────────────────────────────────────────────┘
                         │
┌─────────────────────────────────────────────────┐
│              Docker Execution                    │
│   (Epoch AI images or local image building)      │
└─────────────────────────────────────────────────┘
```

## Provider Support

| Provider | Models | Features | Status |
|----------|--------|----------|--------|
| OpenAI | GPT-4, GPT-3.5 | Streaming, Function Calling | ✅ Complete |
| Anthropic | Claude 3 Family | Native API, Streaming | ✅ Complete |
| Ollama | Local Models | Auto-discovery, No API Key | ✅ Complete |
| OpenRouter | 100+ Models | Multi-provider Gateway | ✅ Complete |

## Documentation

- [Quick Start Guide](Documentation/QuickStart.md) - Get running in 5 minutes
- [Architecture](Documentation/Architecture.md) - Technical design
- [Provider Setup](Documentation/provider-setup/) - Configure each provider
- [Testing Guide](Documentation/Testing-Setup.md) - Run and write tests

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/swebench/runner
cd runner

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
make hooks
```

### Development Commands

```bash
# Run tests
make test          # Run test suite
make check         # Quick CI checks
make pre-pr        # Full CI simulation before PR

# Code quality
make lint          # Auto-fix linting issues
make mypy          # Type checking

# Cleanup
make clean         # Remove build artifacts
```

### Running Tests

```bash
# Unit tests only
pytest tests/unit -v

# Integration tests (requires Docker)
pytest tests/integration -v

# E2E tests
pytest tests/e2e -v

# All tests with coverage
pytest --cov=swebench_runner --cov-report=html
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

MIT - See [LICENSE](LICENSE) for details

## Acknowledgments

- [SWE-bench](https://www.swebench.com/) team for the benchmark
- [Epoch AI](https://github.com/epoch-research) for optimized Docker images
- All contributors and users of this tool
