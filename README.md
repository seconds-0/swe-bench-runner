# SWE-Bench Runner

[![CI](https://github.com/seconds-0/swe-bench-runner/actions/workflows/ci.yml/badge.svg)](https://github.com/seconds-0/swe-bench-runner/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Run any subset of SWE-bench with one clear command and fix any issue in minutes, not hours.

⚠️ **Status**: MVP CLI implemented. Docker execution coming next.

## Vision

SWE-bench has become the de-facto benchmark for evaluating code-fixing agents, but getting the harness to run locally is still painful. This tool makes SWE-bench evaluation so simple that users think "Holy shit, this is what I wanted the whole time!"

## Current Status

**✅ MVP CLI Complete** - Basic command-line interface is ready:

```bash
pip install swebench-runner
swebench run --patches my_patches.jsonl
```

The CLI currently validates patches files and provides foundation for Docker execution (coming next).

## Installation

```bash
pip install swebench-runner
```

## Usage

```bash
# Check version
swebench --version

# Get help
swebench --help
swebench run --help

# Run evaluation (MVP - validation only)
swebench run --patches predictions.jsonl
```

## Current Features

- ✅ Click-based CLI with `swebench` command
- ✅ `run` command with `--patches` flag
- ✅ File validation (exists, not empty, is file)
- ✅ Proper error handling and exit codes
- ✅ Comprehensive test suite (96% coverage)
- ✅ Type checking with mypy
- ✅ Linting with ruff
- ✅ <5KB wheel size

## Coming Next

- 🚧 Docker execution (MVP-DockerRun)
- 🚧 Basic output formatting (MVP-BasicOutput)
- 🚧 Progress tracking and HTML reports

## Documentation

- [Product Requirements Document](Documentation/PRD.md) - What we're building and why
- [UX Plan](Documentation/UX_Plan.md) - How users will interact with the tool
- [Architecture](Documentation/Architecture.md) - Technical design and implementation
- [V2 Features](Documentation/V2_Features.md) - Future enhancements

## Development

This project is being developed with AI assistance following the workplan methodology outlined in [CLAUDE.md](CLAUDE.md).

### Development Commands

```bash
# Install development environment
make install        # Install package + dev dependencies
make hooks         # Install pre-commit hooks

# During development
make check         # Run quick CI checks
make test          # Run full test suite
make lint          # Auto-fix linting issues
make mypy          # Run type checking

# Before opening a PR
make pre-pr        # Run FULL CI simulation
make ci-full       # Alias for pre-pr

# Cleanup
make clean         # Remove build artifacts and caches
```

## License

MIT
