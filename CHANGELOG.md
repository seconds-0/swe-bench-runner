# Changelog

All notable changes to the SWE-bench Runner project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **ARM64/Apple Silicon Support**: Full support for ARM64 architecture with automatic detection and local Docker image building
- **Real-time Progress Tracking**: Live progress updates during Docker builds with phase detection
- **Interactive Setup Wizard**: Improved TUI with better ARM64 handling and flow
- Comprehensive E2E test infrastructure with test doubles
- Documentation consolidation and cleanup
- Unit test coverage improvements
- Python 3.11+ migration and compatibility

### Changed
- Updated README to reflect actual working features and ARM64 support
- Improved test infrastructure with dependency injection
- Enhanced error handling and user experience
- Docker namespace handling for architecture-specific builds

### Fixed
- TUI flow issues with early returns in preflight wizard
- Rich markup errors in console output
- Test hanging issues in CI environment
- Python version compatibility issues
- Resource management and cleanup

## [0.1.0] - 2025-08-07

### Added

#### Core Features
- **CLI Framework**: Click-based command-line interface with `swebench` command
- **Docker Execution**: Integration with official SWE-bench harness
- **Dataset Management**: Automatic fetching from HuggingFace with caching
- **Batch Evaluation**: Support for evaluating multiple patches
- **Progress Tracking**: Real-time progress updates during evaluation
- **HTML Reports**: Comprehensive evaluation reports with statistics

#### Model Provider System
- **Provider Abstraction**: Unified interface for all model providers
- **OpenAI Integration**: GPT-4 and GPT-3.5 support with streaming
- **Anthropic Integration**: Claude 3 family support with native API
- **Ollama Integration**: Local model support with auto-discovery
- **OpenRouter Integration**: Multi-provider gateway with 100+ models
- **Provider CLI**: Commands for init, test, list, and models

#### Advanced Features
- **Circuit Breaker**: Fault tolerance for provider failures
- **Rate Limiting**: Coordinated rate limiting across providers
- **Token Counting**: Unified token counting and cost estimation
- **Streaming Support**: SSE and JSON Lines streaming adapters
- **Authentication**: Bearer token and API key strategies
- **Resource Checking**: Memory and disk space validation

#### Testing Infrastructure
- **E2E Tests**: Comprehensive end-to-end test suite
- **Integration Tests**: Provider integration testing
- **Unit Tests**: Module-level testing with test doubles
- **Test Doubles**: 7 types of test doubles for better isolation
- **CI/CD Pipeline**: GitHub Actions with multi-Python support

### Changed
- Migrated from direct Docker execution to SWE-bench harness
- Switched to Epoch AI optimized images for x86_64
- Improved error messages with actionable fix suggestions
- Enhanced resource checking with configurable thresholds

### Fixed
- Async test hanging issues in CI
- Rate limiter timing issues in tests
- Docker daemon detection on different platforms
- Network error handling and retries
- Large patch handling with environment variable limits

## [0.0.1] - 2025-07-17 (Initial Development)

### Added
- Initial project structure
- Basic CLI with version and help commands
- Project documentation (PRD, Architecture, UX Plan)
- Development environment setup
- Pre-commit hooks and CI configuration

## Commit Categories

### Features (feat)
- CLI command implementation
- Provider integration framework
- Dataset auto-fetch capability
- Batch evaluation support
- E2E test infrastructure

### Fixes (fix)
- CI/CD pipeline issues
- Test hanging and timing issues
- Lint and type checking errors
- Platform compatibility issues
- Resource management bugs

### Documentation (docs)
- Architecture documentation
- Provider setup guides
- Testing documentation
- Implementation plans
- API documentation

### Tests (test)
- Unit test additions
- Integration test fixes
- E2E test implementation
- Test infrastructure improvements

### Chores (chore)
- Dependency updates
- Build configuration
- CI/CD improvements
- Code formatting

## Migration Notes

### From Development to v0.1.0
- Python 3.11+ is now required (previously 3.8+)
- Docker is required for evaluation execution
- Environment variables for provider API keys
- New provider CLI commands available

## Contributors

This project has been developed with AI assistance following test-driven development practices and comprehensive documentation-first approach.

## Links

- [GitHub Repository](https://github.com/swebench/runner)
- [Issue Tracker](https://github.com/swebench/runner/issues)
- [Documentation](./Documentation/)
