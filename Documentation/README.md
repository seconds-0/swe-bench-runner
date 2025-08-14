# SWE-bench Runner Documentation

## Core Documentation

### Product & Design
- **[PRD.md](PRD.md)** - Product requirements and vision
- **[Architecture.md](Architecture.md)** - Technical design and system architecture
- **[UX_Plan.md](UX_Plan.md)** - User experience design and error messages
- **[V2_Features.md](V2_Features.md)** - Future features and roadmap

### Getting Started
- **[QuickStart.md](QuickStart.md)** - Installation and basic usage guide
- **[docs/](../docs/)** - Provider setup guides and tutorials

### Testing
- **[TestPhilosophy.md](TestPhilosophy.md)** - Testing principles and approach
- **[Testing-Setup.md](Testing-Setup.md)** - Comprehensive testing guide including:
  - Local testing setup
  - Test doubles implementation (Phase 2.1)
  - Unit test coverage (Phase 2.3)
  - Implementation history
- **[Integration-Testing-Guide.md](Integration-Testing-Guide.md)** - Provider integration testing
- **[E2E_TEST_ANALYSIS_AND_REMEDIATION.md](E2E_TEST_ANALYSIS_AND_REMEDIATION.md)** - End-to-end test strategy
- **[CI-PreCommit-Parity.md](CI-PreCommit-Parity.md)** - CI and pre-commit configuration

### Project History
- **[Implementation-History.md](Implementation-History.md)** - Complete development timeline
- **[Plans/OVERVIEW-START-HERE.md](Plans/OVERVIEW-START-HERE.md)** - Current project status

## Documentation Structure

```
Documentation/
├── README.md                    # This file - documentation index
├── Core Design
│   ├── PRD.md                  # Product requirements
│   ├── Architecture.md         # Technical architecture
│   ├── UX_Plan.md             # User experience
│   └── V2_Features.md         # Future roadmap
├── Testing
│   ├── TestPhilosophy.md      # Testing principles
│   ├── Testing-Setup.md       # Comprehensive test guide
│   ├── Integration-Testing-Guide.md
│   ├── E2E_TEST_ANALYSIS_AND_REMEDIATION.md
│   └── CI-PreCommit-Parity.md
├── Getting Started
│   └── QuickStart.md          # Quick start guide
├── History
│   └── Implementation-History.md  # Development timeline
└── Plans/
    ├── OVERVIEW-START-HERE.md    # Current status
    ├── active/                    # Active work plans
    ├── completed/                 # Completed plans
    ├── archive/                   # Historical documents
    └── templates/                 # Plan templates
```

## Key Achievements

### Testing Infrastructure
- **Test Doubles**: 97% of environment mocks replaced with clean test doubles
- **Coverage**: Unit tests for critical security and performance paths
- **E2E Tests**: 100% assertion coverage for all error scenarios
- **Performance**: ~20% faster test execution

### Code Quality
- **Type Safety**: Full type hints throughout codebase
- **Error Handling**: Clear, actionable error messages
- **Security**: ReDoS prevention, token budget enforcement
- **Documentation**: Comprehensive guides for all features

### Provider System
- **4 Providers**: OpenAI, Anthropic, Ollama, OpenRouter
- **Unified Interface**: Single abstraction for all providers
- **Advanced Features**: Streaming, rate limiting, circuit breakers
- **Cost Control**: Token counting and budget tracking

## Quick Links

- **Start Here**: [QuickStart.md](QuickStart.md)
- **Current Status**: [Plans/OVERVIEW-START-HERE.md](Plans/OVERVIEW-START-HERE.md)
- **Testing**: [Testing-Setup.md](Testing-Setup.md)
- **History**: [Implementation-History.md](Implementation-History.md)

## Documentation Philosophy

1. **Living Documents**: All docs are updated as code changes
2. **No Fragments**: Consolidated documentation, no scattered pieces
3. **User-Focused**: Every doc helps users succeed
4. **Implementation History**: Past work preserved in Implementation-History.md

---

*Last Updated: 2025-08-07*