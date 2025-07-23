# Makefile for SWE-bench Runner
# Provides simple commands for common development tasks

.PHONY: help install hooks check test pre-pr clean

help:  ## Show this help message
	@echo "SWE-bench Runner Development Commands"
	@echo "===================================="
	@echo ""
	@echo "Usage: make [command]"
	@echo ""
	@echo "Commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

install:  ## Install package in development mode with all dependencies
	pip install -e .
	pip install "pre-commit>=3.5.0" "pytest>=7.4.0" "pytest-cov>=4.1.0" "ruff>=0.1.0" "mypy>=1.5.0" "pip-audit>=2.6.0" "build>=1.0.0" "twine>=4.0.0"

hooks:  ## Install pre-commit hooks
	pre-commit install --install-hooks
	@echo "✅ Pre-commit hooks installed"

check:  ## Run quick CI checks (linting, type checking, critical tests)
	./scripts/check.sh

test:  ## Run full test suite with coverage
	pytest tests/ -v --cov=swebench_runner --cov-report=term-missing

test-no-docker:  ## Run tests without Docker (simulates CI environment safely)
	./scripts/test-no-docker.sh

pre-pr:  ## Run FULL CI simulation before opening a PR
	./scripts/pre-pr.sh

ci-full: pre-pr  ## Alias for pre-pr (run everything CI would run)

clean:  ## Clean build artifacts and caches
	rm -rf dist/ build/ *.egg-info/
	rm -rf .coverage coverage.xml htmlcov/
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Development workflow shortcuts
lint:  ## Run linting with auto-fix
	ruff check --fix src/ tests/

mypy:  ## Run type checking
	mypy src/swebench_runner

quick:  ## Quick checks before commit (lint + mypy + critical tests)
	@echo "Running quick checks..."
	@make lint
	@make mypy
	@pytest tests/test_cli_critical.py -xvs