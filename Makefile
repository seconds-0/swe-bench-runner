# Makefile for SWE-bench Runner
# Provides simple commands for common development tasks

.PHONY: help install hooks check test pre-pr clean reset-cli

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
	# Remove stray coverage data files sometimes created by coverage.py / pytest-cov
	# Includes odd filenames like '@.coverage.*' on macOS or temporary suffixed files
	find . -type f -name ".coverage*" -delete || true
	find . -type f -name "@.coverage*" -delete || true
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Development workflow shortcuts
lint:  ## Run linting with auto-fix
	ruff check --fix src/ tests/

mypy:  ## Run type checking
	mypy src/swebench_runner

# End-to-end testing
test-e2e:  ## Run end-to-end tests (actual CLI execution)
	pytest tests/e2e/ -v -s

test-e2e-quick:  ## Run quick E2E tests (no Docker required)
	pytest tests/e2e/test_cli_happy_path.py -v -s -k "not docker"

test-all: test test-e2e  ## Run all tests including E2E

quick:  ## Quick checks before commit (lint + mypy + critical tests)
	@echo "Running quick checks..."
	@make lint
	@make mypy
	@pytest tests/test_cli_critical.py -xvs

reset-cli:  ## Reinstall CLI in editable mode and verify paths
	@[ -d ".venv" ] && . .venv/bin/activate || true; \
	python -V; \
	pip install -e ".[dev]"; \
	hash -r 2>/dev/null || true; \
	echo "swebench path: $$(which swebench || echo 'not found')"; \
	echo "python path:   $$(which python)"; \
	python -c "import swebench_runner, sys; print('swebench_runner module:', swebench_runner.__file__); print('sys.executable:', sys.executable)"; \
	printf "swebench --version:\n"; swebench --version || true; \
	printf "Direct module (--version):\n"; PYTHONPATH=src python -m swebench_runner.cli --version || true
