#!/bin/bash
# Quick CI check - run before pushing
set -e

echo "Running quick CI checks..."

# Only the checks that actually failed in PR #6
ruff check src/ tests/ || (echo "❌ Lint failed - run: ruff check --fix" && exit 1)
mypy src/ || (echo "❌ Type check failed" && exit 1)
pytest tests/test_cli_critical.py -xvs || (echo "❌ CLI tests failed" && exit 1)

echo "✅ Basic checks passed - OK to push"