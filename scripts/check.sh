#!/bin/bash
# Run all CI checks locally - MUST match CI exactly to prevent drift
# This script runs the same checks as .github/workflows/ci.yml

set -e

echo "🔍 Running full CI checks locally..."
echo "=================================="

# 1. Linting (matches CI lint job)
echo "📝 Running ruff..."
ruff check src/swebench_runner tests || (echo "❌ Lint failed - run: ruff check --fix" && exit 1)

# 2. Type checking (matches CI lint job)
echo "🔍 Running mypy..."
mypy src/swebench_runner || (echo "❌ Type check failed" && exit 1)

# 3. Tests with coverage (matches CI test job)
echo "🧪 Running tests with coverage..."
pytest tests/ -v --cov=swebench_runner --cov-fail-under=85 || (echo "❌ Tests failed or coverage below 85%" && exit 1)

# 4. Check for large files (matches PR checks)
echo "📦 Checking file sizes..."
large_files=$(find . -type f -size +1M | grep -v "^./.git" || true)
if [ ! -z "$large_files" ]; then
    echo "❌ Found files larger than 1MB:"
    echo "$large_files"
    exit 1
fi

# 5. Check for generated files (matches PR checks)
echo "🚫 Checking for generated files..."
if find . -name "*.whl" -o -name "*.egg-info" | grep -v "^./.git"; then
    echo "❌ Found generated files that should not be committed"
    exit 1
fi

# 6. Build check (matches CI build job)
echo "🏗️  Testing package build..."
python -m build --wheel --outdir dist-test/ > /dev/null 2>&1 || (echo "❌ Package build failed" && exit 1)
rm -rf dist-test/

# 7. Security audit (optional - matches CI security job)
if command -v pip-audit &> /dev/null; then
    echo "🔒 Running security audit..."
    pip-audit --desc || echo "⚠️  Security audit had warnings (non-blocking)"
else
    echo "ℹ️  Skipping security audit (pip-audit not installed)"
fi

echo ""
echo "✅ All CI checks passed! Safe to push."
echo ""
echo "💡 Tip: Install pre-commit hooks to run these automatically:"
echo "   pre-commit install --install-hooks"