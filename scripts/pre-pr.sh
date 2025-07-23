#!/bin/bash
# Run FULL CI simulation before opening a PR
# This runs EVERYTHING that CI would run, including things pre-commit skips
# Usage: ./scripts/pre-pr.sh

set -e

echo "🚀 Running FULL CI simulation before PR..."
echo "========================================"
echo ""

# Track failures
FAILURES=0
WARNINGS=0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. First run all pre-commit hooks
echo "📋 Running all pre-commit hooks..."
if ! pre-commit run --all-files; then
    echo -e "${RED}❌ Pre-commit hooks failed${NC}"
    echo "   Fix these first: pre-commit run --all-files"
    exit 1
fi
echo -e "${GREEN}✅ Pre-commit hooks passed${NC}"
echo ""

# 2. Run full test suite with all coverage formats (like CI)
echo "🧪 Running full test suite with coverage..."
if ! pytest tests/ -v --cov=swebench_runner --cov-report=xml --cov-report=term-missing --cov-fail-under=85; then
    echo -e "${RED}❌ Tests failed or coverage below 85%${NC}"
    ((FAILURES++))
fi
echo ""

# 3. Build package and check wheel (missing from pre-commit)
echo "📦 Building package and checking wheel..."
rm -rf dist/
if ! python -m build; then
    echo -e "${RED}❌ Package build failed${NC}"
    ((FAILURES++))
else
    # Check wheel with twine
    if ! twine check dist/*; then
        echo -e "${RED}❌ Wheel validation failed${NC}"
        ((FAILURES++))
    fi
    
    # Check wheel size
    wheel_files=$(find dist -name "*.whl" 2>/dev/null || true)
    if [ -z "$wheel_files" ]; then
        echo -e "${RED}❌ No wheel file found in dist/${NC}"
        ((FAILURES++))
    else
        for wheel in $wheel_files; do
            wheel_size=$(stat -f%z "$wheel" 2>/dev/null || stat -c%s "$wheel" 2>/dev/null || echo 0)
            wheel_size_mb=$((wheel_size / 1048576))
            if [ $wheel_size -gt 1048576 ]; then
                echo -e "${RED}❌ Wheel size ${wheel_size_mb}MB exceeds 1MB limit${NC}"
                ((FAILURES++))
            else
                echo -e "${GREEN}✅ Wheel size OK: ${wheel_size} bytes${NC}"
            fi
        done
    fi
fi
echo ""

# 4. Test installation from wheel (missing from pre-commit)
echo "📥 Testing package installation..."
# Create a temporary virtual environment
TEMP_DIR=$(mktemp -d)
TEMP_VENV="$TEMP_DIR/venv"

# Set up cleanup trap
trap 'deactivate 2>/dev/null || true; rm -rf "$TEMP_DIR" 2>/dev/null || true' EXIT

if ! python -m venv "$TEMP_VENV"; then
    echo -e "${RED}❌ Failed to create test virtual environment${NC}"
    ((FAILURES++))
else
    source "$TEMP_VENV/bin/activate"
    
    if pip install dist/*.whl; then
        # Test that CLI works
        if swebench --version && swebench --help > /dev/null; then
            echo -e "${GREEN}✅ Package installs and CLI works${NC}"
        else
            echo -e "${RED}❌ CLI entry point failed${NC}"
            ((FAILURES++))
        fi
    else
        echo -e "${RED}❌ Package installation failed${NC}"
        ((FAILURES++))
    fi
    
    deactivate
fi

# Remove trap after successful cleanup
trap - EXIT
rm -rf "$TEMP_DIR"
echo ""

# 5. Test on multiple Python versions if available
echo "🐍 Testing on multiple Python versions..."
TESTED_VERSIONS=0
# Only test a lightweight module for version compatibility
for py_version in python3.9 python3.10 python3.11 python3.12; do
    if command -v $py_version &> /dev/null; then
        echo -n "  Testing with $py_version... "
        # Quick import test instead of full test suite
        if $py_version -c "import swebench_runner; print(swebench_runner.__version__)" > /dev/null 2>&1; then
            echo -e "${GREEN}✅${NC}"
            ((TESTED_VERSIONS++))
        else
            echo -e "${RED}❌${NC}"
            echo -e "${YELLOW}   Warning: Import failed on $py_version${NC}"
            ((WARNINGS++))
        fi
    fi
done

if [ $TESTED_VERSIONS -eq 0 ]; then
    echo -e "${YELLOW}⚠️  Warning: Could only test on current Python version${NC}"
    echo "   CI will test on Python 3.9, 3.10, 3.11, and 3.12"
    ((WARNINGS++))
elif [ $TESTED_VERSIONS -lt 4 ]; then
    echo -e "${YELLOW}⚠️  Tested on $TESTED_VERSIONS/4 Python versions${NC}"
    ((WARNINGS++))
else
    echo -e "${GREEN}✅ Tested on all 4 Python versions${NC}"
fi
echo ""

# 6. Platform-specific warnings
echo "🖥️  Platform check..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo -e "${YELLOW}⚠️  Running on macOS - CI also tests on Ubuntu${NC}"
    ((WARNINGS++))
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo -e "${YELLOW}⚠️  Running on Linux - CI also tests on macOS${NC}"
    ((WARNINGS++))
fi
echo ""

# 7. Docker check (for awareness)
echo "🐳 Docker check..."
if docker info > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Docker is running (local tests more accurate)${NC}"
else
    echo -e "${YELLOW}⚠️  Docker not running - CI tests run without Docker too${NC}"
fi
echo ""

# 8. Final dependency audit
echo "🔒 Running security audit..."
if pip-audit --desc; then
    echo -e "${GREEN}✅ Security audit passed${NC}"
else
    echo -e "${YELLOW}⚠️  Security audit has warnings${NC}"
    ((WARNINGS++))
fi
echo ""

# Summary
echo "========================================"
echo "📊 Pre-PR Check Summary"
echo "========================================"

if [ $FAILURES -eq 0 ]; then
    echo -e "${GREEN}✅ All critical checks passed!${NC}"
    
    if [ $WARNINGS -gt 0 ]; then
        echo -e "${YELLOW}⚠️  $WARNINGS warnings (see above)${NC}"
        echo ""
        echo "These warnings indicate differences between your local environment"
        echo "and CI. Your PR will likely pass, but review the warnings."
    else
        echo -e "${GREEN}🎉 No warnings - your PR should pass CI!${NC}"
    fi
    
    echo ""
    echo "Ready to open PR? Next steps:"
    echo "1. git push origin $(git branch --show-current)"
    echo "2. Open PR on GitHub"
    echo "3. Check that all CI jobs pass"
else
    echo -e "${RED}❌ $FAILURES critical failures found${NC}"
    echo ""
    echo "Fix these issues before opening a PR."
    echo "The CI will definitely fail with these errors."
    exit 1
fi

# Clean up
rm -rf dist/ coverage.xml .coverage