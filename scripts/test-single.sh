#!/bin/bash
# Test a single file quickly
# Usage: ./scripts/test-single.sh tests/test_cli_generation_simple.py

if [ -z "$1" ]; then
    echo "Usage: $0 <test_file>"
    echo "Example: $0 tests/test_cli_generation_simple.py"
    exit 1
fi

TEST_FILE="$1"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

echo "🧪 Testing ${TEST_FILE} with Python ${PYTHON_VERSION}..."

docker run --rm \
    -v "$(pwd)":/app \
    -w /app \
    python:${PYTHON_VERSION}-slim \
    bash -c "
        pip install -q -e .[dev] 2>/dev/null && \
        python -m pytest ${TEST_FILE} -v --tb=short -x
    "