#!/bin/bash
# Quick Docker-based test runner (no image building)
# For rapid test iteration during development

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TEST_ARGS="${@:-tests/}"

echo -e "${GREEN}🚀 Quick test run with Python ${PYTHON_VERSION}${NC}"

# Check if we have a specific test file
if [[ "$1" == *".py" ]]; then
    echo -e "${YELLOW}Running specific test: $1${NC}"
else
    echo -e "${YELLOW}Running tests in: ${TEST_ARGS}${NC}"
fi

# Run tests directly in Python container
docker run --rm \
    -v "$(pwd)":/app \
    -w /app \
    -e CI=true \
    -e SWEBENCH_CI_MIN_DISK_GB=15 \
    -e SWEBENCH_CI_MIN_MEMORY_GB=4 \
    python:${PYTHON_VERSION}-slim \
    bash -c "
        pip install -q -e .[dev] && \
        pytest ${TEST_ARGS} -v --tb=short
    "

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ Tests passed!${NC}"
else
    echo -e "${RED}❌ Tests failed!${NC}"
fi

exit $EXIT_CODE