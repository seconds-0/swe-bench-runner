#!/bin/bash
# Docker-based test runner for local development
# This allows running tests with the correct Python version regardless of local setup

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TEST_ARGS="${@:-tests/}"

echo -e "${GREEN}🐳 Running tests in Docker with Python ${PYTHON_VERSION}${NC}"
echo -e "${YELLOW}Test arguments: ${TEST_ARGS}${NC}"

# Build a temporary Docker image with our dependencies
echo -e "${YELLOW}Building test container...${NC}"
docker build -t swebench-test:py${PYTHON_VERSION} - <<EOF
FROM python:${PYTHON_VERSION}-slim

# Install git (needed for some tests)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/
COPY tests/ ./tests/

# Install dependencies
RUN pip install --no-cache-dir -e .[dev]

# Set environment variables for CI-like behavior
ENV CI=true
ENV SWEBENCH_CI_MIN_DISK_GB=15
ENV SWEBENCH_CI_MIN_MEMORY_GB=4

# Run tests by default
CMD ["pytest"]
EOF

# Run tests in the container
echo -e "${YELLOW}Running tests...${NC}"
docker run --rm \
    -v "$(pwd)":/app \
    -w /app \
    --name swebench-test-runner \
    swebench-test:py${PYTHON_VERSION} \
    pytest ${TEST_ARGS}

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ All tests passed!${NC}"
else
    echo -e "${RED}❌ Tests failed with exit code: $EXIT_CODE${NC}"
fi

exit $EXIT_CODE