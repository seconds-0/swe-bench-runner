#!/bin/bash
# Test in CI-like environment (simulating no Docker)
# This uses environment variables to mock Docker unavailability
# instead of actually stopping Docker services

echo "🐳 Testing in CI-like environment (Docker disabled)..."
echo "=================================================="
echo ""

# Set environment variables to simulate CI
export CI=true
export SWEBENCH_MOCK_NO_DOCKER=true
export SWEBENCH_SKIP_DOCKER_TESTS=true

# Additional CI simulation settings
export SWEBENCH_CI_MIN_MEMORY_GB=4
export SWEBENCH_CI_MIN_DISK_GB=20

echo "Environment configured:"
echo "  CI=true"
echo "  SWEBENCH_MOCK_NO_DOCKER=true"
echo "  SWEBENCH_SKIP_DOCKER_TESTS=true"
echo "  SWEBENCH_CI_MIN_MEMORY_GB=4"
echo "  SWEBENCH_CI_MIN_DISK_GB=20"
echo ""

# Run tests
echo "Running tests..."
python3 -m pytest tests/ -v

# Capture exit code
TEST_EXIT_CODE=$?

# Clean up environment
unset CI
unset SWEBENCH_MOCK_NO_DOCKER
unset SWEBENCH_SKIP_DOCKER_TESTS
unset SWEBENCH_CI_MIN_MEMORY_GB
unset SWEBENCH_CI_MIN_DISK_GB

echo ""
echo "=================================================="
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "✅ All tests passed in CI-like environment!"
else
    echo "❌ Tests failed in CI-like environment"
    echo "   This simulates how tests will run in GitHub Actions"
fi

exit $TEST_EXIT_CODE
