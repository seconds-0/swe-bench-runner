#!/bin/bash
# Test in CI-like environment (no Docker)

# Stop Docker to simulate CI
echo "Stopping Docker to simulate CI environment..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    osascript -e 'quit app "Docker"' 2>/dev/null || true
else
    sudo systemctl stop docker 2>/dev/null || true
fi

# Run tests
pytest tests/ -v

echo "Remember to restart Docker!"