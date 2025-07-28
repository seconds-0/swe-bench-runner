#!/bin/bash
# Setup script for Ollama integration testing

set -e

echo "🦙 Ollama Integration Test Setup"
echo "================================"

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama not found. Installing..."
    
    # Detect OS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        echo "Installing Ollama for macOS..."
        curl -fsSL https://ollama.com/install.sh | sh
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        echo "Installing Ollama for Linux..."
        curl -fsSL https://ollama.com/install.sh | sh
    else
        echo "❌ Unsupported OS: $OSTYPE"
        echo "Please install Ollama manually from: https://ollama.com/download"
        exit 1
    fi
else
    echo "✅ Ollama is already installed"
fi

# Check if Ollama is running
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama service is running"
else
    echo "⚠️  Ollama service is not running"
    echo "Starting Ollama service..."
    
    # Start Ollama in background
    ollama serve > /dev/null 2>&1 &
    OLLAMA_PID=$!
    
    # Wait for service to start
    echo -n "Waiting for Ollama to start"
    for i in {1..30}; do
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo ""
            echo "✅ Ollama service started (PID: $OLLAMA_PID)"
            break
        fi
        echo -n "."
        sleep 1
    done
    
    if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo ""
        echo "❌ Failed to start Ollama service"
        exit 1
    fi
fi

# Check if test model is available
TEST_MODEL="${OLLAMA_TEST_MODEL:-llama3.2:1b}"
echo ""
echo "Checking for test model: $TEST_MODEL"

if ollama list | grep -q "$TEST_MODEL"; then
    echo "✅ Test model $TEST_MODEL is already available"
else
    echo "📥 Pulling test model $TEST_MODEL..."
    echo "This may take a few minutes on first run..."
    
    if ollama pull "$TEST_MODEL"; then
        echo "✅ Successfully pulled $TEST_MODEL"
    else
        echo "❌ Failed to pull $TEST_MODEL"
        exit 1
    fi
fi

# Verify setup
echo ""
echo "🔍 Verifying Ollama setup..."

# Test a simple generation
echo "Testing model with a simple prompt..."
if echo "Say hello" | ollama run "$TEST_MODEL" --verbose=false | grep -qi "hello"; then
    echo "✅ Model is working correctly"
else
    echo "⚠️  Model test did not produce expected output"
fi

# Show available models
echo ""
echo "📋 Available models:"
ollama list

echo ""
echo "✅ Ollama is ready for integration testing!"
echo ""
echo "To run Ollama integration tests:"
echo "  pytest tests/integration/test_ollama_integration.py -m integration -v"
echo ""
echo "Or use the integration test runner:"
echo "  python scripts/run_integration_tests.py --provider ollama"