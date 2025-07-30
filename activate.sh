#!/bin/bash
# Activate the Python 3.11 virtual environment for this project

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment with Python 3.11..."
    /opt/homebrew/bin/python3.11 -m venv .venv
fi

source .venv/bin/activate

# Verify correct Python version
python_version=$(python --version 2>&1)
if [[ ! "$python_version" == *"3.11"* ]]; then
    echo "Warning: Expected Python 3.11, but got $python_version"
fi

echo "Activated Python environment: $python_version"
echo "Run 'deactivate' to exit the virtual environment"
