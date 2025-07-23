#!/bin/bash
# Check for generated files that should not be committed
# Used by pre-commit hooks and CI checks

# Find wheel and egg-info files, excluding .git directory
generated_files=$(find . -name "*.whl" -o -name "*.egg-info" | grep -v "^./.git" || true)

if [ -n "$generated_files" ]; then
    echo "Error: Found generated files that should not be committed:"
    echo "$generated_files"
    exit 1
fi

exit 0
