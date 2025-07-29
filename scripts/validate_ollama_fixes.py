#!/usr/bin/env python3
"""Validation script for Ollama integration test fixes."""

import re
from pathlib import Path


def check_api_patterns(filepath: Path) -> list:
    """Check for incorrect API patterns."""
    issues = []

    with open(filepath) as f:
        content = f.read()
        lines = content.split('\n')

    # Patterns to check for (these should NOT exist)
    bad_patterns = [
        (r'messages\s*=\s*\[', 'Should use prompt= instead of messages='),
        (r'response\.choices\[0\]', 'Should use response.content/finish_reason'),
        (r'chunk\.choices\[0\]', 'Should use chunk.content'),
        (r'stream_unified', 'Should use generate_stream'),
    ]

    for line_num, line in enumerate(lines, 1):
        for pattern, message in bad_patterns:
            if re.search(pattern, line):
                issues.append(f"Line {line_num}: {message} - Found: {line.strip()}")

    return issues


def check_good_patterns(filepath: Path) -> list:
    """Check that good patterns are present."""
    good_indicators = []

    with open(filepath) as f:
        content = f.read()

    # Patterns we SHOULD see
    good_patterns = [
        ('prompt=', 'Using unified prompt parameter'),
        ('response.content', 'Using unified response.content'),
        ('response.cost', 'Using unified response.cost (should be 0)'),
        ('response.finish_reason', 'Using unified response.finish_reason'),
        ('generate_stream', 'Using correct streaming method'),
        ('system_message=', 'Using system_message parameter'),
        ('test_network_timeout', 'Has timeout test'),
        ('test_concurrent_requests', 'Has concurrent requests test'),
    ]

    for pattern, description in good_patterns:
        if pattern in content:
            good_indicators.append(f"✅ {description}")

    return good_indicators


def main():
    """Run validation checks."""
    print("=" * 60)
    print("Ollama Integration Test Validation")
    print("=" * 60)

    filepath = Path("tests/integration/test_ollama_integration.py")

    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        return 1

    # Check for bad patterns
    print("\n1. Checking for API mismatches...")
    issues = check_api_patterns(filepath)

    if issues:
        print("❌ Found API issues:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("✅ No API mismatches found!")

    # Check for good patterns
    print("\n2. Checking for correct patterns...")
    good_patterns = check_good_patterns(filepath)
    for pattern in good_patterns:
        print(f"   {pattern}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)

    if issues:
        print(f"❌ FAILED: Found {len(issues)} issues that need fixing")
        return 1
    else:
        print("✅ SUCCESS: All API patterns are correct!")
        return 0


if __name__ == "__main__":
    exit(main())
