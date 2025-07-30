#!/usr/bin/env python3
"""Validation script specifically for Anthropic integration test fixes.

This script validates that all the API mismatches have been fixed.
"""

import re
from pathlib import Path


def check_response_access_patterns(filepath: Path) -> list:
    """Check for incorrect response access patterns."""
    issues = []

    with open(filepath) as f:
        content = f.read()
        lines = content.split('\n')

    # Patterns to check for (these should NOT exist)
    bad_patterns = [
        (r'response\.choices\[0\]\.message\.content', 'Should use response.content'),
        (r'response\.choices\[0\]\.finish_reason', 'Should use response.finish_reason'),
        (r'response\.usage\.prompt_cost', 'Should use response.cost'),
        (r'response\.usage\.completion_cost', 'Should use response.cost'),
        (r'response\.usage\.total_cost', 'Should use response.cost'),
        (r'chunk\.choices\[0\]\.delta\.content', 'Should use chunk.content'),
        (r'chunk\.choices\s*and\s*chunk\.choices\[0\]', 'Should check chunk.content directly'),
    ]

    for line_num, line in enumerate(lines, 1):
        for pattern, message in bad_patterns:
            if re.search(pattern, line):
                issues.append(f"Line {line_num}: {message} - Found: {line.strip()}")

    # Check for environment variable manipulation without monkeypatch
    if 'os.environ["ANTHROPIC_API_KEY"]' in content and 'monkeypatch' not in content:
        issues.append("Environment variable manipulation without monkeypatch fixture")

    return issues


def check_good_patterns(filepath: Path) -> list:
    """Check that good patterns are present."""
    good_indicators = []

    with open(filepath) as f:
        content = f.read()

    # Patterns we SHOULD see
    good_patterns = [
        ('response.content', 'Using unified response.content'),
        ('response.cost', 'Using unified response.cost'),
        ('response.finish_reason', 'Using unified response.finish_reason'),
        ('chunk.content', 'Using unified chunk.content for streaming'),
        ('monkeypatch', 'Using monkeypatch for test isolation'),
    ]

    for pattern, description in good_patterns:
        if pattern in content:
            good_indicators.append(f"✅ {description}")

    return good_indicators


def main():
    """Run validation checks."""
    print("=" * 60)
    print("Anthropic Integration Test Validation")
    print("=" * 60)

    filepath = Path("tests/integration/test_anthropic_integration.py")

    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        return 1

    # Check for bad patterns
    print("\n1. Checking for API mismatches...")
    issues = check_response_access_patterns(filepath)

    if issues:
        print("❌ Found API mismatches:")
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
