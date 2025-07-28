#!/usr/bin/env python3
"""Comprehensive validation script for all integration tests.

This script validates that all provider integration tests follow
our unified API design correctly.
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple


class TestValidator:
    """Validates integration test files for API compliance."""
    
    def __init__(self):
        self.bad_patterns = [
            (r'messages\s*=\s*\[', 'Should use prompt= instead of messages='),
            (r'response\.choices\[0\]', 'Should use response.content/finish_reason'),
            (r'response\.usage\.(prompt|completion|total)_cost', 'Should use response.cost'),
            (r'chunk\.choices\[0\]', 'Should use chunk.content'),
            (r'stream_unified', 'Should use generate_stream'),
            (r'os\.environ\[.*API_KEY.*\]\s*=', 'Should use monkeypatch for env vars'),
        ]
        
        self.good_patterns = [
            ('prompt=', 'Using unified prompt parameter'),
            ('response.content', 'Using unified response.content'),
            ('response.cost', 'Using unified response.cost'),
            ('response.finish_reason', 'Using unified response.finish_reason'),
            ('generate_stream', 'Using correct streaming method'),
            ('monkeypatch', 'Using proper test isolation'),
            ('system_message=', 'Using system_message parameter'),
        ]
        
        self.required_tests = [
            'test_basic_generation',
            'test_streaming_generation',
            'test_model_not_found_error',
            'test_authentication_error',
            'test_cost_calculation',
            'test_model_availability',
        ]
        
        self.recommended_tests = [
            'test_network_timeout_handling',
            'test_concurrent_requests',
            'test_system_message_handling',
            'test_max_tokens_enforcement',
        ]
    
    def check_file(self, filepath: Path) -> Tuple[List[str], List[str], List[str]]:
        """Check a single test file for issues and good patterns."""
        if not filepath.exists():
            return [f"File not found: {filepath}"], [], []
        
        with open(filepath) as f:
            content = f.read()
            lines = content.split('\n')
        
        issues = []
        good_indicators = []
        
        # Check for bad patterns
        for line_num, line in enumerate(lines, 1):
            for pattern, message in self.bad_patterns:
                if re.search(pattern, line):
                    issues.append(f"Line {line_num}: {message} - Found: {line.strip()[:80]}...")
        
        # Check for good patterns
        for pattern, description in self.good_patterns:
            if pattern in content:
                good_indicators.append(f"✅ {description}")
        
        # Check for required tests
        missing_tests = []
        for test_name in self.required_tests:
            if f"def {test_name}" not in content:
                missing_tests.append(f"Missing required test: {test_name}")
        
        # Check for recommended tests
        for test_name in self.recommended_tests:
            if f"def {test_name}" in content:
                good_indicators.append(f"✅ Has {test_name}")
        
        return issues, good_indicators, missing_tests
    
    def validate_all(self) -> int:
        """Validate all integration test files."""
        test_files = [
            ("OpenAI", Path("tests/integration/test_openai_integration.py")),
            ("Anthropic", Path("tests/integration/test_anthropic_integration.py")),
            ("Ollama", Path("tests/integration/test_ollama_integration.py")),
        ]
        
        print("=" * 70)
        print("Comprehensive Integration Test Validation")
        print("=" * 70)
        
        total_issues = 0
        all_results = []
        
        for provider, filepath in test_files:
            print(f"\n## {provider} Integration Tests")
            print("-" * 50)
            
            issues, good_indicators, missing_tests = self.check_file(filepath)
            
            if issues:
                print(f"❌ Found {len(issues)} API issues:")
                for issue in issues[:5]:  # Show first 5 issues
                    print(f"   - {issue}")
                if len(issues) > 5:
                    print(f"   ... and {len(issues) - 5} more issues")
                total_issues += len(issues)
            else:
                print("✅ No API mismatches found!")
            
            if missing_tests:
                print(f"\n⚠️  Missing {len(missing_tests)} required tests:")
                for test in missing_tests:
                    print(f"   - {test}")
            
            if good_indicators:
                print("\n✅ Good patterns found:")
                for indicator in good_indicators:
                    print(f"   {indicator}")
            
            all_results.append({
                'provider': provider,
                'issues': len(issues),
                'missing_tests': len(missing_tests),
                'good_patterns': len(good_indicators)
            })
        
        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        
        print("\n| Provider   | Issues | Missing Tests | Good Patterns |")
        print("|------------|--------|---------------|---------------|")
        for result in all_results:
            print(f"| {result['provider']:10} | {result['issues']:6} | {result['missing_tests']:13} | {result['good_patterns']:13} |")
        
        print(f"\nTotal Issues: {total_issues}")
        
        if total_issues == 0:
            print("\n✅ SUCCESS: All integration tests follow the unified API correctly!")
            return 0
        else:
            print(f"\n❌ FAILED: Found {total_issues} issues that need fixing")
            print("\nNext Steps:")
            print("1. Fix all 'messages=' usage to use 'prompt='")
            print("2. Replace response.choices[0] patterns with response.content")
            print("3. Use response.cost instead of response.usage.*_cost")
            print("4. Use monkeypatch for environment variable manipulation")
            print("5. Add missing required tests")
            return 1


def main():
    """Run the validation."""
    validator = TestValidator()
    return validator.validate_all()


if __name__ == "__main__":
    sys.exit(main())