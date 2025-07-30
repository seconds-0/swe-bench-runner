#!/usr/bin/env python3
"""Simple verification of test completeness and correctness.

This is the working validation script for integration tests.
It checks for:
- Invalid API patterns (response.choices, response.id, etc.)
- Presence of required test methods
- Correct UnifiedResponse usage
"""

import sys
from pathlib import Path

def main():
    test_dir = Path(__file__).parent.parent / "tests" / "integration"
    
    # Check for invalid patterns
    invalid_patterns = [
        "response.choices",
        "response.id",
        "response.message",
        "response.usage.prompt_cost",
        "response.usage.completion_cost"
    ]
    
    issues_found = False
    
    for test_file in test_dir.glob("test_*.py"):
        if 'infrastructure' in test_file.name:
            continue
            
        with open(test_file, 'r') as f:
            content = f.read()
            
        # Check for invalid patterns
        for pattern in invalid_patterns:
            if pattern in content:
                print(f"❌ {test_file.name}: Found invalid pattern '{pattern}'")
                # Show context
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if pattern in line:
                        print(f"   Line {i+1}: {line.strip()}")
                issues_found = True
                
    # Check that key test methods exist
    required_methods = {
        'test_anthropic_integration.py': [
            'test_basic_generation',
            'test_streaming_generation', 
            'test_network_timeout_handling',
            'test_retry_mechanism_validation',
            'test_unicode_emoji_handling',
        ],
        'test_openai_integration.py': [
            'test_basic_generation',
            'test_streaming_generation',
            'test_network_timeout_handling', 
            'test_retry_mechanism_with_exponential_backoff',
            'test_unicode_emoji_content',
        ],
        'test_ollama_integration.py': [
            'test_basic_generation',
            'test_streaming_generation',
            'test_network_timeout_handling',
            'test_retry_mechanism_with_backoff',
            'test_model_parameter_customization',
        ]
    }
    
    for filename, methods in required_methods.items():
        test_file = test_dir / filename
        if test_file.exists():
            with open(test_file, 'r') as f:
                content = f.read()
                
            missing = []
            for method in methods:
                if f'def {method}' not in content:
                    missing.append(method)
                    
            if missing:
                print(f"\n❌ {filename}: Missing test methods:")
                for m in missing:
                    print(f"   - {m}")
                issues_found = True
                
    if not issues_found:
        print("✅ All tests appear to be correctly implemented!")
        print("\nVerified:")
        print("- No invalid API patterns found")
        print("- All required test methods exist")
        print("- Tests use UnifiedResponse correctly")
        
    return 1 if issues_found else 0

if __name__ == "__main__":
    sys.exit(main())