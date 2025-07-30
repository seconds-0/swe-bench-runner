#!/usr/bin/env python3
"""Simple validation script for integration tests.

This script performs basic validation to ensure:
1. UnifiedRequest objects can be created correctly
2. The API format matches what providers expect
3. Tests can import and instantiate providers
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_unified_request_creation():
    """Test that we can create UnifiedRequest objects correctly."""
    print("\n1. Testing UnifiedRequest creation...")
    try:
        from swebench_runner.providers.unified_models import UnifiedRequest

        # Test basic creation with prompt
        request = UnifiedRequest(
            prompt="Test prompt",
            model="gpt-3.5-turbo",
            max_tokens=100,
            temperature=0.7
        )

        # Verify attributes
        assert request.prompt == "Test prompt"
        assert request.model == "gpt-3.5-turbo"
        assert request.max_tokens == 100
        assert request.temperature == 0.7

        print("   ✅ UnifiedRequest creation works correctly")
        return True

    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


def test_provider_imports():
    """Test that we can import all providers."""
    print("\n2. Testing provider imports...")

    providers_to_test = [
        ("OpenAI", "swebench_runner.providers.openai", "OpenAIProvider"),
        ("Anthropic", "swebench_runner.providers.anthropic", "AnthropicProvider"),
        ("Ollama", "swebench_runner.providers.ollama", "OllamaProvider"),
    ]

    all_passed = True
    for name, module_path, class_name in providers_to_test:
        try:
            module = __import__(module_path, fromlist=[class_name])
            provider_class = getattr(module, class_name)
            print(f"   ✅ {name} provider can be imported")
        except Exception as e:
            print(f"   ❌ {name} provider import failed: {e}")
            all_passed = False

    return all_passed


def test_api_format():
    """Test that the API format is correct (no 'messages' parameter)."""
    print("\n3. Testing API format correctness...")

    test_files = [
        "tests/integration/test_openai_integration.py",
        "tests/integration/test_anthropic_integration.py",
        "tests/integration/test_ollama_integration.py"
    ]

    all_passed = True
    for filepath in test_files:
        path = Path(filepath)
        if not path.exists():
            print(f"   ⚠️  {filepath} not found")
            continue

        with open(path) as f:
            content = f.read()

        # Check for incorrect 'messages=' usage in UnifiedRequest calls
        # Look for patterns like UnifiedRequest(...messages=...)
        import re
        unified_request_pattern = r'UnifiedRequest\s*\([^)]*messages\s*='

        if re.search(unified_request_pattern, content):
            print(f"   ❌ {path.name}: Found 'messages=' parameter in UnifiedRequest (should be 'prompt=')")
            all_passed = False
        else:
            # Also check if test is using prompt= correctly
            if "UnifiedRequest" in content and "prompt=" in content:
                print(f"   ✅ {path.name}: Uses correct API format with 'prompt='")
            else:
                print(f"   ✅ {path.name}: Uses correct API format")

    return all_passed


def test_single_test_syntax():
    """Test that a single test from each provider has valid syntax."""
    print("\n4. Testing individual test syntax...")

    # Try to run a simple test function from each provider
    test_modules = [
        ("OpenAI", "tests.integration.test_openai_integration"),
        ("Anthropic", "tests.integration.test_anthropic_integration"),
        ("Ollama", "tests.integration.test_ollama_integration"),
    ]

    all_passed = True
    for name, module_path in test_modules:
        try:
            # Just try to import the module - if syntax is correct, this will work
            module = __import__(module_path, fromlist=[''])
            print(f"   ✅ {name} tests have valid syntax")
        except SyntaxError as e:
            print(f"   ❌ {name} tests have syntax error: {e}")
            all_passed = False
        except Exception as e:
            # Other errors (like missing imports) are OK for syntax check
            print(f"   ✅ {name} tests have valid syntax (import error expected: {type(e).__name__})")

    return all_passed


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("Simple Integration Test Validation")
    print("=" * 60)

    results = []

    # Run each test
    results.append(("UnifiedRequest Creation", test_unified_request_creation()))
    results.append(("Provider Imports", test_provider_imports()))
    results.append(("API Format", test_api_format()))
    results.append(("Test Syntax", test_single_test_syntax()))

    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")

    print(f"\nTotal: {passed}/{total} passed")

    if passed == total:
        print("\n✅ All validation tests passed!")
        return 0
    else:
        print("\n❌ Some validation tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
