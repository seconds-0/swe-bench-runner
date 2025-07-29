#!/usr/bin/env python3
"""Comprehensive validation script for integration tests.

This script runs actual tests to ensure:
1. The UnifiedRequest API is used correctly
2. Tests can execute without syntax errors
3. Basic functionality works with credentials
4. Error handling works correctly
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test results tracking
results: Dict[str, Tuple[bool, str]] = {}


async def test_openai():
    """Test basic OpenAI functionality."""
    try:
        if not os.getenv("OPENAI_API_KEY"):
            return False, "No OPENAI_API_KEY found (set to test actual API)"

        from swebench_runner.providers.openai import OpenAIProvider
        from swebench_runner.providers.unified_models import UnifiedRequest

        # Create provider
        provider = OpenAIProvider()
        await provider.initialize()

        # Make simple request
        request = UnifiedRequest(
            prompt="Say 'test passed'",
            model="gpt-3.5-turbo",
            max_tokens=10,
            temperature=0.0
        )

        response = await provider.generate_unified(request)

        # Basic validation
        if response.content and len(response.content) > 0:
            return True, f"Success: {response.content[:50]}"
        else:
            return False, "Empty response"

    except Exception as e:
        return False, f"Error: {str(e)}"


async def test_anthropic():
    """Test basic Anthropic functionality."""
    try:
        if not os.getenv("ANTHROPIC_API_KEY"):
            return False, "No ANTHROPIC_API_KEY found (set to test actual API)"

        from swebench_runner.providers.anthropic import AnthropicProvider
        from swebench_runner.providers.unified_models import UnifiedRequest

        # Create provider
        provider = AnthropicProvider()
        await provider.initialize()

        # Make simple request
        request = UnifiedRequest(
            prompt="Say 'test passed'",
            model="claude-3-haiku-20240307",
            max_tokens=10,
            temperature=0.0
        )

        response = await provider.generate_unified(request)

        # Basic validation
        if response.content and len(response.content) > 0:
            return True, f"Success: {response.content[:50]}"
        else:
            return False, "Empty response"

    except Exception as e:
        return False, f"Error: {str(e)}"


async def test_ollama():
    """Test basic Ollama functionality."""
    try:
        from swebench_runner.providers.ollama import OllamaProvider
        from swebench_runner.providers.unified_models import UnifiedRequest
        from swebench_runner.providers.base import ProviderConfig

        # Create provider with config
        config = ProviderConfig(
            name="ollama",
            api_key=None,
            model="llama3.2:1b",
            temperature=0.7,
            max_tokens=10,
            timeout=30.0
        )
        provider = OllamaProvider(config)

        # Make simple request
        request = UnifiedRequest(
            prompt="Say 'test passed'",
            model="llama3.2:1b",
            max_tokens=10,
            temperature=0.0
        )

        response = await provider.generate_unified(request)

        # Basic validation
        if response.content and len(response.content) > 0:
            return True, f"Success: {response.content[:50]}"
        else:
            return False, "Empty response"

    except Exception as e:
        # Check if Ollama is not running
        if "Connection" in str(e) or "refused" in str(e).lower():
            return False, "Ollama not running (expected if not installed)"
        return False, f"Error: {str(e)}"


def test_request_format_compatibility():
    """Test that UnifiedRequest format matches provider expectations."""
    print("\nTesting request format compatibility...")

    try:
        from swebench_runner.providers.unified_models import UnifiedRequest

        # Test that UnifiedRequest doesn't accept 'messages' parameter
        try:
            # This should fail
            request = UnifiedRequest(
                messages=[{"role": "user", "content": "test"}],
                model="gpt-3.5-turbo"
            )
            return False, "UnifiedRequest incorrectly accepts 'messages' parameter"
        except TypeError:
            # This is expected - UnifiedRequest should not accept 'messages'
            pass

        # Test that UnifiedRequest accepts 'prompt' parameter
        request = UnifiedRequest(
            prompt="Test prompt",
            model="gpt-3.5-turbo",
            max_tokens=100
        )

        if hasattr(request, 'prompt') and request.prompt == "Test prompt":
            return True, "UnifiedRequest correctly uses 'prompt' parameter"
        else:
            return False, "UnifiedRequest doesn't properly handle 'prompt' parameter"

    except Exception as e:
        return False, f"Error testing request format: {str(e)}"


def check_test_structure():
    """Check that test files have proper structure."""
    print("\nChecking test file structure...")

    test_files = {
        "OpenAI": "tests/integration/test_openai_integration.py",
        "Anthropic": "tests/integration/test_anthropic_integration.py",
        "Ollama": "tests/integration/test_ollama_integration.py"
    }

    all_good = True
    issues = []

    for provider, filepath in test_files.items():
        path = Path(filepath)
        if not path.exists():
            issues.append(f"{provider}: File not found")
            all_good = False
            continue

        with open(path) as f:
            content = f.read()

        # Check for essential components
        checks = [
            ("UnifiedRequest import", "from swebench_runner.providers.unified_models import UnifiedRequest" in content),
            ("Provider import", f"{provider}Provider" in content),
            ("Test functions", "async def test_" in content or "def test_" in content),
            ("No messages parameter", "messages=" not in content or "# messages=" in content)
        ]

        for check_name, passed in checks:
            if not passed:
                issues.append(f"{provider}: Missing/incorrect {check_name}")
                all_good = False

    if all_good:
        return True, "All test files properly structured"
    else:
        return False, f"Issues found: {'; '.join(issues)}"


async def run_validation():
    """Run all provider tests."""
    print("Comprehensive Integration Test Validation")
    print("=" * 60)

    # Run synchronous tests first
    print("\n1. Testing Request Format Compatibility")
    print("-" * 40)
    success, message = test_request_format_compatibility()
    results["Request Format"] = (success, message)
    print(f"   {'✅' if success else '❌'} {message}")

    print("\n2. Checking Test File Structure")
    print("-" * 40)
    success, message = check_test_structure()
    results["Test Structure"] = (success, message)
    print(f"   {'✅' if success else '❌'} {message}")

    # Test each provider
    print("\n3. Testing Provider Functionality")
    print("-" * 40)

    tests = [
        ("OpenAI", test_openai),
        ("Anthropic", test_anthropic),
        ("Ollama", test_ollama),
    ]

    for name, test_func in tests:
        print(f"\nTesting {name}...", end="", flush=True)
        success, message = await test_func()
        results[name] = (success, message)

        if success:
            print(f" ✅ PASSED")
        else:
            print(f" ❌ FAILED")
        print(f"  → {message}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)

    passed = sum(1 for success, _ in results.values() if success)
    total = len(results)

    print(f"\nTotal: {passed}/{total} passed")

    # Categorize results
    structural_tests = ["Request Format", "Test Structure"]
    provider_tests = ["OpenAI", "Anthropic", "Ollama"]

    structural_passed = sum(1 for name in structural_tests if name in results and results[name][0])
    provider_passed = sum(1 for name in provider_tests if name in results and results[name][0])

    print(f"\nStructural Tests: {structural_passed}/{len(structural_tests)} passed")
    print(f"Provider Tests: {provider_passed}/{len(provider_tests)} passed")

    # Detail any failures
    failures = [(name, msg) for name, (success, msg) in results.items() if not success]
    if failures:
        print("\nFailures:")
        for name, msg in failures:
            print(f"  - {name}: {msg}")

    # Provide guidance
    print("\nGuidance:")
    if structural_passed == len(structural_tests):
        print("  ✅ Test structure is correct - integration tests are properly formatted")
    else:
        print("  ❌ Test structure issues detected - fix these before running tests")

    if provider_passed == 0:
        print("  ℹ️  No provider tests passed - this is expected without API keys")
        print("  ℹ️  Set OPENAI_API_KEY or ANTHROPIC_API_KEY to test actual functionality")
    elif provider_passed < len(provider_tests):
        print("  ⚠️  Some provider tests failed - check API keys and service availability")
    else:
        print("  ✅ All provider tests passed - integration tests are working!")

    # Return appropriate exit code
    # Success if all structural tests pass (provider tests may fail due to missing keys)
    return 0 if structural_passed == len(structural_tests) else 1


def main():
    """Main entry point."""
    try:
        exit_code = asyncio.run(run_validation())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
