# Task 3: Create Integration Test Validation Script

**Branch**: fix/integration-test-validation
**File**: `scripts/validate-integration-tests.py`
**Priority**: HIGH

## Objective
Create a simple validation script that runs one basic test from each provider to confirm the integration tests actually work after remediation.

## Requirements

1. **Simple and Direct**: No complex frameworks, just run tests
2. **Clear Output**: Show which providers pass/fail
3. **Graceful Handling**: Don't crash on missing credentials
4. **Exit Codes**: 0 for success, 1 for any failures

## Script Structure

```python
#!/usr/bin/env python3
"""Validate that integration tests can actually run.

This script runs a minimal test for each provider to ensure:
1. The UnifiedRequest API is used correctly
2. Tests can execute without syntax errors
3. Basic functionality works with credentials
"""

import asyncio
import os
import sys
from typing import Dict, List, Tuple

# Test results tracking
results: Dict[str, Tuple[bool, str]] = {}

async def test_openai():
    """Test basic OpenAI functionality."""
    try:
        if not os.getenv("OPENAI_API_KEY"):
            return False, "No OPENAI_API_KEY found"

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
            return False, "No ANTHROPIC_API_KEY found"

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
        if "Connection" in str(e):
            return False, "Ollama not running (expected)"
        return False, f"Error: {str(e)}"

async def run_validation():
    """Run all provider tests."""
    print("Integration Test Validation")
    print("=" * 50)

    # Test each provider
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
    print("\n" + "=" * 50)
    print("Summary:")
    passed = sum(1 for success, _ in results.values() if success)
    total = len(results)

    print(f"  Passed: {passed}/{total}")

    # Detail any failures
    failures = [(name, msg) for name, (success, msg) in results.items() if not success]
    if failures:
        print("\nFailures:")
        for name, msg in failures:
            print(f"  - {name}: {msg}")

    # Return appropriate exit code
    return 0 if passed == total else 1

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
        sys.exit(1)

if __name__ == "__main__":
    main()
```

## Additional Validation Script for Test Syntax

Create `scripts/check-integration-test-syntax.py`:

```python
#!/usr/bin/env python3
"""Quick syntax check for integration tests."""

import ast
import sys
from pathlib import Path

def check_test_file(filepath: Path) -> List[str]:
    """Check a test file for common API misuse."""
    errors = []

    with open(filepath) as f:
        content = f.read()

    # Parse AST
    tree = ast.parse(content)

    # Look for UnifiedRequest calls with 'messages' parameter
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if hasattr(node.func, 'id') and node.func.id == 'UnifiedRequest':
                for keyword in node.keywords:
                    if keyword.arg == 'messages':
                        errors.append(f"Line {keyword.lineno}: Found 'messages=' parameter (should be 'prompt=')")

    return errors

def main():
    """Check all integration test files."""
    test_dir = Path("tests/integration")
    files_to_check = [
        "test_openai_integration.py",
        "test_anthropic_integration.py",
        "test_ollama_integration.py"
    ]

    total_errors = 0

    for filename in files_to_check:
        filepath = test_dir / filename
        if filepath.exists():
            errors = check_test_file(filepath)
            if errors:
                print(f"\n{filename}:")
                for error in errors:
                    print(f"  {error}")
                total_errors += len(errors)
            else:
                print(f"{filename}: ✅ OK")

    if total_errors:
        print(f"\nFound {total_errors} API usage errors!")
        sys.exit(1)
    else:
        print("\nAll tests use correct API! ✅")
        sys.exit(0)

if __name__ == "__main__":
    main()
```

## Testing Instructions

1. Make the scripts executable:
   ```bash
   chmod +x scripts/validate-integration-tests.py
   chmod +x scripts/check-integration-test-syntax.py
   ```

2. Run syntax check first:
   ```bash
   ./scripts/check-integration-test-syntax.py
   ```

3. Run validation (requires at least one API key):
   ```bash
   ./scripts/validate-integration-tests.py
   ```

## Success Criteria

- Scripts run without syntax errors
- Syntax checker detects 'messages=' usage
- Validation script handles missing credentials gracefully
- Clear output showing pass/fail status
- Appropriate exit codes (0 for success, 1 for failure)

## Notes

- Keep scripts simple and focused
- Don't over-engineer - these are validation tools
- Ensure scripts work even without all API keys
- Make output clear and actionable
