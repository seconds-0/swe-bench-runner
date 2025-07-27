#!/usr/bin/env python3
"""Integration test for all Phase 2 components."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from swebench_runner.providers import MockProvider, ProviderConfig
from swebench_runner.generation import (
    PatchGenerator,
    PromptBuilder,
    ResponseParser,
    TokenManager,
    PatchValidator,
    BatchProcessor,
    TemplateStyle
)

# Sample SWE-bench instance
SAMPLE_INSTANCE = {
    "instance_id": "test_instance_1",
    "repo": "test/repo",
    "problem_statement": "Fix a bug where the function returns None instead of an empty list",
    "base_commit": "abc123",
    "test_files": {
        "test_module.py": """
def test_function():
    result = buggy_function()
    assert result == [], f"Expected empty list, got {result}"
"""
    },
    "code_files": {
        "module.py": """
def buggy_function():
    # Bug: returns None instead of empty list
    if True:
        return None
    return []
"""
    }
}

# Mock response that contains a valid patch
MOCK_PATCH_RESPONSE = """I'll fix the bug where the function returns None instead of an empty list.

```diff
--- a/module.py
+++ b/module.py
@@ -1,5 +1,5 @@
 def buggy_function():
-    # Bug: returns None instead of empty list
+    # Fixed: return empty list instead of None
     if True:
-        return None
+        return []
     return []
```

This patch changes the return value from None to an empty list as expected by the test.
"""

async def test_integration():
    """Test all components working together."""
    print("🧪 Running integration test for Phase 2 components...")

    # 1. Setup MockProvider
    config = ProviderConfig(name="mock", model="mock-small")
    provider = MockProvider(config, mock_responses={}, mock_errors={})
    # Override the generate method to always return our patch response
    original_generate = provider.generate

    async def mock_generate_patch(prompt: str, **kwargs):
        # Always return our patch response for testing
        # Mock provider called for testing
        # Create a ModelResponse directly with our patch content
        from swebench_runner.providers.base import ModelResponse
        return ModelResponse(
            content=MOCK_PATCH_RESPONSE,
            model="mock-small",
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            cost=0.001,
            latency_ms=100,
            provider="mock",
            finish_reason="stop"
        )

    provider.generate = mock_generate_patch

    # 2. Setup components
    prompt_builder = PromptBuilder(template_style=TemplateStyle.DETAILED)
    response_parser = ResponseParser()
    token_manager = TokenManager()
    patch_validator = PatchValidator()

    # 3. Create PatchGenerator with response parser
    generator = PatchGenerator(
        provider=provider,
        response_parser=response_parser
    )

    print("✅ All components initialized")

    # 4. Test individual components

    # Test PromptBuilder
    context = prompt_builder.build_context(SAMPLE_INSTANCE)
    prompt = prompt_builder.build_prompt(context)
    print(f"✅ PromptBuilder: Generated {len(prompt)} character prompt")

    # Test TokenManager
    token_count = token_manager.count_tokens(prompt, "mock-small")
    print(f"✅ TokenManager: Counted {token_count} tokens")

    # Test ResponseParser
    parse_result = response_parser.extract_patch(MOCK_PATCH_RESPONSE)
    assert parse_result.patch is not None, "Failed to extract patch"
    print(f"✅ ResponseParser: Extracted patch with confidence {parse_result.confidence}")

    # Test PatchValidator
    validation_result = patch_validator.validate(parse_result.patch, SAMPLE_INSTANCE)
    assert validation_result.is_valid, f"Patch validation failed: {validation_result.issues}"
    print(f"✅ PatchValidator: Validated patch (score: {validation_result.score})")

    # 5. Test PatchGenerator integration
    result = await generator.generate_patch(SAMPLE_INSTANCE)
    assert result.success, f"Generation failed: {result.error}"
    assert result.patch is not None, "No patch generated"
    print(f"✅ PatchGenerator: Generated patch successfully (cost: ${result.cost:.4f})")

    # 6. Test BatchProcessor
    batch_processor = BatchProcessor(generator=generator, max_concurrent=2)
    instances = [SAMPLE_INSTANCE, {**SAMPLE_INSTANCE, "instance_id": "test_instance_2"}]

    # The mock provider is already set up to return our patch response for any prompt

    batch_result = await batch_processor.process_batch(instances)
    assert batch_result.stats.completed == 2, f"Expected 2 completed, got {batch_result.stats.completed}"
    assert batch_result.stats.failed == 0, f"Unexpected failures: {batch_result.failed}"
    print(f"✅ BatchProcessor: Processed {batch_result.stats.completed} instances")

    print("\n🎉 Integration test completed successfully!")
    print(f"📊 Final stats: {batch_result.stats.completed} completed, total cost: ${batch_result.stats.total_cost:.4f}")

    return True

if __name__ == "__main__":
    success = asyncio.run(test_integration())
    if success:
        print("\n✨ All Phase 2 components are working correctly together!")
        sys.exit(0)
    else:
        print("\n❌ Integration test failed!")
        sys.exit(1)
