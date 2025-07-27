"""Integration tests for ResponseParser with PatchGenerator."""

import asyncio

import pytest

from swebench_runner.generation import PatchGenerator, ResponseParser
from swebench_runner.providers import MockProvider
from swebench_runner.providers.base import ModelResponse
from swebench_runner.providers.config import ProviderConfig


class SimpleMockProvider(MockProvider):
    """Simple mock provider that returns configured responses in order."""

    def __init__(self, config: ProviderConfig, responses: list):
        super().__init__(config)
        self.responses = responses
        self.response_index = 0

    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Return the next response in the list."""
        if self.response_index >= len(self.responses):
            response = "No more responses configured"
        else:
            response = self.responses[self.response_index]
            self.response_index += 1

        await asyncio.sleep(self.response_delay)

        return ModelResponse(
            content=response,
            model=self.config.model or self.default_model,
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
            cost=0.001,
            latency_ms=int(self.response_delay * 1000),
            provider=self.name,
            finish_reason="stop",
            raw_response={"mock": True}
        )


class TestResponseParserIntegration:
    """Test ResponseParser integration with PatchGenerator."""

    @pytest.mark.asyncio
    async def test_patch_generator_uses_response_parser(self):
        """Test that PatchGenerator correctly uses ResponseParser."""
        # Configure mock to return a response with a fenced diff
        test_response = """
Here's the fix for your issue:

```diff
--- a/src/main.py
+++ b/src/main.py
@@ -10,4 +10,4 @@
 def hello():
-    print("Hello")
+    print("Hello, World!")
    return True
```

This should resolve the problem!
"""

        # Set up mock provider
        config = ProviderConfig(name="mock", model="mock-small")
        provider = SimpleMockProvider(config, [test_response])

        # Create a custom response parser with specific settings
        parser = ResponseParser(
            auto_fix_common_issues=True,
            min_confidence=0.5
        )

        # Create generator with custom parser
        generator = PatchGenerator(
            provider=provider,
            response_parser=parser
        )

        # Test instance
        instance = {
            "instance_id": "test-123",
            "repo": "test/repo",
            "problem_statement": "Fix the greeting message"
        }

        # Generate patch
        result = await generator.generate_patch(instance)

        # Verify results
        assert result.success is True
        assert result.patch is not None
        assert "Hello, World!" in result.patch
        assert result.metadata["format_detected"] == "fenced_diff"
        assert result.metadata["extraction_confidence"] >= 0.8

    @pytest.mark.asyncio
    async def test_handles_various_formats(self):
        """Test handling of various patch formats."""
        # Test different response formats
        test_cases = [
            # Git diff format
            ("""
diff --git a/test.py b/test.py
index abc123..def456 100644
--- a/test.py
+++ b/test.py
@@ -1,3 +1,3 @@
 def test():
-    return False
+    return True
""", "git_diff"),

            # File blocks format
            ("""
<file>src/utils.py</file>
<old>
def calculate(x):
    return x * 2
</old>
<new>
def calculate(x):
    return x * 3
</new>
""", "file_blocks"),

            # Search/replace format
            ("""
In file config.py:

SEARCH:
DEBUG = False
REPLACE:
DEBUG = True
""", "search_replace"),
        ]

        for response_text, expected_format in test_cases:
            config = ProviderConfig(name="mock", model="mock-small")
            provider = SimpleMockProvider(config, [response_text])
            generator = PatchGenerator(provider=provider)

            instance = {
                "instance_id": f"test-{expected_format}",
                "repo": "test/repo",
                "problem_statement": "Test problem"
            }

            result = await generator.generate_patch(instance)

            assert result.success is True
            assert result.patch is not None
            assert result.metadata["format_detected"] == expected_format

    @pytest.mark.asyncio
    async def test_auto_fix_malformed_patches(self):
        """Test that auto-fix corrects common issues."""
        # Response with wrong line counts in hunk header
        test_response = """
--- a/file.py
+++ b/file.py
@@ -1,1 +1,1 @@
-old_line1
-old_line2
+new_line1
+new_line2
+new_line3
"""

        config = ProviderConfig(name="mock", model="mock-small")
        provider = SimpleMockProvider(config, [test_response])

        # Parser with auto-fix enabled
        parser = ResponseParser(auto_fix_common_issues=True)
        generator = PatchGenerator(provider=provider, response_parser=parser)

        instance = {
            "instance_id": "test-autofix",
            "repo": "test/repo",
            "problem_statement": "Test auto-fix"
        }

        result = await generator.generate_patch(instance)

        assert result.success is True
        assert result.patch is not None
        # Check that the hunk header was fixed
        assert "@@ -1,2 +1,3 @@" in result.patch

    @pytest.mark.asyncio
    async def test_handles_parse_failures_gracefully(self):
        """Test graceful handling of unparseable responses."""
        # First response is unparseable, second has valid patch
        responses = [
            "This is just regular text without any patch.",
            """
--- a/fix.py
+++ b/fix.py
@@ -1,1 +1,1 @@
-broken = True
+broken = False
"""
        ]

        config = ProviderConfig(name="mock", model="mock-small")
        provider = SimpleMockProvider(config, responses)
        generator = PatchGenerator(provider=provider, max_retries=2)

        instance = {
            "instance_id": "test-retry",
            "repo": "test/repo",
            "problem_statement": "Fix the broken flag"
        }

        result = await generator.generate_patch(instance)

        # Should succeed on retry with higher temperature
        assert result.success is True
        assert result.attempts == 2
        assert result.patch is not None
        assert "broken = False" in result.patch
