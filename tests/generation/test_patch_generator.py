"""Tests for the PatchGenerator class."""


import pytest

from swebench_runner.generation.patch_generator import PatchGenerator
from swebench_runner.providers import ModelProvider, ModelResponse, ProviderConfig
from swebench_runner.providers.exceptions import (
    ProviderRateLimitError,
    ProviderTokenLimitError,
)

# Use anyio which is already installed - configure backend per test


class MockProvider(ModelProvider):
    """Mock provider for testing."""

    name = "mock"
    description = "Mock provider for testing"
    requires_api_key = False

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.generate_responses = []
        self.generate_call_count = 0

    def _init_capabilities(self):
        from swebench_runner.providers import ProviderCapabilities
        return ProviderCapabilities(max_context_length=4096)

    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Mock generate that returns predefined responses."""
        if self.generate_call_count < len(self.generate_responses):
            response = self.generate_responses[self.generate_call_count]
            self.generate_call_count += 1
            if isinstance(response, Exception):
                raise response
            return response
        else:
            return ModelResponse(
                content="No more responses configured",
                model="mock",
                cost=0.001,
            )

    def estimate_cost(self, prompt_tokens: int, max_tokens: int) -> float:
        """Mock cost estimation."""
        return 0.001

    @classmethod
    def _config_from_env(
        cls, env_vars: dict[str, str], model: str = None
    ) -> ProviderConfig:
        """Mock config from env."""
        return ProviderConfig(name="mock", model=model or "mock-model")


@pytest.fixture
def mock_provider():
    """Create a mock provider for testing."""
    config = ProviderConfig(name="mock", model="mock-model")
    return MockProvider(config)


@pytest.fixture
def patch_generator(mock_provider):
    """Create a PatchGenerator with mock provider."""
    return PatchGenerator(mock_provider)


@pytest.fixture
def sample_instance():
    """Create a sample SWE-bench instance."""
    return {
        "instance_id": "test-123",
        "repo": "test/repo",
        "problem_statement": "Fix the bug in the code",
        "hints_text": "Look at the function foo()",
    }


@pytest.mark.asyncio
async def test_generate_patch_success(patch_generator, mock_provider, sample_instance):
    """Test successful patch generation."""
    # Configure mock to return a valid patch
    mock_provider.generate_responses = [
        ModelResponse(
            content="""Here's the fix:

```diff
--- a/file.py
+++ b/file.py
@@ -1,5 +1,5 @@
 def foo():
-    return "bug"
+    return "fixed"
```

This should fix the issue.""",
            model="mock-model",
            cost=0.002,
            finish_reason="stop",
        )
    ]

    result = await patch_generator.generate_patch(sample_instance)

    assert result.success is True
    assert result.instance_id == "test-123"
    assert result.model == "mock-model"
    assert result.attempts == 1
    assert result.cost == 0.002
    assert result.patch is not None
    assert "--- a/file.py" in result.patch
    assert "+    return \"fixed\"" in result.patch


@pytest.mark.asyncio
async def test_generate_patch_retry_on_parse_failure(
    patch_generator, mock_provider, sample_instance
):
    """Test retry with temperature adjustment on parse failure."""
    # First response has no valid patch, second one does
    mock_provider.generate_responses = [
        ModelResponse(
            content="Sorry, I cannot generate a patch.",
            model="mock-model",
            cost=0.001,
        ),
        ModelResponse(
            content="""diff --git a/file.py b/file.py
--- a/file.py
+++ b/file.py
@@ -1,3 +1,3 @@
 def foo():
-    return "bug"
+    return "fixed"
""",
            model="mock-model",
            cost=0.002,
        ),
    ]

    result = await patch_generator.generate_patch(sample_instance)

    assert result.success is True
    assert result.attempts == 2
    assert result.cost == 0.003
    assert result.metadata["final_temperature"] == 0.1  # Increased from 0.0


@pytest.mark.asyncio
async def test_generate_patch_token_limit_retry(
    patch_generator, mock_provider, sample_instance
):
    """Test retry with context reduction on token limit error."""
    # First attempt hits token limit, second succeeds
    mock_provider.generate_responses = [
        ProviderTokenLimitError("Token limit exceeded"),
        ModelResponse(
            content="""```diff
--- a/file.py
+++ b/file.py
@@ -1,2 +1,2 @@
-def broken():
+def fixed():
```""",
            model="mock-model",
            cost=0.002,
        ),
    ]

    result = await patch_generator.generate_patch(sample_instance)

    assert result.success is True
    assert result.attempts == 2
    assert result.truncated is True
    assert result.cost == 0.002


@pytest.mark.asyncio
async def test_generate_patch_rate_limit_retry(
    patch_generator, mock_provider, sample_instance
):
    """Test retry with wait on rate limit error."""
    # Configure short wait for testing
    patch_generator.rate_limit_wait_seconds = 0.1

    # First attempt hits rate limit, second succeeds
    mock_provider.generate_responses = [
        ProviderRateLimitError("Rate limit exceeded"),
        ModelResponse(
            content="""--- a/test.py
+++ b/test.py
@@ -1 +1 @@
-broken
+fixed""",
            model="mock-model",
            cost=0.001,
        ),
    ]

    result = await patch_generator.generate_patch(sample_instance)

    assert result.success is True
    assert result.attempts == 2


@pytest.mark.asyncio
async def test_generate_patch_all_attempts_fail(
    patch_generator, mock_provider, sample_instance
):
    """Test when all retry attempts fail."""
    # All attempts return invalid responses
    mock_provider.generate_responses = [
        ModelResponse(content="No patch here", model="mock-model", cost=0.001),
        ModelResponse(content="Still no patch", model="mock-model", cost=0.001),
        ModelResponse(content="Nope", model="mock-model", cost=0.001),
    ]

    result = await patch_generator.generate_patch(sample_instance)

    assert result.success is False
    assert result.attempts == 3
    assert result.cost == 0.003
    assert result.error == "Failed after 3 attempts"
    assert result.patch is None


@pytest.mark.asyncio
async def test_generate_batch(patch_generator, mock_provider):
    """Test batch generation with multiple instances."""
    instances = [
        {"instance_id": "test-1", "problem_statement": "Fix bug 1"},
        {"instance_id": "test-2", "problem_statement": "Fix bug 2"},
        {"instance_id": "test-3", "problem_statement": "Fix bug 3"},
    ]

    # Configure responses for all instances
    mock_provider.generate_responses = [
        ModelResponse(
            content=f"--- a/file{i}.py\n+++ b/file{i}.py\n@@ -1 +1 @@\n-bug\n+fixed",
            model="mock-model",
            cost=0.001,
        )
        for i in range(1, 4)
    ]

    results = await patch_generator.generate_batch(instances, concurrency=2)

    assert len(results) == 3
    assert all(r.success for r in results)
    assert [r.instance_id for r in results] == ["test-1", "test-2", "test-3"]
    assert sum(r.cost for r in results) == 0.003


@pytest.mark.asyncio
async def test_generate_batch_with_progress_callback(patch_generator, mock_provider):
    """Test batch generation with progress tracking."""
    instances = [
        {"instance_id": f"test-{i}", "problem_statement": f"Fix bug {i}"}
        for i in range(5)
    ]

    # Configure responses
    mock_provider.generate_responses = [
        ModelResponse(
            content=f"--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-bug{i}\n+fixed{i}",
            model="mock-model",
            cost=0.001,
        )
        for i in range(5)
    ]

    # Track progress calls
    progress_calls = []

    def progress_callback(instance_id: str, current: int, total: int):
        progress_calls.append((instance_id, current, total))

    results = await patch_generator.generate_batch(
        instances,
        concurrency=2,
        progress_callback=progress_callback
    )

    assert len(results) == 5
    assert all(r.success for r in results)
    # Should have 2 calls per instance (start and end)
    assert len(progress_calls) == 10


def test_build_basic_prompt(patch_generator):
    """Test basic prompt building."""
    instance = {
        "instance_id": "django-123",
        "repo": "django/django",
        "problem_statement": "The admin interface crashes when...",
        "hints_text": "Check the admin.py file",
        "test_patch": "def test_admin():\n    assert False",
    }

    prompt = patch_generator._build_basic_prompt(instance)

    assert "django/django" in prompt
    assert "django-123" in prompt
    assert "The admin interface crashes" in prompt
    assert "Check the admin.py file" in prompt
    assert "def test_admin()" in prompt
    assert "unified diff format" in prompt


def test_extract_basic_patch_diff_block(patch_generator):
    """Test patch extraction from diff code block."""
    response = """Here's the fix:

```diff
--- a/src/main.py
+++ b/src/main.py
@@ -10,3 +10,3 @@
 def process():
-    return None
+    return "fixed"
```

This should resolve the issue."""

    # Use the response_parser to extract the patch
    result = patch_generator.response_parser.extract_patch(response)
    patch = result.patch

    assert patch is not None
    assert "--- a/src/main.py" in patch
    assert "+    return \"fixed\"" in patch
    assert "```" not in patch


def test_extract_basic_patch_direct_diff(patch_generator):
    """Test patch extraction from direct diff format."""
    response = """diff --git a/test.py b/test.py
index 123..456 100644
--- a/test.py
+++ b/test.py
@@ -1,3 +1,3 @@
 def main():
-    print("broken")
+    print("fixed")
     return 0

That's the patch you need."""

    # Use the response_parser to extract the patch
    result = patch_generator.response_parser.extract_patch(response)
    patch = result.patch

    assert patch is not None
    # When patch starts with ---, diff --git line might not be included
    assert "--- a/test.py" in patch
    assert "+++ b/test.py" in patch
    assert "+    print(\"fixed\")" in patch
    assert "That's the patch" not in patch


def test_validate_patch_valid(patch_generator):
    """Test patch validation with valid patches."""
    valid_patches = [
        "--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new",
        (
            "diff --git a/src/main.py b/src/main.py\n--- a/src/main.py\n"
            "+++ b/src/main.py\n@@ -1,3 +1,3 @@\n-def foo():\n+def bar():"
        ),
    ]

    for patch in valid_patches:
        # Use the response_parser to validate the patch
        result = patch_generator.response_parser.validate_patch(patch)
        assert result.is_valid is True


def test_validate_patch_invalid(patch_generator):
    """Test patch validation with invalid patches."""
    invalid_patches = [
        "",
        "   ",
        "This is not a patch",
        "--- a/file.py",  # Missing actual changes
        "Just some random text with no diff markers",
    ]

    for patch in invalid_patches:
        # Use the response_parser to validate the patch
        result = patch_generator.response_parser.validate_patch(patch)
        assert result.is_valid is False


def test_reduce_context(patch_generator):
    """Test context reduction for token limit handling."""
    instance = {"instance_id": "test-123"}

    # Create a long prompt
    long_problem = "This is a very long problem statement. " * 100
    prompt = f"""Fix the following issue in test/repo:

Problem: {long_problem}

Instance ID: test-123

Additional Context: Some extra context here

The following test was written to reproduce this issue:
```diff
def test_something():
    assert True
```

Generate a patch in unified diff format."""

    reduced = patch_generator._reduce_context(prompt, instance)

    assert len(reduced) < len(prompt)
    # Either truncated message or optional sections removed
    assert ("truncated" in reduced) or ("Additional Context" not in reduced)
    assert "Instance ID: test-123" in reduced
    assert "Generate a patch" in reduced
