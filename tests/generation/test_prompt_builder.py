"""Test the PromptBuilder class."""

import pytest

from swebench_runner.generation.prompt_builder import (
    PromptBuilder,
    PromptContext,
    TemplateStyle,
)


class TestPromptContext:
    """Test the PromptContext dataclass."""

    def test_empty_context(self):
        """Test empty context initialization."""
        context = PromptContext()
        assert context.system_prompt == ""
        assert context.user_prompt == ""
        assert context.code_context == {}
        assert context.test_context == {}
        assert context.problem_statement == ""
        assert context.instance_id == ""
        assert context.repo_name == ""
        assert context.base_commit == ""
        assert context.hints == []
        assert context.metadata == {}

    def test_total_length(self):
        """Test total length calculation."""
        context = PromptContext(
            system_prompt="System",
            user_prompt="User",
            problem_statement="Problem",
            code_context={"file1.py": "code1", "file2.py": "code2"},
            test_context={"test1.py": "test1"},
            hints=["hint1", "hint2"]
        )
        # System(6) + User(4) + Problem(7) + code1(5) + code2(5) + test1(5) + hint1(5) + hint2(5) = 42
        assert context.total_length() == 42


class TestPromptBuilder:
    """Test the PromptBuilder class."""

    @pytest.fixture
    def basic_instance(self):
        """Create a basic SWE-bench instance."""
        return {
            "instance_id": "test-123",
            "repo": "user/repo",
            "base_commit": "abc123",
            "problem_statement": "There is a bug in the function that needs fixing.",
            "test_patch": "diff --git a/test.py b/test.py\n+def test_bug():\n+    assert False",
            "hints_text": "Check the validation logic"
        }

    @pytest.fixture
    def complex_instance(self):
        """Create a complex SWE-bench instance with multiple fields."""
        return {
            "instance_id": "complex-456",
            "repo": "org/project",
            "base_commit": "def456",
            "problem_statement": "A complex bug involving multiple files and edge cases.",
            "test_patch": "diff --git a/test.py b/test.py\n+def test_complex():\n+    assert something",
            "hints_text": "Look at the error handling",
            "hints": ["Additional hint 1", "Additional hint 2"],
            "code_files": {
                "main.py": "def main():\n    return 42",
                "utils.py": "def helper():\n    pass"
            },
            "test_files": {
                "test_main.py": "def test_main():\n    assert main() == 42"
            },
            "created_at": "2024-01-01",
            "version": "1.0"
        }

    def test_init_defaults(self):
        """Test default initialization."""
        builder = PromptBuilder()
        assert builder.template_style == TemplateStyle.DETAILED
        assert builder.include_test_info is True
        assert builder.max_code_context_lines == 500
        assert builder.max_test_lines == 200
        assert builder.prioritize_recent_changes is True
        assert builder.include_hints is True
        assert builder.system_prompt_override is None

    def test_init_custom(self):
        """Test custom initialization."""
        builder = PromptBuilder(
            template_style=TemplateStyle.CONCISE,
            include_test_info=False,
            max_code_context_lines=100,
            max_test_lines=50,
            prioritize_recent_changes=False,
            include_hints=False,
            system_prompt_override="Custom system prompt"
        )
        assert builder.template_style == TemplateStyle.CONCISE
        assert builder.include_test_info is False
        assert builder.max_code_context_lines == 100
        assert builder.max_test_lines == 50
        assert builder.prioritize_recent_changes is False
        assert builder.include_hints is False
        assert builder.system_prompt_override == "Custom system prompt"

    def test_build_context_basic(self, basic_instance):
        """Test building context from a basic instance."""
        builder = PromptBuilder()
        context = builder.build_context(basic_instance)

        assert context.instance_id == "test-123"
        assert context.repo_name == "user/repo"
        assert context.base_commit == "abc123"
        assert context.problem_statement == "There is a bug in the function that needs fixing."
        assert "test_patch" in context.test_context
        assert context.hints == ["Check the validation logic"]
        assert context.system_prompt != ""

    def test_build_context_no_hints(self, basic_instance):
        """Test building context without hints."""
        builder = PromptBuilder(include_hints=False)
        context = builder.build_context(basic_instance)

        assert context.hints == []

    def test_build_context_no_tests(self, basic_instance):
        """Test building context without test info."""
        builder = PromptBuilder(include_test_info=False)
        context = builder.build_context(basic_instance)

        assert context.test_context == {}

    def test_build_context_complex(self, complex_instance):
        """Test building context from a complex instance."""
        builder = PromptBuilder()
        context = builder.build_context(complex_instance)

        assert context.instance_id == "complex-456"
        assert len(context.code_context) == 2
        assert "main.py" in context.code_context
        assert "utils.py" in context.code_context
        assert len(context.test_context) == 2  # test_patch + test_main.py
        assert len(context.hints) == 3  # hints_text + 2 from hints list
        assert context.metadata["created_at"] == "2024-01-01"
        assert context.metadata["version"] == "1.0"

    def test_build_context_missing_fields(self):
        """Test building context with missing fields."""
        builder = PromptBuilder()
        context = builder.build_context({})

        assert context.instance_id == "unknown"
        assert context.repo_name == ""
        assert context.base_commit == ""
        assert context.problem_statement == ""

    def test_build_prompt_detailed(self, basic_instance):
        """Test building a detailed prompt."""
        builder = PromptBuilder(template_style=TemplateStyle.DETAILED)
        context = builder.build_context(basic_instance)
        prompt = builder.build_prompt(context)

        assert "## Task: Fix Bug in user/repo" in prompt
        assert "**Instance ID**: test-123" in prompt  # Fixed: Markdown bold format
        assert "**Base Commit**: abc123" in prompt  # Fixed: Markdown bold format
        assert "### Problem Description" in prompt
        assert "There is a bug in the function" in prompt
        assert "### Failing Tests" in prompt
        assert "### Instructions" in prompt
        assert "Return only the patch in unified diff format" in prompt

    def test_build_prompt_concise(self, basic_instance):
        """Test building a concise prompt."""
        builder = PromptBuilder(template_style=TemplateStyle.CONCISE)
        context = builder.build_context(basic_instance)
        prompt = builder.build_prompt(context)

        assert "Fix this bug in user/repo:" in prompt
        assert "There is a bug" in prompt
        assert "Tests failing:" in prompt
        assert "Return unified diff patch only." in prompt
        assert len(prompt) < 1000  # Should be concise

    def test_build_prompt_cot(self, basic_instance):
        """Test building a chain-of-thought prompt."""
        builder = PromptBuilder(template_style=TemplateStyle.COT)
        context = builder.build_context(basic_instance)
        prompt = builder.build_prompt(context)

        assert "Let's fix this bug step by step" in prompt
        assert "Step 1: Understand the problem" in prompt
        assert "Step 2: Analyze failing tests" in prompt
        assert "Step 5: Generate the patch" in prompt

    def test_build_prompt_minimal(self, basic_instance):
        """Test building a minimal prompt."""
        builder = PromptBuilder(template_style=TemplateStyle.MINIMAL)
        context = builder.build_context(basic_instance)
        prompt = builder.build_prompt(context)

        assert "Bug:" in prompt
        assert "Repo: user/repo" in prompt
        assert "Fix with unified diff patch." in prompt
        assert len(prompt) < 300  # Should be very minimal

    def test_build_prompt_with_override(self, basic_instance):
        """Test building prompt with template override."""
        builder = PromptBuilder(template_style=TemplateStyle.DETAILED)
        context = builder.build_context(basic_instance)

        # Override to use minimal template
        prompt = builder.build_prompt(context, template=TemplateStyle.MINIMAL)
        assert len(prompt) < 300  # Should use minimal template

    def test_reduce_context(self, complex_instance):
        """Test context reduction."""
        builder = PromptBuilder()
        context = builder.build_context(complex_instance)
        original_length = context.total_length()

        reduced = builder.reduce_context(context, reduction_factor=0.5)
        reduced_length = reduced.total_length()

        assert reduced_length < original_length
        assert reduced.instance_id == context.instance_id  # Should preserve key fields
        assert reduced.repo_name == context.repo_name
        assert len(reduced.hints) <= len(context.hints)
        assert len(reduced.code_context) <= len(context.code_context)

    def test_reduce_context_extreme(self, complex_instance):
        """Test extreme context reduction."""
        builder = PromptBuilder()
        context = builder.build_context(complex_instance)

        # Extreme reduction
        reduced = builder.reduce_context(context, reduction_factor=0.1)

        assert reduced.total_length() < context.total_length()  # Should be reduced
        assert reduced.instance_id == context.instance_id  # Should still preserve key fields

    def test_build_retry_prompt_parse_failure(self):
        """Test building retry prompt for parse failure."""
        builder = PromptBuilder()
        original = "Fix this bug"
        failed = "Here is my explanation of the fix..."

        retry = builder.build_retry_prompt(original, failed, "parse_failure")

        assert "could not be parsed as a valid unified diff" in retry
        assert "Please ensure your response:" in retry
        assert "Here was your previous response:" in retry
        assert "Original task:" in retry
        assert original in retry

    def test_build_retry_prompt_no_changes(self):
        """Test building retry prompt for empty patch."""
        builder = PromptBuilder()
        original = "Fix this bug"
        failed = "diff --git a/file.py b/file.py"

        retry = builder.build_retry_prompt(original, failed, "no_changes")

        assert "appears to be empty" in retry
        assert "Include actual changes" in retry

    def test_build_retry_prompt_generic(self):
        """Test building retry prompt for generic error."""
        builder = PromptBuilder()
        original = "Fix this bug"
        failed = "some response"

        retry = builder.build_retry_prompt(original, failed, "validation_error")

        assert "failed validation: validation_error" in retry
        assert "Original task:" in retry

    def test_truncate_content_basic(self):
        """Test basic content truncation."""
        builder = PromptBuilder()
        content = "\n".join([f"Line {i}" for i in range(100)])

        truncated = builder._truncate_content(content, max_lines=10)

        assert truncated.count('\n') <= 10
        assert "truncated" in truncated

    def test_truncate_content_preserve_errors(self):
        """Test truncation with error preservation."""
        builder = PromptBuilder()
        content = "\n".join([
            "Normal line 1",
            "Normal line 2",
            "ERROR: This is an error",
            "Normal line 3",
            "AssertionError: Test failed",
            "Normal line 4",
        ])

        truncated = builder._truncate_content(content, max_lines=3, preserve_errors=True)

        # Should prioritize keeping error lines
        assert "ERROR: This is an error" in truncated or "AssertionError" in truncated
        assert "truncated" in truncated  # Should indicate truncation

    def test_truncate_content_preserve_structure(self):
        """Test truncation with structure preservation."""
        builder = PromptBuilder()
        content = """import os
import sys

class MyClass:
    def __init__(self):
        self.x = 1
        self.y = 2
        self.z = 3

    def method(self):
        # Long implementation
        for i in range(100):
            print(i)
        return True

def standalone_function():
    pass
"""

        truncated = builder._truncate_content(content, max_lines=8, preserve_structure=True)

        # Should keep imports and at least some structure elements
        assert "import os" in truncated or "import sys" in truncated
        assert "class MyClass:" in truncated or "def " in truncated
        assert "truncated" in truncated  # Should indicate truncation

    def test_extract_error_context(self):
        """Test error context extraction."""
        builder = PromptBuilder()
        test_output = """
Running tests...
test_1 passed
test_2 FAILED
Traceback (most recent call last):
  File "test.py", line 10, in test_2
    assert value == expected
AssertionError: 5 != 10
test_3 passed
"""

        error_context = builder._extract_error_context(test_output)

        assert "FAILED" in error_context
        assert "Traceback" in error_context
        assert "AssertionError: 5 != 10" in error_context
        assert "test_1 passed" not in error_context  # Should not include passing tests

    def test_format_code_context(self):
        """Test code context formatting."""
        builder = PromptBuilder()
        code_files = {
            "main.py": "def main():\n    return 42",
            "utils.py": "def helper():\n    pass"
        }

        formatted = builder._format_code_context(code_files)

        assert "#### main.py" in formatted
        assert "```python" in formatted
        assert "def main():" in formatted
        assert "#### utils.py" in formatted
        assert "def helper():" in formatted

    def test_format_test_context(self):
        """Test test context formatting."""
        builder = PromptBuilder()
        test_files = {
            "test_patch": "diff --git a/test.py b/test.py\n+def test():\n+    pass",
            "test_main.py": "def test_main():\n    assert True"
        }

        formatted = builder._format_test_context(test_files)

        assert "#### Test Patch" in formatted
        assert "```diff" in formatted
        assert "#### test_main.py" in formatted
        assert "```python" in formatted

    def test_summarize_test_failures(self):
        """Test test failure summarization."""
        builder = PromptBuilder()
        test_context = {
            "test_1.py": "AssertionError: Expected 5 but got 10\nAnother error line",
            "test_2.py": "ValueError: Invalid input\nMore context",
            "test_3.py": "Test failed without clear error"
        }

        summary = builder._summarize_test_failures(test_context)

        # Check that summaries contain key information
        assert "test_1.py" in summary
        assert "AssertionError" in summary
        assert "test_2.py" in summary
        assert "ValueError" in summary
        assert "test_3.py" in summary

    def test_summarize_code_context(self):
        """Test code context summarization."""
        builder = PromptBuilder()
        code_files = {
            "main.py": "class MainClass:\n    def method1(self):\n        pass\n    def method2(self):\n        pass",
            "utils.py": "def helper1():\n    pass\ndef helper2():\n    pass"
        }

        summary = builder._summarize_code_context(code_files)

        # Check that summaries contain key information
        assert "main.py" in summary
        assert "class MainClass" in summary or "def method" in summary
        assert "utils.py" in summary
        assert "def helper" in summary

    def test_system_prompt_override(self):
        """Test system prompt override."""
        custom_prompt = "You are a specialized bug fixer."
        builder = PromptBuilder(system_prompt_override=custom_prompt)
        context = builder.build_context({})

        assert context.system_prompt == custom_prompt

    def test_all_template_styles(self, basic_instance):
        """Test that all template styles produce valid prompts."""
        for style in TemplateStyle:
            builder = PromptBuilder(template_style=style)
            context = builder.build_context(basic_instance)
            prompt = builder.build_prompt(context)

            assert prompt  # Should not be empty
            assert isinstance(prompt, str)
            assert len(prompt) > 10  # Should have some content
