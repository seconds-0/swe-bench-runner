#!/usr/bin/env python3
"""Demo script showing how to use the PromptBuilder."""

from swebench_runner.generation.prompt_builder import (
    PromptBuilder,
    TemplateStyle,
)


def main():
    """Demonstrate PromptBuilder usage."""
    # Example SWE-bench instance
    instance = {
        "instance_id": "django-12345",
        "repo": "django/django",
        "base_commit": "abc123def456",
        "problem_statement": """
When using Django's ORM with PostgreSQL, the `__in` lookup with an empty list
causes a database error. The query should return an empty queryset instead.

Example:
```python
User.objects.filter(id__in=[])  # This raises an error
```

Expected behavior: Should return an empty queryset.
Actual behavior: Raises PostgreSQL syntax error.
""",
        "test_patch": """
diff --git a/tests/test_lookups.py b/tests/test_lookups.py
+def test_empty_in_lookup():
+    # Test that empty __in lookup returns empty queryset
+    result = User.objects.filter(id__in=[])
+    assert result.count() == 0
""",
        "hints_text": "The issue is in the SQL generation for the IN clause",
    }

    print("=== PromptBuilder Demo ===\n")

    # 1. Detailed prompt (default)
    print("1. DETAILED Template:")
    print("-" * 50)
    builder = PromptBuilder(template_style=TemplateStyle.DETAILED)
    context = builder.build_context(instance)
    prompt = builder.build_prompt(context)
    print(prompt[:500] + "...\n")

    # 2. Concise prompt
    print("2. CONCISE Template:")
    print("-" * 50)
    builder = PromptBuilder(template_style=TemplateStyle.CONCISE)
    context = builder.build_context(instance)
    prompt = builder.build_prompt(context)
    print(prompt + "\n")

    # 3. Chain of Thought prompt
    print("3. CHAIN OF THOUGHT Template:")
    print("-" * 50)
    builder = PromptBuilder(template_style=TemplateStyle.COT)
    context = builder.build_context(instance)
    prompt = builder.build_prompt(context)
    print(prompt[:600] + "...\n")

    # 4. Minimal prompt
    print("4. MINIMAL Template:")
    print("-" * 50)
    builder = PromptBuilder(template_style=TemplateStyle.MINIMAL)
    context = builder.build_context(instance)
    prompt = builder.build_prompt(context)
    print(prompt + "\n")

    # 5. Demonstrate context reduction
    print("5. Context Reduction Demo:")
    print("-" * 50)
    builder = PromptBuilder()

    # Create a large instance
    large_instance = instance.copy()
    large_instance["code_files"] = {
        f"file{i}.py": f"# File {i}\n" + "x = 1\n" * 100
        for i in range(5)
    }

    context = builder.build_context(large_instance)
    original_length = context.total_length()
    print(f"Original context length: {original_length} characters")

    reduced = builder.reduce_context(context, reduction_factor=0.5)
    reduced_length = reduced.total_length()
    print(f"Reduced context length: {reduced_length} characters")
    print(f"Reduction: {(1 - reduced_length/original_length)*100:.1f}%\n")

    # 6. Retry prompt demo
    print("6. Retry Prompt Demo:")
    print("-" * 50)
    original_prompt = "Fix the bug in Django ORM"
    failed_response = "I understand the issue. The problem is in the query generation..."

    retry_prompt = builder.build_retry_prompt(
        original_prompt,
        failed_response,
        error_reason="parse_failure"
    )
    print(retry_prompt[:400] + "...\n")

    print("=== Demo Complete ===")


if __name__ == "__main__":
    main()
