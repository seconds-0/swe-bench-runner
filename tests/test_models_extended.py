"""Extended tests for models module to improve coverage."""

import pytest

from swebench_runner.models import EvaluationResult, Patch


class TestPatchModel:
    """Test Patch model behavior."""

    def test_patch_creation_minimal(self):
        """Test creating patch with minimal fields."""
        patch = Patch(instance_id="test-123", patch="diff --git a/test.py b/test.py\n+fix")

        assert patch.instance_id == "test-123"
        assert patch.patch == "diff --git a/test.py b/test.py\n+fix"

    def test_patch_creation_with_valid_diff(self):
        """Test creating patch with valid diff format."""
        patch_content = """diff --git a/django/views.py b/django/views.py
index 1234567..abcdefg 100644
--- a/django/views.py
+++ b/django/views.py
@@ -10,7 +10,7 @@ def process_request(request):
-    return HttpResponse("old")
+    return HttpResponse("new")"""

        patch = Patch(
            instance_id="django__django-12345",
            patch=patch_content
        )

        assert patch.instance_id == "django__django-12345"
        assert "diff --git" in patch.patch
        assert "django/views.py" in patch.patch

    def test_patch_validation_small_patch(self):
        """Test validation with small patch."""
        patch = Patch(instance_id="test", patch="diff --git a/test.py b/test.py\n+small fix")

        # Should not raise
        patch.validate(max_size_mb=1)

    def test_patch_validation_exact_limit(self):
        """Test validation at exact size limit."""
        # Create patch exactly 1MB
        # Create patch exactly 1MB with valid diff header
        large_content = "diff --git a/test.py b/test.py\n+" + "x" * (1024 * 1024 - 32)
        patch = Patch(instance_id="test", patch=large_content)

        # Should not raise at exact limit
        patch.validate(max_size_mb=1)

    def test_patch_validation_custom_limit(self):
        """Test validation with custom size limit."""
        # Create 2MB patch
        # Create 2MB patch with valid diff header
        large_content = "diff --git a/test.py b/test.py\n+" + "x" * (2 * 1024 * 1024)
        patch = Patch(instance_id="test", patch=large_content)

        # Should not raise with 3MB limit
        patch.validate(max_size_mb=3)

        # Should raise with 1MB limit
        with pytest.raises(ValueError) as exc_info:
            patch.validate(max_size_mb=1)

        assert "PATCH_TOO_LARGE" in str(exc_info.value)

    def test_patch_size_calculation(self):
        """Test patch size calculation."""
        # Test with unicode characters (each emoji is multiple bytes)
        patch = Patch(instance_id="test", patch="diff --git a/test.py b/test.py\n+Hello 👋 World 🌍")

        # Should handle unicode properly
        patch.validate(max_size_mb=1)

    def test_patch_empty_validation(self):
        """Test validation with empty patch."""
        patch = Patch(instance_id="test", patch="")

        # Empty patch should fail validation
        with pytest.raises(ValueError) as exc_info:
            patch.validate(max_size_mb=1)

        assert "patch cannot be empty" in str(exc_info.value)

    def test_patch_validation_empty_instance_id(self):
        """Test validation with empty instance_id."""
        patch = Patch(instance_id="", patch="diff --git a/test.py b/test.py\n+fix")

        with pytest.raises(ValueError) as exc_info:
            patch.validate()

        assert "instance_id cannot be empty" in str(exc_info.value)

    def test_patch_validation_invalid_format(self):
        """Test validation with invalid patch format."""
        patch = Patch(instance_id="test", patch="not a valid diff")

        with pytest.raises(ValueError) as exc_info:
            patch.validate()

        assert "patch must be in unified diff format" in str(exc_info.value)

    def test_patch_validation_binary_content(self):
        """Test validation with binary file patch."""
        patch = Patch(
            instance_id="test",
            patch="""diff --git a/image.png b/image.png
GIT binary patch
literal 12345
Binary content here..."""
        )

        with pytest.raises(ValueError) as exc_info:
            patch.validate()

        assert "binary files which are not allowed" in str(exc_info.value)


class TestEvaluationResult:
    """Test EvaluationResult model behavior."""

    def test_evaluation_result_success(self):
        """Test creating successful evaluation result."""
        result = EvaluationResult(
            instance_id="test-123",
            passed=True,
            error=None
        )

        assert result.instance_id == "test-123"
        assert result.passed is True
        assert result.error is None

    def test_evaluation_result_failure(self):
        """Test creating failed evaluation result."""
        result = EvaluationResult(
            instance_id="test-456",
            passed=False,
            error="Tests failed: 2 errors"
        )

        assert result.instance_id == "test-456"
        assert result.passed is False
        assert result.error == "Tests failed: 2 errors"

    def test_evaluation_result_with_long_error(self):
        """Test evaluation result with long error message."""
        long_error = "x" * 1000  # Long error message
        result = EvaluationResult(
            instance_id="test-789",
            passed=False,
            error=long_error
        )

        assert result.error == long_error

    def test_evaluation_result_equality(self):
        """Test evaluation result equality."""
        result1 = EvaluationResult("test", True, None)
        result2 = EvaluationResult("test", True, None)
        result3 = EvaluationResult("test", False, "error")

        # Test inequality
        assert result1 != result3  # Different values

        # dataclasses use value equality by default
        assert result1 == result2  # Same values
        assert result1.instance_id == result2.instance_id
        assert result1.passed == result2.passed

    def test_evaluation_result_empty_instance_id(self):
        """Test creating evaluation result with empty instance_id."""
        with pytest.raises(ValueError) as exc_info:
            EvaluationResult(
                instance_id="",
                passed=True,
                error=None
            )

        assert "instance_id cannot be empty" in str(exc_info.value)
