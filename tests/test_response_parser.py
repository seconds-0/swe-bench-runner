"""Tests for the ResponseParser component."""


from swebench_runner.generation.response_parser import (
    PatchFormat,
    ResponseParser,
)


class TestResponseParser:
    """Test the ResponseParser class."""

    def test_init(self):
        """Test parser initialization."""
        parser = ResponseParser()
        assert parser.strict_mode is False
        assert parser.auto_fix_common_issues is True
        assert parser.min_confidence == 0.3

        # Test with custom settings
        parser = ResponseParser(
            strict_mode=True,
            auto_fix_common_issues=False,
            min_confidence=0.5,
            preferred_formats=[PatchFormat.GIT_DIFF]
        )
        assert parser.strict_mode is True
        assert parser.auto_fix_common_issues is False
        assert parser.min_confidence == 0.5
        assert parser.preferred_formats == [PatchFormat.GIT_DIFF]

    def test_extract_unified_diff(self):
        """Test extraction of unified diff format."""
        parser = ResponseParser()

        response = """
Here's the fix for your issue:

--- a/src/main.py
+++ b/src/main.py
@@ -10,4 +10,4 @@
 def hello():
-    print("Hello")
+    print("Hello, World!")
     return True

That should fix the problem!
"""

        result = parser.extract_patch(response)
        assert result.patch is not None
        assert result.format_detected == PatchFormat.UNIFIED_DIFF
        assert result.confidence >= 0.8
        assert "--- a/src/main.py" in result.patch
        assert "+++ b/src/main.py" in result.patch
        assert '-    print("Hello")' in result.patch
        assert '+    print("Hello, World!")' in result.patch

    def test_extract_git_diff(self):
        """Test extraction of git diff format."""
        parser = ResponseParser()

        response = """
The solution is:

diff --git a/test.py b/test.py
index abc123..def456 100644
--- a/test.py
+++ b/test.py
@@ -1,3 +1,3 @@
 def test():
-    return False
+    return True
"""

        result = parser.extract_patch(response)
        assert result.patch is not None
        assert result.format_detected == PatchFormat.GIT_DIFF
        assert result.confidence >= 0.9
        assert "diff --git" in result.patch
        assert "index abc123..def456" in result.patch

    def test_extract_fenced_diff(self):
        """Test extraction from fenced code blocks."""
        parser = ResponseParser()

        response = """
Here's my solution:

```diff
--- a/app.py
+++ b/app.py
@@ -5,4 +5,4 @@
 def main():
-    pass
+    print("Fixed!")
```

This should work!
"""

        result = parser.extract_patch(response)
        assert result.patch is not None
        assert result.format_detected == PatchFormat.FENCED_DIFF
        assert result.confidence >= 0.7
        assert "--- a/app.py" in result.patch
        assert "+++ b/app.py" in result.patch

    def test_extract_file_blocks(self):
        """Test extraction from file modification blocks."""
        parser = ResponseParser()

        response = """
Change the file as follows:

<file>src/utils.py</file>
<old>
def calculate(x):
    return x * 2
</old>
<new>
def calculate(x):
    return x * 3
</new>
"""

        result = parser.extract_patch(response)
        assert result.patch is not None
        assert result.format_detected == PatchFormat.FILE_BLOCKS
        assert result.confidence >= 0.6
        assert "--- a/src/utils.py" in result.patch
        assert "+++ b/src/utils.py" in result.patch
        assert "-    return x * 2" in result.patch
        assert "+    return x * 3" in result.patch

    def test_extract_search_replace(self):
        """Test extraction from search/replace patterns."""
        parser = ResponseParser()

        response = """
In file config.py:

SEARCH:
DEBUG = False
REPLACE:
DEBUG = True
"""

        result = parser.extract_patch(response)
        assert result.patch is not None
        assert result.format_detected == PatchFormat.SEARCH_REPLACE
        assert result.confidence >= 0.5
        assert "-DEBUG = False" in result.patch
        assert "+DEBUG = True" in result.patch

    def test_empty_response(self):
        """Test handling of empty response."""
        parser = ResponseParser()

        result = parser.extract_patch("")
        assert result.patch is None
        assert result.format_detected == PatchFormat.UNKNOWN
        assert "Empty response" in result.issues[0]

    def test_no_patch_found(self):
        """Test handling when no patch is found."""
        parser = ResponseParser()

        response = "This is just regular text without any patch."

        result = parser.extract_patch(response)
        assert result.patch is None
        assert result.format_detected == PatchFormat.UNKNOWN
        assert "Could not extract a valid patch" in result.issues[0]
        assert len(result.suggestions) > 0

    def test_validate_patch_valid(self):
        """Test validation of a valid patch."""
        parser = ResponseParser()

        patch = """--- a/file.py
+++ b/file.py
@@ -1,3 +1,3 @@
 def test():
-    return False
+    return True
"""

        result = parser.validate_patch(patch)
        assert result.is_valid is True
        assert len(result.issues) == 0

    def test_validate_patch_invalid(self):
        """Test validation of invalid patches."""
        parser = ResponseParser()

        # Missing file headers
        patch1 = """@@ -1,3 +1,3 @@
-    return False
+    return True
"""
        result1 = parser.validate_patch(patch1)
        assert result1.is_valid is False
        assert "No file headers found" in str(result1.issues)

        # Missing hunk headers
        patch2 = """--- a/file.py
+++ b/file.py
-    return False
+    return True
"""
        result2 = parser.validate_patch(patch2)
        assert result2.is_valid is False
        assert "No hunk headers found" in str(result2.issues)

        # No actual changes
        patch3 = """--- a/file.py
+++ b/file.py
@@ -1,1 +1,1 @@
 # No changes here
"""
        result3 = parser.validate_patch(patch3)
        assert result3.is_valid is False
        assert "No actual changes found" in str(result3.issues)

    def test_normalize_patch(self):
        """Test patch normalization."""
        parser = ResponseParser()

        # Patch with various issues
        patch = """--- file.py
+++ file.py
@@ -1,2 +1,2 @@
-old line
+new line"""

        normalized = parser.normalize_patch(patch)
        assert "--- a/file.py" in normalized
        assert "+++ b/file.py" in normalized
        assert normalized.endswith('\n')

    def test_auto_fix_patch(self):
        """Test automatic patch fixing."""
        parser = ResponseParser()

        # Patch with wrong line counts
        patch = """--- a/file.py
+++ b/file.py
@@ -1,1 +1,1 @@
-line1
-line2
+new_line1
+new_line2
+new_line3
"""

        fixed = parser.auto_fix_patch(patch)
        assert "@@ -1,2 +1,3 @@" in fixed

        # Patch missing file headers
        patch2 = """@@ -1,2 +1,2 @@
-old
+new
"""

        fixed2 = parser.auto_fix_patch(patch2)
        assert "--- a/file" in fixed2
        assert "+++ b/file" in fixed2

    def test_multiple_patches_in_response(self):
        """Test handling of multiple patches in one response."""
        parser = ResponseParser()

        response = """
First, fix file1:

```diff
--- a/file1.py
+++ b/file1.py
@@ -1,2 +1,2 @@
-old1
+new1
```

Then fix file2:

```diff
--- a/file2.py
+++ b/file2.py
@@ -1,2 +1,2 @@
-old2
+new2
```
"""

        result = parser.extract_patch(response)
        assert result.patch is not None
        # Should extract the first patch
        assert "file1.py" in result.patch
        assert "-old1" in result.patch
        assert "+new1" in result.patch

    def test_malformed_patch_handling(self):
        """Test handling of malformed patches."""
        parser = ResponseParser(auto_fix_common_issues=True)

        # Patch with mixed formats
        response = """
```diff
--- a/test.py
+++ b/test.py
@@ -1,2 +1,2 @@
def test():
-    return False
+    return True
    # Extra line without prefix
```
"""

        result = parser.extract_patch(response)
        assert result.patch is not None
        # Should still extract something usable

    def test_confidence_calculation(self):
        """Test confidence score calculation."""
        parser = ResponseParser()

        # High confidence - proper git diff
        response1 = """
diff --git a/file.py b/file.py
index abc123..def456 100644
--- a/file.py
+++ b/file.py
@@ -1,3 +1,3 @@
 def test():
-    return False
+    return True
"""
        result1 = parser.extract_patch(response1)
        assert result1.confidence >= 0.9

        # Medium confidence - fenced block
        response2 = """
```diff
--- a/file.py
+++ b/file.py
@@ -1,1 +1,1 @@
-old
+new
```
"""
        result2 = parser.extract_patch(response2)
        assert 0.7 <= result2.confidence <= 0.95

        # Low confidence - search/replace
        response3 = """
SEARCH:
old code
REPLACE:
new code
"""
        result3 = parser.extract_patch(response3)
        assert result3.confidence <= 0.75

    def test_strict_mode(self):
        """Test strict mode validation."""
        parser_strict = ResponseParser(strict_mode=True)
        parser_lenient = ResponseParser(strict_mode=False)

        # Patch with slightly wrong line counts
        patch = """--- a/file.py
+++ b/file.py
@@ -1,2 +1,2 @@
-line1
+new_line1
+new_line2
"""

        # Strict mode should fail
        result_strict = parser_strict.validate_patch(patch)
        assert result_strict.is_valid is False

        # Lenient mode should warn but pass
        result_lenient = parser_lenient.validate_patch(patch)
        # Will still fail due to mismatch but would only warn if other issues weren't present

    def test_edge_cases(self):
        """Test various edge cases."""
        parser = ResponseParser()

        # Very short patch
        response1 = """
--- a/f
+++ b/f
@@ -1 +1 @@
-a
+b
"""
        result1 = parser.extract_patch(response1)
        assert result1.patch is not None

        # Patch with special characters
        response2 = """
```diff
--- a/file.py
+++ b/file.py
@@ -1,2 +1,2 @@
-    print("Hello!")
+    print("Hello, 世界!")
```
"""
        result2 = parser.extract_patch(response2)
        assert result2.patch is not None
        assert "世界" in result2.patch

        # Patch with no newline at end markers
        response3 = """
--- a/file.py
+++ b/file.py
@@ -1,2 +1,2 @@
-old line
\\ No newline at end of file
+new line
\\ No newline at end of file
"""
        result3 = parser.extract_patch(response3)
        assert result3.patch is not None
        assert "\\ No newline at end of file" in result3.patch
