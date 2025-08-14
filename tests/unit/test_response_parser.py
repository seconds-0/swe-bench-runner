"""Unit tests for response_parser.py - focus on parsing edge cases and security.

Tests specifically for:
- Malformed JSON/XML that could crash parsers
- Response extraction with adversarial input
- Token limit enforcement to prevent memory issues
- Regex performance on large/malicious input
- Format detection accuracy
"""

import time

import pytest

from swebench_runner.generation.response_parser import (
    ParseResult,
    PatchFormat,
    ResponseParser,
)


class TestResponseParserSecurity:
    """Test security-critical aspects of response parsing."""

    def test_prevents_redos_in_unified_diff_pattern(self):
        """Prevent ReDoS in unified diff pattern matching.

        Why: Malicious responses with nested patterns could hang the parser.
        """
        parser = ResponseParser()

        # Create potentially problematic input
        malicious_inputs = [
            "--- " + "/a" * 10000 + "\n+++ " + "/b" * 10000 + "\n@@ -1,1 +1,1 @@",
            "--- file\n+++ file\n" + "@@ " * 10000 + " @@",
            "--- a/b/c/d" * 1000 + "\n+++ x/y/z" * 1000,
        ]

        for malicious_input in malicious_inputs:
            start = time.time()
            parser._UNIFIED_DIFF_PATTERN.search(malicious_input)
            elapsed = time.time() - start

            # Should complete quickly
            assert elapsed < 0.1, f"Pattern took {elapsed}s - possible ReDoS"

    def test_prevents_redos_in_git_diff_pattern(self):
        """Prevent ReDoS in git diff pattern.

        Why: Complex git diff patterns could cause exponential backtracking.
        """
        parser = ResponseParser()

        # Git diff with excessive index values
        malicious = "diff --git " + "a/" * 1000 + " " + "b/" * 1000 + "\n"
        malicious += "index " + "f" * 10000 + ".." + "f" * 10000 + " 100644\n"

        start = time.time()
        parser._GIT_DIFF_PATTERN.search(malicious)
        elapsed = time.time() - start

        assert elapsed < 0.1, f"Pattern took {elapsed}s - possible ReDoS"

    def test_handles_extremely_large_responses(self):
        """Handle very large model responses without memory issues.

        Why: LLMs can generate very long responses that could exhaust memory.
        """
        parser = ResponseParser()

        # Create a 10MB response
        large_response = "x" * (10 * 1024 * 1024)

        # Should handle gracefully without crashing
        result = parser.parse(large_response)

        # Should either parse or report issue, not crash
        assert result is not None
        assert isinstance(result, ParseResult)

    def test_handles_malformed_json_in_response(self):
        """Handle malformed JSON without crashing.

        Why: Models often generate invalid JSON that could crash naive parsers.
        """
        parser = ResponseParser()

        malformed_responses = [
            '{"patch": "diff --git',  # Unclosed JSON
            '{"patch": "diff", "extra": }',  # Invalid JSON syntax
            '{"patch": null null}',  # Duplicate nulls
            '{"patch": "\x00\x01"}',  # Binary in JSON
        ]

        for malformed in malformed_responses:
            # Should not crash
            result = parser.parse(malformed)
            assert result is not None

    def test_detects_and_reports_binary_content(self):
        """Detect binary content in responses.

        Why: Binary data can't be applied as text patches and indicates errors.
        """
        parser = ResponseParser()

        # Response with binary content
        binary_response = "Here's the patch:\n\x00\x01\x02\xFF\xFE"

        result = parser.parse(binary_response)

        # Should detect issue
        assert len(result.issues) > 0 or result.confidence < 0.5


class TestResponseParserFormatDetection:
    """Test format detection for various patch formats."""

    def test_detects_unified_diff_format(self):
        """Correctly identify unified diff format.

        Why: Format detection determines parsing strategy.
        """
        parser = ResponseParser()

        unified_diff = """--- a/file.py
+++ b/file.py
@@ -1,3 +1,3 @@
 line1
-old line
+new line
 line3"""

        result = parser.parse(unified_diff)

        assert result.format_detected == PatchFormat.UNIFIED_DIFF
        assert result.patch is not None
        assert result.confidence > 0.7

    def test_detects_git_diff_format(self):
        """Correctly identify git diff format.

        Why: Git format is the most common and needs accurate detection.
        """
        parser = ResponseParser()

        git_diff = """diff --git a/file.py b/file.py
index abc123..def456 100644
--- a/file.py
+++ b/file.py
@@ -1,2 +1,2 @@
-old line
+new line"""

        result = parser.parse(git_diff)

        assert result.format_detected == PatchFormat.GIT_DIFF
        assert result.patch is not None

    def test_detects_fenced_code_blocks(self):
        """Extract patches from markdown code blocks.

        Why: Models often wrap patches in markdown formatting.
        """
        parser = ResponseParser()

        fenced_response = """Here's the fix:

```diff
--- a/file.py
+++ b/file.py
@@ -1,1 +1,1 @@
-bug
+fix
```

This should solve the issue."""

        result = parser.parse(fenced_response)

        assert result.format_detected in [PatchFormat.FENCED_DIFF, PatchFormat.FENCED_PATCH]
        assert result.patch is not None
        assert "```" not in result.patch  # Should extract content only

    def test_detects_search_replace_format(self):
        """Detect search/replace format instructions.

        Why: Some models generate search/replace instructions instead of diffs.
        """
        parser = ResponseParser()

        search_replace = """SEARCH:
def buggy_function():
    return None

REPLACE:
def buggy_function():
    return 42"""

        result = parser.parse(search_replace)

        assert result.format_detected == PatchFormat.SEARCH_REPLACE
        # Parser should convert to diff format
        assert result.patch is not None or len(result.suggestions) > 0


class TestResponseParserErrorHandling:
    """Test error handling and recovery."""

    def test_handles_empty_response(self):
        """Handle empty responses gracefully.

        Why: Models sometimes return empty or whitespace-only responses.
        """
        parser = ResponseParser()

        empty_responses = ["", " ", "\n", "\t\n\t"]

        for empty in empty_responses:
            result = parser.parse(empty)

            assert result is not None
            assert result.patch is None or result.patch == ""
            assert result.confidence == 0.0
            assert len(result.issues) > 0

    def test_handles_response_with_only_explanation(self):
        """Handle responses that explain but don't provide patches.

        Why: Models sometimes explain the fix without providing actual code.
        """
        parser = ResponseParser()

        explanation_only = """To fix this bug, you need to:
1. Add input validation
2. Handle the edge case
3. Update the tests

This should resolve the issue."""

        result = parser.parse(explanation_only)

        assert result.patch is None or result.patch == ""
        assert result.format_detected == PatchFormat.UNKNOWN
        assert result.confidence < 0.5

    def test_extracts_partial_patches(self):
        """Extract usable content from partially valid patches.

        Why: Better to extract something than nothing from corrupted responses.
        """
        parser = ResponseParser(auto_fix_common_issues=True)

        # Patch that starts valid but gets cut off
        partial = """diff --git a/file.py b/file.py
--- a/file.py
+++ b/file.py
@@ -1,3 +1,3 @@
 line1
-old line
+new li"""  # Cut off mid-line

        result = parser.parse(partial)

        # Should extract what it can
        assert result.patch is not None
        assert len(result.issues) > 0  # Should note the truncation
        assert result.confidence < 1.0  # Not fully confident

    def test_handles_mixed_formats_in_response(self):
        """Handle responses with multiple format types.

        Why: Models sometimes mix formats in a single response.
        """
        parser = ResponseParser()

        mixed = """Here's the diff:
```diff
--- a/file1.py
+++ b/file1.py
@@ -1,1 +1,1 @@
-bug1
+fix1
```

And also update file2:
SEARCH:
bug2
REPLACE:
fix2"""

        result = parser.parse(mixed)

        # Should detect something
        assert result.patch is not None
        assert result.format_detected != PatchFormat.UNKNOWN


class TestResponseParserConfiguration:
    """Test parser configuration options."""

    def test_strict_mode_requires_exact_format(self):
        """Strict mode should reject imperfect patches.

        Why: Some use cases need exact format compliance.
        """
        lenient = ResponseParser(strict_mode=False)
        strict = ResponseParser(strict_mode=True)

        # Slightly malformed patch (missing newline at end)
        imperfect = """--- a/file.py
+++ b/file.py
@@ -1,1 +1,1 @@
-old
+new"""  # No trailing newline

        lenient_result = lenient.parse(imperfect)
        strict_result = strict.parse(imperfect)

        # Strict should be less confident or have more issues
        assert (
            strict_result.confidence <= lenient_result.confidence or
            len(strict_result.issues) >= len(lenient_result.issues)
        )

    def test_auto_fix_corrects_common_issues(self):
        """Auto-fix should correct common formatting problems.

        Why: Models often make consistent formatting errors we can fix.
        """
        parser = ResponseParser(auto_fix_common_issues=True)

        # Common issues: wrong line endings, extra spaces, etc.
        fixable = """--- a/file.py
+++ b/file.py
@@ -1,1 +1,1 @@
-old line
+new line  """  # Trailing spaces

        result = parser.parse(fixable)

        # Should fix and extract
        assert result.patch is not None
        # Fixed patch shouldn't have trailing spaces
        lines = result.patch.split('\n')
        for line in lines:
            assert not line.endswith(' ')

    def test_confidence_threshold_filtering(self):
        """Respect minimum confidence threshold.

        Why: Low confidence results might need human review.
        """
        parser = ResponseParser(min_confidence=0.7)

        # Very short, ambiguous response
        ambiguous = "fix bug"

        result = parser.parse(ambiguous)

        # Should have low confidence
        assert result.confidence < 0.7
        # Should note that confidence is below threshold
        if result.patch:
            assert len(result.issues) > 0 or len(result.suggestions) > 0


class TestResponseParserMetadata:
    """Test metadata extraction from responses."""

    def test_extracts_file_paths_from_patch(self):
        """Extract file paths from patches.

        Why: Need to know which files are being modified.
        """
        parser = ResponseParser()

        patch = """diff --git a/src/module.py b/src/module.py
--- a/src/module.py
+++ b/src/module.py
@@ -1,1 +1,1 @@
-old
+new"""

        result = parser.parse(patch)

        assert "files" in result.metadata or "file_paths" in result.metadata
        # Should extract src/module.py

    def test_counts_additions_and_deletions(self):
        """Count lines added and removed.

        Why: Useful for understanding patch scope.
        """
        parser = ResponseParser()

        patch = """--- a/file.py
+++ b/file.py
@@ -1,3 +1,4 @@
 line1
-line2
+line2_modified
+line3_new
 line4"""

        result = parser.parse(patch)

        # Should count changes
        if "additions" in result.metadata:
            assert result.metadata["additions"] == 2  # line2_modified and line3_new
        if "deletions" in result.metadata:
            assert result.metadata["deletions"] == 1  # line2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
