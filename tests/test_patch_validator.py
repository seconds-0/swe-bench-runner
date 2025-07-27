"""Tests for PatchValidator component."""

import pytest

from swebench_runner.generation.patch_validator import (
    Issue,
    IssueLevel,
    PatchValidator,
)


class TestPatchValidator:
    """Test the PatchValidator class."""

    @pytest.fixture
    def validator(self):
        """Create a PatchValidator instance."""
        return PatchValidator()

    @pytest.fixture
    def strict_validator(self):
        """Create a strict PatchValidator instance."""
        return PatchValidator(strict_mode=True)

    def test_init_defaults(self):
        """Test default initialization."""
        validator = PatchValidator()
        assert validator.check_syntax_enabled is True
        assert validator.check_semantics_enabled is True
        assert validator.check_file_exists is False
        assert validator.strict_mode is False
        assert validator.max_patch_size == 100000
        assert validator.max_files_changed == 50

    def test_init_custom(self):
        """Test custom initialization."""
        validator = PatchValidator(
            check_syntax=False,
            check_semantics=False,
            check_file_exists=True,
            strict_mode=True,
            max_patch_size=50000,
            max_files_changed=25
        )
        assert validator.check_syntax_enabled is False
        assert validator.check_semantics_enabled is False
        assert validator.check_file_exists is True
        assert validator.strict_mode is True
        assert validator.max_patch_size == 50000
        assert validator.max_files_changed == 25

    def test_validate_empty_patch(self, validator):
        """Test validation of empty patch."""
        result = validator.validate("")
        assert not result.is_valid
        assert not result.syntax_valid
        assert not result.semantic_valid
        assert len(result.issues) == 1
        assert result.issues[0].level == IssueLevel.ERROR
        assert "Empty patch" in result.issues[0].message
        assert result.score == 0.0

    def test_validate_none_patch(self, validator):
        """Test validation of None patch."""
        result = validator.validate(None)
        assert not result.is_valid
        assert not result.syntax_valid
        assert not result.semantic_valid
        assert len(result.issues) == 1
        assert result.issues[0].level == IssueLevel.ERROR
        assert "Empty patch" in result.issues[0].message

    def test_validate_too_large_patch(self, validator):
        """Test patch size limit."""
        large_patch = "x" * (validator.max_patch_size + 1)
        result = validator.validate(large_patch)
        assert not result.is_valid
        assert len(result.issues) == 1
        assert "Patch too large" in result.issues[0].message

    def test_valid_unified_diff(self, validator):
        """Test validation of valid unified diff."""
        patch = """--- a/test.py
+++ b/test.py
@@ -1,3 +1,3 @@
 def hello():
-    print("old")
+    print("new")
"""
        result = validator.validate(patch)
        assert result.is_valid
        assert result.syntax_valid
        assert result.semantic_valid
        assert len(result.issues) == 0
        assert result.score > 0.8

    def test_valid_git_diff(self, validator):
        """Test validation of valid git diff."""
        patch = """diff --git a/test.py b/test.py
index abc123..def456 100644
--- a/test.py
+++ b/test.py
@@ -1,3 +1,3 @@
 def hello():
-    print("old")
+    print("new")
"""
        result = validator.validate(patch)
        assert result.is_valid
        assert result.syntax_valid
        assert result.semantic_valid
        assert len(result.issues) == 0
        assert result.score > 0.8

    def test_syntax_missing_file_headers(self, validator):
        """Test syntax check with missing file headers."""
        patch = """@@ -1,3 +1,3 @@
 def hello():
-    print("old")
+    print("new")
"""
        result = validator.check_syntax(patch)
        assert not result.is_valid
        assert any("No file headers found" in issue.message for issue in result.issues)

    def test_syntax_invalid_hunk_header(self, validator):
        """Test syntax check with invalid hunk header."""
        patch = """--- a/test.py
+++ b/test.py
@@ invalid hunk @@
 def hello():
-    print("old")
+    print("new")
"""
        result = validator.check_syntax(patch)
        assert not result.is_valid
        assert any(
            "Invalid hunk header format" in issue.message for issue in result.issues
        )

    def test_syntax_missing_plus_header(self, validator):
        """Test syntax check with missing +++ header."""
        patch = """--- a/test.py
@@ -1,3 +1,3 @@
 def hello():
-    print("old")
+    print("new")
"""
        result = validator.check_syntax(patch)
        assert not result.is_valid
        assert any(
            "Expected +++ line after ---" in issue.message for issue in result.issues
        )

    def test_syntax_hunk_count_mismatch_strict(self, strict_validator):
        """Test hunk count mismatch in strict mode."""
        patch = """--- a/test.py
+++ b/test.py
@@ -1,3 +1,2 @@
 def hello():
-    print("old")
+    print("new")
"""
        result = strict_validator.check_syntax(patch)
        assert not result.is_valid
        assert any("line count mismatch" in issue.message for issue in result.issues)

    def test_syntax_hunk_count_mismatch_lenient(self, validator):
        """Test hunk count mismatch in lenient mode."""
        patch = """--- a/test.py
+++ b/test.py
@@ -1,3 +1,2 @@
 def hello():
-    print("old")
+    print("new")
"""
        result = validator.check_syntax(patch)
        # Should be valid but with warnings
        assert result.is_valid
        assert any("line count mismatch" in issue.message for issue in result.issues
                  if issue.level == IssueLevel.WARNING)

    def test_syntax_invalid_line_prefix(self, validator):
        """Test syntax check with invalid line prefix."""
        patch = """--- a/test.py
+++ b/test.py
@@ -1,3 +1,3 @@
 def hello():
*    invalid_prefix
+    print("new")
"""
        result = validator.check_syntax(patch)
        assert not result.is_valid
        assert any("Invalid line prefix" in issue.message for issue in result.issues)

    def test_semantics_duplicate_files(self, validator):
        """Test semantic check with duplicate file modifications."""
        patch = """--- a/test.py
+++ b/test.py
@@ -1,1 +1,1 @@
-old
+new
--- a/test.py
+++ b/test.py
@@ -2,1 +2,1 @@
-old2
+new2
"""
        result = validator.check_semantics(patch)
        assert not result.is_valid
        assert any("Duplicate modification" in issue.message for issue in result.issues)

    def test_semantics_binary_content(self, validator):
        """Test semantic check with binary content."""
        patch = f"""--- a/test.py
+++ b/test.py
@@ -1,1 +1,1 @@
-old
+{chr(0)}{chr(1)}binary_content
"""
        result = validator.check_semantics(patch)
        assert not result.is_valid
        assert any(
            "Binary content detected" in issue.message for issue in result.issues
        )

    def test_semantics_only_deletions(self, validator):
        """Test semantic check with only deletions."""
        patch = """--- a/test.py
+++ b/test.py
@@ -1,3 +1,1 @@
 def hello():
-    print("old")
-    print("also old")
"""
        result = validator.check_semantics(patch)
        # Should be valid but with warning
        assert result.is_valid
        assert any("only contains deletions" in issue.message for issue in result.issues
                  if issue.level == IssueLevel.WARNING)

    def test_semantics_too_many_files(self, validator):
        """Test semantic check with too many files."""
        # Create patch with more files than allowed
        patch_parts = []
        for i in range(validator.max_files_changed + 1):
            patch_parts.append(f"""--- a/file{i}.py
+++ b/file{i}.py
@@ -1,1 +1,1 @@
-old{i}
+new{i}
""")
        patch = "\n".join(patch_parts)

        result = validator.check_semantics(patch)
        assert not result.is_valid
        assert any("Too many files changed" in issue.message for issue in result.issues)

    def test_common_issues_mixed_line_endings(self, validator):
        """Test common issues check with mixed line endings."""
        patch = "--- a/test.py\r\n+++ b/test.py\n@@ -1,1 +1,1 @@\n-old\n+new"
        issues = validator.check_common_issues(patch)
        assert any("Mixed line endings" in issue.message for issue in issues)

    def test_common_issues_truncated_patch(self, validator):
        """Test common issues check with truncated patch."""
        patch = """--- a/test.py
+++ b/test.py
@@ -1,5 +1,5 @@
 def hello():
-    print("old")
+    print("new")"""  # Missing expected lines
        issues = validator.check_common_issues(patch)
        assert any("truncated" in issue.message.lower() for issue in issues)

    def test_common_issues_empty_hunks(self, validator):
        """Test common issues check with empty hunks."""
        patch = """--- a/test.py
+++ b/test.py
@@ -1,0 +1,0 @@
"""
        issues = validator.check_common_issues(patch)
        assert any("Empty hunk" in issue.message for issue in issues)

    def test_get_patch_summary(self, validator):
        """Test patch summary generation."""
        patch = """--- a/test1.py
+++ b/test1.py
@@ -1,3 +1,3 @@
 def hello():
-    print("old")
+    print("new")
--- a/test2.py
+++ b/test2.py
@@ -1,2 +1,3 @@
 def world():
     pass
+    print("added")
"""
        summary = validator.get_patch_summary(patch)
        assert summary["files_changed"] == 2
        assert summary["lines_added"] == 2
        assert summary["lines_removed"] == 1
        assert summary["hunks"] == 2
        assert "a/test1.py" in summary["files"]
        assert "a/test2.py" in summary["files"]

    def test_is_empty_patch(self, validator):
        """Test empty patch detection."""
        # Truly empty
        assert validator.is_empty_patch("")
        assert validator.is_empty_patch(None)

        # Only headers, no changes
        empty_patch = """--- a/test.py
+++ b/test.py
@@ -1,1 +1,1 @@
 unchanged_line
"""
        assert validator.is_empty_patch(empty_patch)

        # Has changes
        nonempty_patch = """--- a/test.py
+++ b/test.py
@@ -1,1 +1,1 @@
-old
+new
"""
        assert not validator.is_empty_patch(nonempty_patch)

    def test_extract_affected_files(self, validator):
        """Test file extraction."""
        patch = """diff --git a/src/test1.py b/src/test1.py
index abc..def 100644
--- a/src/test1.py
+++ b/src/test1.py
@@ -1,1 +1,1 @@
-old
+new
--- a/src/test2.py
+++ b/src/test2.py
@@ -1,1 +1,1 @@
-old2
+new2
"""
        files = validator.extract_affected_files(patch)
        assert len(files) == 2
        assert "src/test1.py" in files
        assert "src/test2.py" in files

    def test_suggest_fixes(self, validator):
        """Test fix suggestions."""
        issues = [
            Issue(IssueLevel.ERROR, "line count mismatch", suggestion="Run auto_fix()"),
            Issue(IssueLevel.WARNING, "Mixed line endings", suggestion="Convert to LF"),
            Issue(
                IssueLevel.ERROR, "Binary content detected", suggestion="Remove binary data"
            ),
        ]

        suggestions = validator.suggest_fixes("", issues)
        assert "Run auto_fix()" in suggestions
        assert "Convert to LF" in suggestions
        assert "Remove binary data" in suggestions

    def test_can_auto_fix(self, validator):
        """Test auto-fix capability detection."""
        fixable = Issue(IssueLevel.ERROR, "line count mismatch in hunk")
        not_fixable = Issue(IssueLevel.ERROR, "Binary content detected")

        assert validator.can_auto_fix(fixable)
        assert validator.can_auto_fix(Issue(IssueLevel.WARNING, "Mixed line endings"))
        assert not validator.can_auto_fix(not_fixable)

    def test_apply_fix_line_endings(self, validator):
        """Test fixing mixed line endings."""
        patch = "--- a/test.py\r\n+++ b/test.py\r\n@@ -1,1 +1,1 @@\r\n-old\r\n+new"
        issue = Issue(IssueLevel.WARNING, "Mixed line endings detected")

        fixed_patch = validator.apply_fix(patch, issue)
        assert "\r\n" not in fixed_patch
        assert fixed_patch.count("\n") == patch.count("\r\n")

    def test_apply_fix_hunk_header(self, validator):
        """Test fixing hunk header counts."""
        lines = [
            "--- a/test.py",
            "+++ b/test.py",
            "@@ -1,3 +1,2 @@",  # Wrong counts
            " def hello():",
            "-    print('old')",
            "+    print('new')",
            " end"
        ]
        patch = "\n".join(lines)

        # Create issue pointing to hunk header
        issue = Issue(IssueLevel.ERROR, "Hunk old line count mismatch", line_number=3)

        fixed_patch = validator.apply_fix(patch, issue)
        assert "@@ -1,3 +1,3 @@" in fixed_patch  # Should be corrected

    def test_validate_with_instance_metadata(self, validator):
        """Test validation with instance metadata."""
        patch = """--- a/test.py
+++ b/test.py
@@ -1,1 +1,1 @@
-old
+new
"""
        instance = {"instance_id": "test_123"}

        result = validator.validate(patch, instance)
        assert result.metadata["instance_id"] == "test_123"
        assert "patch_size" in result.metadata
        assert "line_count" in result.metadata

    def test_confidence_scoring(self, validator):
        """Test confidence score calculation."""
        # Perfect patch
        perfect_patch = """--- a/test.py
+++ b/test.py
@@ -1,1 +1,1 @@
-old
+new
"""
        result = validator.validate(perfect_patch)
        assert result.score >= 0.9

        # Patch with warnings (lenient mode should produce warnings, not errors)
        warning_patch = """--- a/test.py
+++ b/test.py
@@ -1,4 +1,3 @@
 def hello():
-    print("old")
+    print("new")
 end
"""
        result = validator.validate(warning_patch)
        # Should have warnings about count mismatch but still be valid
        assert result.is_valid
        assert len(result.warnings) > 0
        assert 0.7 <= result.score <= 1.0

    def test_parse_patch_structure(self, validator):
        """Test patch structure parsing."""
        git_patch = """diff --git a/test.py b/test.py
index abc..def 100644
--- a/test.py
+++ b/test.py
@@ -1,3 +1,3 @@
 def hello():
-    print("old")
+    print("new")
@@ -10,2 +10,2 @@
 def world():
-    print("old2")
+    print("new2")
"""
        structure = validator._parse_patch_structure(git_patch)

        assert structure["format"] == "git"
        assert len(structure["files"]) == 1
        assert len(structure["hunks"]) == 2
        assert structure["files"][0]["from"] == "a/test.py"
        assert structure["files"][0]["to"] == "b/test.py"

    def test_disabled_checks(self):
        """Test validator with disabled checks."""
        validator = PatchValidator(check_syntax=False, check_semantics=False)

        # Even invalid patch should pass if checks are disabled
        bad_patch = "invalid patch content"
        result = validator.validate(bad_patch)

        # Only common issues should be found
        assert result.syntax_valid is True  # No syntax check performed
        assert result.semantic_valid is True  # No semantic check performed

    def test_file_path_normalization(self, validator):
        """Test file path extraction handles various formats."""
        patch = """--- a/path/to/file.py
+++ b/path/to/file.py
@@ -1,1 +1,1 @@
-old
+new
"""
        files = validator.extract_affected_files(patch)
        assert files == ["path/to/file.py"]

        # Git diff format
        git_patch = """diff --git a/other/file.py b/other/file.py
--- a/other/file.py
+++ b/other/file.py
@@ -1,1 +1,1 @@
-old
+new
"""
        files = validator.extract_affected_files(git_patch)
        assert files == ["other/file.py"]

    def test_newline_marker_validation(self, validator):
        """Test validation of 'No newline at end of file' markers."""
        # Misplaced marker
        patch = """\\ No newline at end of file
--- a/test.py
+++ b/test.py
@@ -1,1 +1,1 @@
-old
+new
"""
        issues = validator.check_common_issues(patch)
        assert any("Misplaced 'No newline at end of file' marker" in issue.message
                  for issue in issues)

        # Properly placed marker
        proper_patch = """--- a/test.py
+++ b/test.py
@@ -1,1 +1,1 @@
-old
+new
\\ No newline at end of file
"""
        issues = validator.check_common_issues(proper_patch)
        assert not any("Misplaced" in issue.message for issue in issues)
