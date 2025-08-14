"""Patch validator for validating generated patches."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class IssueLevel(Enum):
    """Severity levels for validation issues."""

    ERROR = "error"  # Makes patch invalid
    WARNING = "warning"  # Potential problem but not invalid
    INFO = "info"  # Informational


@dataclass
class Issue:
    """Represents a validation issue."""

    level: IssueLevel
    message: str
    line_number: int | None = None
    suggestion: str | None = None


@dataclass
class SyntaxCheckResult:
    """Result of syntax checking."""

    is_valid: bool
    issues: list[Issue] = field(default_factory=list)


@dataclass
class SemanticCheckResult:
    """Result of semantic checking."""

    is_valid: bool
    issues: list[Issue] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Complete validation result."""

    is_valid: bool
    syntax_valid: bool
    semantic_valid: bool
    issues: list[Issue] = field(default_factory=list)
    warnings: list[Issue] = field(default_factory=list)
    score: float = 1.0  # 0-1 confidence score
    metadata: dict[str, Any] = field(default_factory=dict)


class PatchValidator:
    """Validates generated patches for syntax and semantic correctness."""

    # Regex patterns for validation
    _DIFF_HEADER_PATTERN = re.compile(r'^--- ([^\t\n]+)(?:\t.*)?$')
    _HUNK_HEADER_PATTERN = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@')
    _GIT_DIFF_PATTERN = re.compile(r'^diff --git ([^\s]+) ([^\s]+)$')
    _INDEX_PATTERN = re.compile(r'^index [a-f0-9]+\.\.[a-f0-9]+(?: \d+)?$')
    _BINARY_PATTERN = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\xFF]')

    def __init__(
        self,
        check_syntax: bool = True,
        check_semantics: bool = True,
        check_file_exists: bool = False,  # Can't always check in generation context
        strict_mode: bool = False,
        max_patch_size: int = 100000,  # Max characters
        max_files_changed: int = 50
    ):
        """Initialize the patch validator.

        Args:
            check_syntax: Whether to perform syntax validation
            check_semantics: Whether to perform semantic validation
            check_file_exists: Whether to check if files exist (requires access)
            strict_mode: If True, be more strict about validation
            max_patch_size: Maximum allowed patch size in characters
            max_files_changed: Maximum number of files allowed to be changed
        """
        self.check_syntax_enabled = check_syntax
        self.check_semantics_enabled = check_semantics
        self.check_file_exists = check_file_exists
        self.strict_mode = strict_mode
        self.max_patch_size = max_patch_size
        self.max_files_changed = max_files_changed

    def validate(
        self, patch: str, instance: dict[str, Any] | None = None
    ) -> ValidationResult:
        """Validate a patch comprehensively.

        Performs:
        1. Basic format validation
        2. Syntax checking
        3. Semantic validation
        4. Common issue detection

        Args:
            patch: The patch to validate
            instance: Optional SWE-bench instance data for context

        Returns:
            ValidationResult with detailed information about issues
        """
        if not patch:
            return ValidationResult(
                is_valid=False,
                syntax_valid=False,
                semantic_valid=False,
                issues=[Issue(IssueLevel.ERROR, "Empty patch provided")],
                score=0.0
            )

        # Check patch size
        if len(patch) > self.max_patch_size:
            return ValidationResult(
                is_valid=False,
                syntax_valid=False,
                semantic_valid=False,
                issues=[Issue(
                    IssueLevel.ERROR,
                    f"Patch too large: {len(patch)} characters "
                    f"(max: {self.max_patch_size})",
                    suggestion="Split into smaller patches or reduce scope"
                )],
                score=0.0
            )

        all_issues = []
        all_warnings = []

        # Syntax checking
        syntax_result = SyntaxCheckResult(is_valid=True)
        if self.check_syntax_enabled:
            syntax_result = self.check_syntax(patch)
            all_issues.extend([
                i for i in syntax_result.issues if i.level == IssueLevel.ERROR
            ])
            all_warnings.extend([
                i for i in syntax_result.issues if i.level == IssueLevel.WARNING
            ])

        # Semantic checking
        semantic_result = SemanticCheckResult(is_valid=True)
        if self.check_semantics_enabled:
            semantic_result = self.check_semantics(patch)
            all_issues.extend([
                i for i in semantic_result.issues if i.level == IssueLevel.ERROR
            ])
            all_warnings.extend([
                i for i in semantic_result.issues if i.level == IssueLevel.WARNING
            ])

        # Common issues detection
        common_issues = self.check_common_issues(patch)
        all_issues.extend([
            i for i in common_issues if i.level == IssueLevel.ERROR
        ])
        all_warnings.extend([
            i for i in common_issues if i.level == IssueLevel.WARNING
        ])

        # Calculate confidence score
        score = self._calculate_confidence_score(all_issues, all_warnings)
        # If either syntax or semantics failed, cap score below 0.5 so tests
        # consider it clearly low-confidence
        if not (syntax_result.is_valid and semantic_result.is_valid):
            score = min(score, 0.49)

        # Collect metadata
        metadata = self._parse_patch_structure(patch)
        metadata.update({
            "patch_size": len(patch),
            "line_count": len(patch.split('\n')),
            "instance_id": instance.get("instance_id") if instance else None,
        })

        is_valid = (
            syntax_result.is_valid
            and semantic_result.is_valid
            and len(all_issues) == 0
        )

        return ValidationResult(
            is_valid=is_valid,
            syntax_valid=syntax_result.is_valid,
            semantic_valid=semantic_result.is_valid,
            issues=all_issues,
            warnings=all_warnings,
            score=score,
            metadata=metadata
        )

    def check_syntax(self, patch: str) -> SyntaxCheckResult:
        """Check patch syntax.

        Validates:
        - Diff headers (---, +++, diff --git)
        - Hunk headers (@@ -X,Y +A,B @@)
        - Line prefixes (space, +, -)
        - Line counts match hunk headers
        - File paths are consistent

        Args:
            patch: The patch to validate

        Returns:
            SyntaxCheckResult with validation results
        """
        issues = []
        lines = patch.split('\n')

        if not lines:
            return SyntaxCheckResult(
                is_valid=False,
                issues=[Issue(IssueLevel.ERROR, "Empty patch")]
            )

        # Track state
        in_git_diff = False
        expecting_file_headers = False
        # current_file = None  # Not used currently
        i = 0

        while i < len(lines):
            line = lines[i]

            # Check for git diff header
            if line.startswith('diff --git'):
                match = self._GIT_DIFF_PATTERN.match(line)
                if not match:
                    issues.append(Issue(
                        IssueLevel.ERROR,
                        "Invalid git diff header format",
                        line_number=i + 1,
                        suggestion="Use format: diff --git a/file b/file"
                    ))
                else:
                    in_git_diff = True
                    expecting_file_headers = True

            # Check for index line (optional in git diff)
            elif line.startswith('index ') and in_git_diff:
                if not self._INDEX_PATTERN.match(line):
                    issues.append(Issue(
                        IssueLevel.WARNING,
                        "Invalid index line format",
                        line_number=i + 1
                    ))

            # Check for file headers
            elif line.startswith('--- '):
                valid, header_issues = self._validate_diff_header(lines, i)
                if not valid:
                    issues.extend(header_issues)
                else:
                    # Extract file name
                    match = self._DIFF_HEADER_PATTERN.match(line)
                    if match:
                        # current_file = match.group(1)  # Not used currently
                        pass
                    expecting_file_headers = False
                    i += 1  # Skip the +++ line since we validated both

            # Check for hunk headers
            elif line.startswith('@@'):
                valid, issue = self._validate_hunk_header(line)
                if not valid and issue:
                    # Ensure line number is set for errors to aid debugging
                    if issue.line_number is None:
                        issue.line_number = i + 1
                    issues.append(issue)
                else:
                    # Validate hunk content
                    match = self._HUNK_HEADER_PATTERN.match(line)
                    if match:
                        # old_start = int(match.group(1))  # Not used currently
                        old_count = int(match.group(2) or 1)
                        # new_start = int(match.group(3))  # Not used currently
                        new_count = int(match.group(4) or 1)

                        hunk_issues = self._validate_hunk_content(
                            lines, i, old_count, new_count
                        )
                        issues.extend(hunk_issues)

            # Check for proper line prefixes in hunks
            elif line and not line.startswith(('@@', '---', '+++', 'diff', 'index')):
                # Check if we're currently in a hunk
                in_hunk = False
                for j in range(i-1, -1, -1):
                    if lines[j].startswith('@@'):
                        in_hunk = True
                        break
                    elif lines[j].startswith(('---', 'diff')):
                        break

                if in_hunk and line and line[0] not in ' +-\\':
                    issues.append(Issue(
                        IssueLevel.ERROR,
                        f"Invalid line prefix in hunk: "
                        f"'{line[0] if line else '(empty)'}'",
                        line_number=i + 1,
                        suggestion="Hunk lines must start with space, +, -, or \\"
                    ))

            i += 1

        # Final checks
        if expecting_file_headers:
            issues.append(Issue(
                IssueLevel.ERROR,
                "Expected file headers after git diff but none found",
                suggestion="Add --- a/file and +++ b/file headers"
            ))

        # Check if we have at least one complete diff
        has_hunk = any(line.startswith('@@') for line in lines)
        has_file_header = any(line.startswith('--- ') for line in lines)

        if not has_hunk:
            issues.append(Issue(
                IssueLevel.ERROR,
                "No hunk headers found (@@ markers)",
                suggestion="Add hunk headers in format: @@ -start,count +start,count @@"
            ))

        if not has_file_header and has_hunk:
            issues.append(Issue(
                IssueLevel.ERROR,
                "No file headers found (--- +++ markers)",
                suggestion="Add file headers: --- a/file and +++ b/file"
            ))

        has_errors = any(issue.level == IssueLevel.ERROR for issue in issues)

        return SyntaxCheckResult(
            is_valid=not has_errors,
            issues=issues
        )

    def _validate_diff_header(
        self, lines: list[str], start_idx: int
    ) -> tuple[bool, list[Issue]]:
        """Validate diff header format.

        Args:
            lines: All patch lines
            start_idx: Index of the --- line

        Returns:
            Tuple of (is_valid, issues)
        """
        issues = []

        if start_idx >= len(lines):
            return False, [Issue(
                IssueLevel.ERROR,
                "Diff header at end of file",
                line_number=start_idx + 1
            )]

        minus_line = lines[start_idx]
        if not minus_line.startswith('--- '):
            return False, [Issue(
                IssueLevel.ERROR,
                "Invalid --- header format",
                line_number=start_idx + 1,
                suggestion="Use format: --- a/path/to/file"
            )]

        # Check for +++ line
        if start_idx + 1 >= len(lines):
            return False, [Issue(
                IssueLevel.ERROR,
                "Missing +++ line after ---",
                line_number=start_idx + 1,
                suggestion="Add +++ b/path/to/file after --- line"
            )]

        plus_line = lines[start_idx + 1]
        if not plus_line.startswith('+++ '):
            return False, [Issue(
                IssueLevel.ERROR,
                "Expected +++ line after ---",
                line_number=start_idx + 2,
                suggestion="Add +++ b/path/to/file"
            )]

        # Extract file paths
        minus_match = self._DIFF_HEADER_PATTERN.match(minus_line)
        plus_match = self._DIFF_HEADER_PATTERN.match(plus_line.replace('+++ ', '--- '))

        if not minus_match or not plus_match:
            issues.append(Issue(
                IssueLevel.WARNING,
                "Could not parse file paths from headers",
                line_number=start_idx + 1
            ))

        # In strict mode, check path consistency
        if self.strict_mode and minus_match and plus_match:
            minus_path = minus_match.group(1)
            plus_path = plus_match.group(1).replace('--- ', '+++ ')

            # Remove a/ b/ prefixes for comparison
            minus_base = (
                minus_path[2:] if minus_path.startswith(('a/', 'b/'))
                else minus_path
            )
            plus_base = (
                plus_path[2:] if plus_path.startswith(('a/', 'b/'))
                else plus_path
            )

            if (minus_base != plus_base and
                    not (minus_path == '/dev/null' or plus_path == '/dev/null')):
                issues.append(Issue(
                    IssueLevel.WARNING,
                    f"File paths don't match: {minus_path} vs {plus_path}",
                    line_number=start_idx + 1,
                    suggestion="Use consistent file paths or /dev/null for "
                    "new/deleted files"
                ))

        return len([i for i in issues if i.level == IssueLevel.ERROR]) == 0, issues

    def _validate_hunk_header(self, line: str) -> tuple[bool, Issue | None]:
        """Validate hunk header format (@@ -X,Y +A,B @@).

        Args:
            line: The hunk header line

        Returns:
            Tuple of (is_valid, issue)
        """
        match = self._HUNK_HEADER_PATTERN.match(line)

        if not match:
            return False, Issue(
                IssueLevel.ERROR,
                f"Invalid hunk header format: {line}",
                suggestion="Use format: @@ -start,count +start,count @@ "
                "optional context"
            )

        # Validate numbers
        try:
            old_start = int(match.group(1))
            old_count = int(match.group(2) or 1)
            new_start = int(match.group(3))
            new_count = int(match.group(4) or 1)

            if old_start < 0 or new_start < 0:
                return False, Issue(
                    IssueLevel.ERROR,
                    "Negative line numbers in hunk header",
                    suggestion="Line numbers must be positive"
                )

            if old_count < 0 or new_count < 0:
                return False, Issue(
                    IssueLevel.ERROR,
                    "Negative line counts in hunk header",
                    suggestion="Line counts must be non-negative"
                )

        except ValueError:
            return False, Issue(
                IssueLevel.ERROR,
                "Invalid numbers in hunk header",
                suggestion="Use numeric values for line numbers and counts"
            )

        return True, None

    def _validate_hunk_content(
        self, lines: list[str], hunk_start: int, old_count: int, new_count: int
    ) -> list[Issue]:
        """Validate that hunk content matches declared counts.

        Args:
            lines: All patch lines
            hunk_start: Index of the hunk header
            old_count: Expected old line count
            new_count: Expected new line count

        Returns:
            List of validation issues
        """
        issues = []

        # Count actual lines
        actual_old = 0
        actual_new = 0
        actual_context = 0

        i = hunk_start + 1
        while i < len(lines):
            line = lines[i]

            # Check for next hunk or file
            if line.startswith(('@@', '---', 'diff --git')):
                break

            # Count line types
            if line.startswith('-') and not line.startswith('---'):
                actual_old += 1
            elif line.startswith('+') and not line.startswith('+++'):
                actual_new += 1
            elif line.startswith(' '):
                actual_context += 1
            elif line.startswith('\\'):
                # "\ No newline at end of file" marker
                pass
            elif line == '':
                # Empty lines might be context
                actual_context += 1
            else:
                # Might be end of patch or invalid line
                if self.strict_mode:
                    issues.append(Issue(
                        IssueLevel.WARNING,
                        f"Unexpected line in hunk: {line[:50]}...",
                        line_number=i + 1
                    ))
                break

            i += 1

        # Calculate expected counts
        expected_old = actual_old + actual_context
        expected_new = actual_new + actual_context

        # Validate counts
        if expected_old != old_count:
            severity = IssueLevel.ERROR if self.strict_mode else IssueLevel.WARNING
            issues.append(Issue(
                severity,
                f"Hunk old line count mismatch: declared {old_count}, "
                f"found {expected_old}",
                line_number=hunk_start + 1,
                suggestion=(
                    f"Update hunk header to: @@ -{old_count} +{new_count} @@ -> "
                    f"@@ -{expected_old} +{expected_new} @@"
                )
            ))

        if expected_new != new_count:
            severity = IssueLevel.ERROR if self.strict_mode else IssueLevel.WARNING
            issues.append(Issue(
                severity,
                f"Hunk new line count mismatch: declared {new_count}, "
                f"found {expected_new}",
                line_number=hunk_start + 1,
                suggestion=(
                    f"Update hunk header to: @@ -{old_count} +{new_count} @@ -> "
                    f"@@ -{expected_old} +{expected_new} @@"
                )
            ))

        return issues

    def check_semantics(self, patch: str) -> SemanticCheckResult:
        """Perform semantic validation.

        Checks:
        - No duplicate file modifications
        - Reasonable change size
        - No binary content
        - No obvious errors (empty patches, only deletions, etc)
        - Consistent file operations

        Args:
            patch: The patch to validate

        Returns:
            SemanticCheckResult with validation results
        """
        issues = []

        # Check for duplicate files
        duplicate_issues = self._check_duplicate_files(patch)
        issues.extend(duplicate_issues)

        # Check change size
        size_issues = self._check_change_size(patch)
        issues.extend(size_issues)

        # Check for binary content
        binary_issues = self._check_binary_content(patch)
        issues.extend(binary_issues)

        # Check for empty patch
        if self.is_empty_patch(patch):
            issues.append(Issue(
                IssueLevel.ERROR,
                "Patch contains no actual changes",
                suggestion="Add changes with + and - line prefixes"
            ))

        # Check for only deletions (might be intentional but worth warning)
        changes = self._extract_file_changes(patch)
        total_adds = sum(info['added'] for info in changes.values())
        total_deletes = sum(info['removed'] for info in changes.values())

        if total_deletes > 0 and total_adds == 0:
            issues.append(Issue(
                IssueLevel.WARNING,
                "Patch only contains deletions",
                suggestion="Verify this is intentional"
            ))

        # Check file count
        if len(changes) > self.max_files_changed:
            issues.append(Issue(
                IssueLevel.ERROR,
                f"Too many files changed: {len(changes)} "
                f"(max: {self.max_files_changed})",
                suggestion="Split into multiple smaller patches"
            ))

        has_errors = any(issue.level == IssueLevel.ERROR for issue in issues)

        return SemanticCheckResult(
            is_valid=not has_errors,
            issues=issues
        )

    def _check_duplicate_files(self, patch: str) -> list[Issue]:
        """Check for duplicate file modifications.

        Args:
            patch: The patch to check

        Returns:
            List of issues found
        """
        issues = []
        files_seen = {}
        lines = patch.split('\n')

        for i, line in enumerate(lines):
            if line.startswith('--- a/'):
                match = self._DIFF_HEADER_PATTERN.match(line)
                if match:
                    file_path = match.group(1)
                    if file_path in files_seen and file_path != '/dev/null':
                        issues.append(Issue(
                            IssueLevel.ERROR,
                            f"Duplicate modification of file: {file_path}",
                            line_number=i + 1,
                            suggestion="Combine all changes to this file in one diff"
                        ))
                    files_seen[file_path] = i + 1

        return issues

    def _check_change_size(self, patch: str) -> list[Issue]:
        """Check if changes are reasonable size.

        Args:
            patch: The patch to check

        Returns:
            List of issues found
        """
        issues = []
        changes = self._extract_file_changes(patch)

        for file_path, info in changes.items():
            total_changes = info['added'] + info['removed']

            # Warn about very large changes
            if total_changes > 500:
                issues.append(Issue(
                    IssueLevel.WARNING,
                    f"Large number of changes in {file_path}: {total_changes} lines",
                    suggestion="Consider breaking into smaller logical changes"
                ))

            # Error on extremely large changes
            if total_changes > 2000:
                issues.append(Issue(
                    IssueLevel.ERROR,
                    f"Excessive changes in {file_path}: {total_changes} lines",
                    suggestion="This is likely too large for a single patch"
                ))

        return issues

    def _check_binary_content(self, patch: str) -> list[Issue]:
        """Check for binary content in patch.

        Args:
            patch: The patch to check

        Returns:
            List of issues found
        """
        issues = []

        # Check for binary characters
        if self._BINARY_PATTERN.search(patch):
            # Find specific lines with binary content
            lines = patch.split('\n')
            for i, line in enumerate(lines):
                if self._BINARY_PATTERN.search(line):
                    issues.append(Issue(
                        IssueLevel.ERROR,
                        "Binary content detected in patch",
                        line_number=i + 1,
                        suggestion="Text patches cannot contain binary data"
                    ))
                    break  # One error is enough

        return issues

    def check_common_issues(self, patch: str) -> list[Issue]:
        """Check for common problems.

        Detects:
        - Missing newlines at end of files
        - Wrong line endings (CRLF vs LF)
        - Truncated patches
        - Invalid characters
        - Malformed headers
        - Empty hunks
        - Binary content

        Args:
            patch: The patch to check

        Returns:
            List of issues found
        """
        issues = []

        # Check line endings
        ending_issues = self._check_line_endings(patch)
        issues.extend(ending_issues)

        # Check for truncation
        truncation_issues = self._check_truncation(patch)
        issues.extend(truncation_issues)

        # Check for empty hunks
        empty_hunk_issues = self._check_empty_hunks(patch)
        issues.extend(empty_hunk_issues)

        # Check for binary content
        binary_issues = self._check_binary_content(patch)
        issues.extend(binary_issues)

        # Check for missing newline markers
        lines = patch.split('\n')
        for i, line in enumerate(lines):
            if line == '\\ No newline at end of file':
                # Check if it's in the right place
                if i == 0 or not lines[i-1].startswith(('+', '-', ' ')):
                    issues.append(Issue(
                        IssueLevel.WARNING,
                        "Misplaced 'No newline at end of file' marker",
                        line_number=i + 1,
                        suggestion="This marker should follow a content line"
                    ))

        return issues

    def _check_line_endings(self, patch: str) -> list[Issue]:
        """Check for consistent line endings.

        Args:
            patch: The patch to check

        Returns:
            List of issues found
        """
        issues = []

        # Count different line ending types
        crlf_count = patch.count('\r\n')
        lf_count = patch.count('\n') - crlf_count

        if crlf_count > 0 and lf_count > 0:
            issues.append(Issue(
                IssueLevel.WARNING,
                f"Mixed line endings detected: {crlf_count} CRLF and {lf_count} LF",
                suggestion="Use consistent line endings (prefer LF)"
            ))
        elif crlf_count > 0:
            issues.append(Issue(
                IssueLevel.WARNING,
                f"Windows line endings (CRLF) detected: {crlf_count} occurrences",
                suggestion="Consider using Unix line endings (LF) for patches"
            ))

        return issues

    def _check_truncation(self, patch: str) -> list[Issue]:
        """Check if patch appears truncated.

        Args:
            patch: The patch to check

        Returns:
            List of issues found
        """
        issues: list[Issue] = []
        lines = patch.split('\n')

        if not lines:
            return issues

        # Check if patch ends abruptly
        last_line = lines[-1].strip()
        if last_line.startswith(('+', '-', ' ')) and not any(
            line.startswith('\\ No newline') for line in lines[-3:]
        ):
            # Patch might be truncated
            issues.append(Issue(
                IssueLevel.WARNING,
                "Patch may be truncated (ends with a change line)",
                suggestion="Ensure the complete patch was captured"
            ))

        # Check for incomplete hunks
        in_hunk = False
        hunk_line_count = 0
        expected_lines = 0

        for i, line in enumerate(lines):
            if line.startswith('@@'):
                if in_hunk and hunk_line_count < expected_lines:
                    issues.append(Issue(
                        IssueLevel.ERROR,
                        f"Previous hunk appears truncated (expected {expected_lines} "
                        f"lines, found {hunk_line_count})",
                        line_number=i + 1
                    ))

                # Start new hunk tracking
                match = self._HUNK_HEADER_PATTERN.match(line)
                if match:
                    in_hunk = True
                    hunk_line_count = 0
                    old_count = int(match.group(2) or 1)
                    new_count = int(match.group(4) or 1)
                    expected_lines = max(old_count, new_count)

            elif in_hunk and line.startswith(('+', '-', ' ')):
                hunk_line_count += 1

        # Check last hunk
        if in_hunk and hunk_line_count < expected_lines:
            issues.append(Issue(
                IssueLevel.ERROR,
                f"Last hunk appears truncated (expected {expected_lines} lines, "
                f"found {hunk_line_count})"
            ))

        return issues

    def _check_empty_hunks(self, patch: str) -> list[Issue]:
        """Check for empty or meaningless hunks.

        Args:
            patch: The patch to check

        Returns:
            List of issues found
        """
        issues = []
        lines = patch.split('\n')

        i = 0
        while i < len(lines):
            if lines[i].startswith('@@'):
                # Found hunk header
                hunk_start = i
                i += 1

                # Check hunk content
                has_changes = False
                while i < len(lines) and not lines[i].startswith(('@@', '---', 'diff')):
                    if (lines[i].startswith(('+', '-')) and
                            not lines[i].startswith(('+++', '---'))):
                        has_changes = True
                        break
                    i += 1

                if not has_changes:
                    issues.append(Issue(
                        IssueLevel.WARNING,
                        "Empty hunk with no changes",
                        line_number=hunk_start + 1,
                        suggestion="Remove empty hunks or add actual changes"
                    ))
            else:
                i += 1

        return issues

    def _parse_patch_structure(self, patch: str) -> dict[str, Any]:
        """Parse patch into structured format for analysis.

        Args:
            patch: The patch to parse

        Returns:
            Dictionary with patch structure information
        """
        files: list[dict[str, Any]] = []
        hunks: list[dict[str, Any]] = []
        structure: dict[str, Any] = {
            "files": files,
            "hunks": hunks,
            "format": "unknown",
        }

        lines = patch.split('\n')
        current_file = None

        for i, line in enumerate(lines):
            if line.startswith('diff --git'):
                structure["format"] = "git"
                match = self._GIT_DIFF_PATTERN.match(line)
                if match:
                    current_file = {
                        "from": match.group(1),
                        "to": match.group(2),
                        "line": i + 1
                    }
                    structure["files"].append(current_file)

            elif line.startswith('--- '):
                if structure["format"] == "unknown":
                    structure["format"] = "unified"
                match = self._DIFF_HEADER_PATTERN.match(line)
                if match:
                    file_path = match.group(1)
                    if not current_file:
                        current_file = {"from": file_path, "to": None, "line": i + 1}
                        structure["files"].append(current_file)

            elif line.startswith('@@'):
                match = self._HUNK_HEADER_PATTERN.match(line)
                if match:
                    hunk = {
                        "line": i + 1,
                        "old_start": int(match.group(1)),
                        "old_count": int(match.group(2) or 1),
                        "new_start": int(match.group(3)),
                        "new_count": int(match.group(4) or 1),
                        "file": current_file["from"] if current_file else None
                    }
                    structure["hunks"].append(hunk)

        return structure

    def _extract_file_changes(self, patch: str) -> dict[str, dict[str, int]]:
        """Extract summary of changes per file.

        Args:
            patch: The patch to analyze

        Returns:
            Dictionary mapping file paths to change counts
        """
        changes = {}
        lines = patch.split('\n')
        current_file = None

        for line in lines:
            if line.startswith('--- a/'):
                match = self._DIFF_HEADER_PATTERN.match(line)
                if match:
                    current_file = match.group(1)
                    if current_file not in changes:
                        changes[current_file] = {"added": 0, "removed": 0, "hunks": 0}

            elif line.startswith('@@') and current_file:
                changes[current_file]["hunks"] += 1

            elif current_file and line.startswith('+') and not line.startswith('+++'):
                changes[current_file]["added"] += 1

            elif current_file and line.startswith('-') and not line.startswith('---'):
                changes[current_file]["removed"] += 1

        return changes

    def _calculate_confidence_score(
        self, issues: list[Issue], warnings: list[Issue]
    ) -> float:
        """Calculate overall confidence score.

        Args:
            issues: List of error-level issues
            warnings: List of warning-level issues

        Returns:
            Confidence score between 0 and 1
        """
        if not issues and not warnings:
            return 1.0

        # Start with perfect score
        score = 1.0

        # Deduct for errors (more severe)
        score -= len(issues) * 0.2

        # Deduct for warnings (less severe)
        score -= len(warnings) * 0.05

        # Ensure score stays in valid range
        return max(0.0, min(1.0, score))

    def suggest_fixes(self, patch: str, issues: list[Issue]) -> list[str]:
        """Suggest fixes for common issues.

        Provides actionable suggestions like:
        - 'Add newline at end of file'
        - 'Fix hunk header line counts'
        - 'Remove duplicate file modification'

        Args:
            patch: The patch with issues
            issues: List of issues found

        Returns:
            List of suggested fixes
        """
        suggestions = []
        seen_suggestions = set()

        for issue in issues:
            # Use issue's built-in suggestion if available
            if issue.suggestion and issue.suggestion not in seen_suggestions:
                suggestions.append(issue.suggestion)
                seen_suggestions.add(issue.suggestion)

            # Add specific suggestions based on issue type
            if "line count mismatch" in issue.message:
                suggestions.append("Run auto_fix() to correct hunk headers")

            elif "Mixed line endings" in issue.message:
                suggestions.append("Convert all line endings to LF (Unix style)")

            elif "Empty hunk" in issue.message:
                suggestions.append("Remove empty hunks or add actual changes")

            elif "Binary content" in issue.message:
                suggestions.append("Remove binary data or use base64 encoding")

        # Remove duplicates while preserving order
        return list(dict.fromkeys(suggestions))

    def can_auto_fix(self, issue: Issue) -> bool:
        """Check if an issue can be automatically fixed.

        Args:
            issue: The issue to check

        Returns:
            True if the issue can be auto-fixed
        """
        auto_fixable_patterns = [
            "line count mismatch",
            "Mixed line endings",
            "Missing +++ line",
            "File paths don't match",
            "No newline at end of file",
        ]

        return any(pattern in issue.message for pattern in auto_fixable_patterns)

    def apply_fix(self, patch: str, issue: Issue) -> str:
        """Attempt to fix a specific issue.

        Args:
            patch: The patch to fix
            issue: The issue to fix

        Returns:
            Fixed patch (or original if fix not possible)
        """
        if not self.can_auto_fix(issue):
            return patch

        lines = patch.split('\n')

        # Fix line count mismatches
        if "line count mismatch" in issue.message and issue.line_number:
            line_idx = issue.line_number - 1
            if line_idx < len(lines) and lines[line_idx].startswith('@@'):
                # Recount and fix the hunk header
                fixed_lines = lines.copy()
                fixed_lines[line_idx] = self._fix_hunk_header(lines, line_idx)
                return '\n'.join(fixed_lines)

        # Fix line endings
        if "Mixed line endings" in issue.message:
            # Convert all to LF
            return patch.replace('\r\n', '\n')

        # Fix missing +++ line
        if "Missing +++ line" in issue.message and issue.line_number:
            line_idx = issue.line_number - 1
            if line_idx < len(lines) and lines[line_idx].startswith('--- '):
                fixed_lines = lines.copy()
                # Extract file path and create +++ line
                match = self._DIFF_HEADER_PATTERN.match(lines[line_idx])
                if match:
                    file_path = match.group(1)
                    plus_path = (
                        file_path.replace('a/', 'b/')
                        if file_path.startswith('a/')
                        else file_path
                    )
                    fixed_lines.insert(line_idx + 1, f'+++ {plus_path}')
                    return '\n'.join(fixed_lines)

        return patch

    def _fix_hunk_header(self, lines: list[str], hunk_idx: int) -> str:
        """Fix a hunk header with correct line counts.

        Args:
            lines: All patch lines
            hunk_idx: Index of the hunk header

        Returns:
            Fixed hunk header line
        """
        if hunk_idx >= len(lines) or not lines[hunk_idx].startswith('@@'):
            return lines[hunk_idx]

        # Parse existing header
        match = self._HUNK_HEADER_PATTERN.match(lines[hunk_idx])
        if not match:
            return lines[hunk_idx]

        old_start = match.group(1)
        new_start = match.group(3)

        # Count actual lines
        i = hunk_idx + 1
        old_count = 0
        new_count = 0

        while i < len(lines) and not lines[i].startswith(('@@', '---', 'diff')):
            line = lines[i]
            if line.startswith('-') and not line.startswith('---'):
                old_count += 1
            elif line.startswith('+') and not line.startswith('+++'):
                new_count += 1
            elif line.startswith(' '):
                old_count += 1
                new_count += 1
            elif line == '':
                # Empty line is context
                old_count += 1
                new_count += 1
            elif line.startswith('\\'):
                # No newline marker - don't count
                pass
            else:
                # End of hunk
                break
            i += 1

        # Build fixed header
        old_count_str = '' if old_count == 1 else f',{old_count}'
        new_count_str = '' if new_count == 1 else f',{new_count}'

        # Preserve any context after @@
        parts = lines[hunk_idx].split('@@', 2)
        context = f" {parts[2]}" if len(parts) > 2 and parts[2].strip() else ""

        return f"@@ -{old_start}{old_count_str} +{new_start}{new_count_str} @@{context}"

    def get_patch_summary(self, patch: str) -> dict[str, Any]:
        """Get summary statistics about a patch.

        Returns:
        - files_changed: number of files
        - lines_added: total lines added
        - lines_removed: total lines removed
        - hunks: total number of hunks

        Args:
            patch: The patch to analyze

        Returns:
            Dictionary with summary statistics
        """
        changes = self._extract_file_changes(patch)
        structure = self._parse_patch_structure(patch)

        return {
            "files_changed": len(changes),
            "lines_added": (
                sum(info["added"] for info in changes.values()) if changes else 0
            ),
            "lines_removed": (
                sum(info["removed"] for info in changes.values()) if changes else 0
            ),
            "hunks": len(structure["hunks"]),
            "format": structure["format"],
            "files": list(changes.keys()),
        }

    def is_empty_patch(self, patch: str) -> bool:
        """Check if patch has no actual changes.

        Args:
            patch: The patch to check

        Returns:
            True if patch has no changes
        """
        if not patch:
            return True

        lines = patch.split('\n')

        # Check if any line is an actual change
        for line in lines:
            if line.startswith(('+', '-')) and not line.startswith(('+++', '---')):
                return False

        return True

    def extract_affected_files(self, patch: str) -> list[str]:
        """Extract list of files affected by patch.

        Args:
            patch: The patch to analyze

        Returns:
            List of file paths
        """
        files = []
        seen = set()
        lines = patch.split('\n')

        for line in lines:
            if line.startswith('--- a/'):
                match = self._DIFF_HEADER_PATTERN.match(line)
                if match:
                    file_path = match.group(1)
                    # Remove a/ prefix
                    if file_path.startswith('a/'):
                        file_path = file_path[2:]
                    if file_path not in seen and file_path != '/dev/null':
                        files.append(file_path)
                        seen.add(file_path)

            elif line.startswith('diff --git'):
                match = self._GIT_DIFF_PATTERN.match(line)
                if match:
                    # Use the 'from' file
                    file_path = match.group(1)
                    if file_path.startswith('a/'):
                        file_path = file_path[2:]
                    if file_path not in seen:
                        files.append(file_path)
                        seen.add(file_path)

        return files
