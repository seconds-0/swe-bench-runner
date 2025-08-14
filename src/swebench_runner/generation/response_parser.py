"""Response parser for extracting patches from model outputs."""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class PatchFormat(Enum):
    """Detected patch format types."""

    UNIFIED_DIFF = "unified_diff"
    GIT_DIFF = "git_diff"
    FENCED_DIFF = "fenced_diff"
    FENCED_PATCH = "fenced_patch"
    FILE_BLOCKS = "file_blocks"
    SEARCH_REPLACE = "search_replace"
    EDIT_INSTRUCTIONS = "edit_instructions"
    UNKNOWN = "unknown"


@dataclass
class ParseResult:
    """Result from parsing a model response."""

    patch: str | None = None
    format_detected: PatchFormat = PatchFormat.UNKNOWN
    confidence: float = 0.0  # 0-1 confidence score
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result from patch validation."""

    is_valid: bool
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class ResponseParser:
    """Extracts patches from model responses."""

    # Compiled regex patterns for performance
    _UNIFIED_DIFF_PATTERN = re.compile(
        r'^--- [a-zA-Z0-9_/.-]+\n\+\+\+ [a-zA-Z0-9_/.-]+\n@@ .+ @@',
        re.MULTILINE
    )

    _GIT_DIFF_PATTERN = re.compile(
        r'^diff --git [a-zA-Z0-9_/.-]+ [a-zA-Z0-9_/.-]+\n'
        r'(?:index [a-f0-9]+\.\.[a-f0-9]+ \d+\n)?'
        r'--- [a-zA-Z0-9_/.-]+\n\+\+\+ [a-zA-Z0-9_/.-]+',
        re.MULTILINE
    )

    _FENCED_BLOCK_PATTERN = re.compile(
        r'```(?:diff|patch)\n(.*?)```',
        re.DOTALL
    )

    _FILE_BLOCK_PATTERN = re.compile(
        r'<file>([^<]+)</file>.*?<old>(.*?)</old>.*?<new>(.*?)</new>',
        re.DOTALL
    )

    _SEARCH_REPLACE_PATTERN = re.compile(
        r'(?:SEARCH|<<<old>>>):\s*\n(.*?)\n(?:REPLACE|<<<new>>>):\s*\n(.*?)(?:\n\n|$)',
        re.DOTALL | re.IGNORECASE
    )

    _HUNK_HEADER_PATTERN = re.compile(
        r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@',
        re.MULTILINE
    )

    def __init__(
        self,
        strict_mode: bool = False,
        auto_fix_common_issues: bool = True,
        min_confidence: float = 0.3,
        preferred_formats: list[PatchFormat] | None = None
    ):
        """Initialize the response parser.

        Args:
            strict_mode: If True, require exact format compliance
            auto_fix_common_issues: If True, attempt to fix common formatting issues
            min_confidence: Minimum confidence score to accept a result
            preferred_formats: List of formats to try first, in order
        """
        self.strict_mode = strict_mode
        self.auto_fix_common_issues = auto_fix_common_issues
        self.min_confidence = min_confidence
        self.preferred_formats = preferred_formats or [
            PatchFormat.GIT_DIFF,
            PatchFormat.FENCED_DIFF,
            PatchFormat.FENCED_PATCH,
            PatchFormat.UNIFIED_DIFF,
        ]

    def parse(self, response: str) -> ParseResult:
        """Parse a model response to extract a patch.

        This is an alias for extract_patch() for compatibility.

        Args:
            response: The model response to parse

        Returns:
            ParseResult with extracted patch and metadata
        """
        result = self.extract_patch(response)
        # Enrich metadata with file paths and change counts if a patch was found
        if result.patch:
            metadata = dict(result.metadata)
            files = []
            additions = 0
            deletions = 0
            for line in result.patch.split('\n'):
                if line.startswith('--- a/'):
                    path = line[6:].strip()
                    # Remove a/ prefix
                    if path.startswith('a/'):
                        path = path[2:]
                    files.append(path)
                elif line.startswith('+') and not line.startswith('+++'):
                    additions += 1
                elif line.startswith('-') and not line.startswith('---'):
                    deletions += 1
            if files:
                metadata.setdefault('files', list(dict.fromkeys(files)))
                metadata.setdefault('file_paths', list(dict.fromkeys(files)))
            metadata.setdefault('additions', additions)
            metadata.setdefault('deletions', deletions)
            result.metadata = metadata
        return result

    def extract_patch(
        self, response: str, instance: dict[str, Any] | None = None
    ) -> ParseResult:
        """Extract patch using multiple strategies.

        Args:
            response: The model's response text
            instance: Optional SWE-bench instance data for context

        Returns:
            ParseResult with the best extracted patch or failure information
        """
        if not response:
            return ParseResult(
                issues=["Empty response provided"],
                suggestions=["Ensure the model generates a response"]
            )

        logger.debug(
            f"Attempting to extract patch from response of length {len(response)}"
        )

        # Try each parsing strategy
        results = []

        # Try preferred formats first
        strategies = [
            (PatchFormat.GIT_DIFF, self._extract_git_diff),
            (PatchFormat.UNIFIED_DIFF, self._extract_unified_diff),
            (PatchFormat.FENCED_DIFF, self._extract_fenced_blocks),
            (PatchFormat.FENCED_PATCH, self._extract_fenced_blocks),
            (PatchFormat.FILE_BLOCKS, self._extract_file_blocks),
            (PatchFormat.SEARCH_REPLACE, self._extract_search_replace),
            (PatchFormat.EDIT_INSTRUCTIONS, self._extract_edit_instructions),
        ]

        # Reorder based on preferences
        ordered_strategies = []
        for pref_format in self.preferred_formats:
            for fmt, func in strategies:
                if fmt == pref_format:
                    ordered_strategies.append((fmt, func))
        # Add remaining strategies
        for strategy in strategies:
            if strategy not in ordered_strategies:
                ordered_strategies.append(strategy)

        # Try each strategy
        for format_type, extract_func in ordered_strategies:
            try:
                # Special handling for fenced blocks to detect correct format
                if format_type in [PatchFormat.FENCED_DIFF, PatchFormat.FENCED_PATCH]:
                    patch, confidence = extract_func(response)
                    # Determine actual format from fence type
                    if patch and '```diff' in response:
                        format_type = PatchFormat.FENCED_DIFF
                    elif patch and '```patch' in response:
                        format_type = PatchFormat.FENCED_PATCH
                else:
                    patch, confidence = extract_func(response)

                if patch and confidence >= self.min_confidence:
                    logger.debug(
                        f"Extracted patch using {format_type.value} with confidence "
                        f"{confidence:.2f}"
                    )

                    # Validate the patch
                    validation = self.validate_patch(patch)
                    pre_fix_issues = list(validation.issues)
                    pre_fix_warnings = list(validation.warnings)

                    # Auto-fix if enabled and needed
                    # (fix both invalid patches and patches with warnings)
                    if (self.auto_fix_common_issues and
                            (not validation.is_valid or validation.warnings)):
                        logger.debug("Attempting to auto-fix patch issues")
                        patch = self.auto_fix_patch(patch)
                        validation = self.validate_patch(patch)

                    # Normalize the patch
                    if validation.is_valid:
                        patch = self.normalize_patch(patch)

                    # Calculate final confidence
                    final_confidence = self._calculate_confidence(patch, format_type)
                    # If truncation likely, lower confidence slightly
                    if any(
                        isinstance(m, str) and 'truncat' in m.lower()
                        for m in (pre_fix_issues + pre_fix_warnings + validation.issues + validation.warnings)
                    ):
                        final_confidence = min(final_confidence, 0.95)

                    # If patch appears truncated or has structural warnings,
                    # surface them as issues in the ParseResult
                    if not validation.is_valid or validation.warnings:
                        for warn in validation.warnings:
                            # Include line number if available
                            msg = warn.message
                            if getattr(warn, 'line_number', None):
                                msg = f"Line {warn.line_number}: {msg}"
                            results_issues = []
                        # We'll attach in object below; keep per-result issues minimal

                    # Build parse result
                    combined_issues = pre_fix_issues + pre_fix_warnings + validation.issues + validation.warnings
                    # Heuristic truncation detection on the extracted patch text
                    if patch and not patch.endswith('\n'):
                        last = patch.split('\n')[-1]
                        if last and last[0] in ['+', '-', ' ']:
                            combined_issues.append("Patch may be truncated (no trailing newline)")
                            final_confidence = min(final_confidence, 0.7)
                    # Deduplicate while preserving order
                    combined_issues = list(dict.fromkeys(combined_issues))
                    results.append(ParseResult(
                        patch=patch,
                        format_detected=format_type,
                        confidence=final_confidence,
                        issues=combined_issues,
                        suggestions=(
                            ["Consider using unified diff format for "
                             "better compatibility"]
                            if format_type not in [
                                PatchFormat.UNIFIED_DIFF, PatchFormat.GIT_DIFF
                            ]
                            else []
                        ),
                        metadata={
                            "original_length": len(response),
                            "patch_length": len(patch) if patch else 0,
                            "validation_warnings": validation.warnings,
                        }
                    ))

                    # Return immediately if we have a valid result with good confidence
                    if validation.is_valid and final_confidence >= 0.8:
                        return results[-1]

            except Exception as e:
                logger.warning(f"Error with {format_type.value} extraction: {e}")

        # Return the best result we found
        if results:
            best_result = max(results, key=lambda r: r.confidence)
            if best_result.confidence >= self.min_confidence:
                return best_result

        # No valid patch found
        return ParseResult(
            issues=[
                "Could not extract a valid patch from the response",
                "Response did not match any known patch format"
            ],
            suggestions=[
                "Ensure the model outputs a patch in unified diff format",
                "Start patches with '--- a/file' and '+++ b/file' headers",
                "Use @@ markers for hunks",
                "Prefix changed lines with + or -"
            ],
            metadata={
                "response_length": len(response),
                "formats_tried": len(ordered_strategies)
            }
        )

    def _extract_unified_diff(self, response: str) -> tuple[str | None, float]:
        """Extract standard unified diff format.

        Pattern:
        --- a/file.py
        +++ b/file.py
        @@ -1,4 +1,4 @@
        -old line
        +new line
        """
        # Skip if this looks like a git diff or is in a fenced block
        if 'diff --git' in response or '```diff' in response or '```patch' in response:
            return None, 0.0

        # Look for unified diff pattern
        match = self._UNIFIED_DIFF_PATTERN.search(response)
        if not match:
            return None, 0.0

        # Find the start of the patch
        start_pos = response.find('--- ')
        if start_pos == -1:
            return None, 0.0

        # Extract from start to end of patch
        patch_lines = []
        lines = response[start_pos:].split('\n')
        in_patch = True

        for line in lines:
            # Check if we've reached the end of the patch
            if in_patch and line.strip():
                # Common end markers
                if (line.startswith('```') or
                    (not line.startswith((' ', '+', '-', '@', '---', '+++', '\\')) and
                     not re.match(r'^(diff|index|new file|deleted)', line))):
                    break

            patch_lines.append(line)

        patch = '\n'.join(patch_lines).rstrip()

        # Calculate confidence based on patch structure
        confidence = 0.9  # High base confidence for unified diff
        if '@@' not in patch:
            confidence -= 0.3
        if patch.count('---') != patch.count('+++'):
            confidence -= 0.2

        return patch, confidence

    def _extract_git_diff(self, response: str) -> tuple[str | None, float]:
        """Extract git-style diff format.

        Pattern:
        diff --git a/file.py b/file.py
        index abc123..def456 100644
        --- a/file.py
        +++ b/file.py
        @@ -1,4 +1,4 @@
        """
        match = self._GIT_DIFF_PATTERN.search(response)
        if not match:
            return None, 0.0

        # Find the start of the patch
        start_pos = response.find('diff --git')
        if start_pos == -1:
            return None, 0.0

        # Extract the full git diff
        patch_lines = []
        lines = response[start_pos:].split('\n')

        for line in lines:
            # Stop at obvious end markers
            if line.strip() and line.startswith('```'):
                break
            # Stop if we hit non-diff content (but allow empty lines)
            if (line.strip() and
                    not line.startswith((
                        ' ', '+', '-', '@', 'diff', '---', '+++', 'index',
                        '\\', 'new file', 'deleted'
                    ))):
                # Check if it might be a commit message or other git metadata
                if not re.match(r'^(Author:|Date:|commit [a-f0-9]+)', line):
                    break

            patch_lines.append(line)

        patch = '\n'.join(patch_lines).rstrip()

        # High confidence for git format
        confidence = 0.95
        if '@@' not in patch:
            confidence -= 0.3

        return patch, confidence

    def _extract_fenced_blocks(self, response: str) -> tuple[str | None, float]:
        """Extract from fenced code blocks.

        Patterns:
        ```diff
        content
        ```

        ```patch
        content
        ```
        """
        # Try both diff and patch fence types
        for fence_type in ['diff', 'patch']:
            pattern = re.compile(f'```{fence_type}\n(.*?)```', re.DOTALL)
            matches = pattern.findall(response)

            if matches:
                # Use the first match (could be extended to handle multiple)
                patch = matches[0].strip()

                # Confidence based on fence type and content
                base_confidence = 0.8 if fence_type == 'diff' else 0.75

                # Adjust confidence based on content
                if self._looks_like_unified_diff(patch):
                    confidence = base_confidence + 0.1
                elif self._looks_like_git_diff(patch):
                    confidence = base_confidence + 0.15
                else:
                    confidence = base_confidence

                return patch, confidence

        # Also try generic code blocks
        generic_pattern = re.compile(r'```\n(.*?)```', re.DOTALL)
        matches = generic_pattern.findall(response)

        for match in matches:
            if self._looks_like_unified_diff(match) or self._looks_like_git_diff(match):
                return match.strip(), 0.6  # Lower confidence for generic blocks

        return None, 0.0

    def _extract_file_blocks(self, response: str) -> tuple[str | None, float]:
        """Extract from file modification blocks.

        Pattern:
        <file>path/to/file.py</file>
        <old>
        old content
        </old>
        <new>
        new content
        </new>
        """
        matches = self._FILE_BLOCK_PATTERN.findall(response)

        if not matches:
            return None, 0.0

        # Convert file blocks to unified diff
        patch_parts = []

        for file_path, old_content, new_content in matches:
            file_path = file_path.strip()
            old_content = old_content.strip()
            new_content = new_content.strip()

            # Generate a diff for this file
            diff = self._construct_unified_diff(
                file_path,
                old_content,
                new_content
            )
            patch_parts.append(diff)

        patch = '\n'.join(patch_parts)

        # Medium confidence for file blocks
        confidence = 0.7

        return patch, confidence

    def _extract_search_replace(self, response: str) -> tuple[str | None, float]:
        """Extract from search/replace patterns.

        Patterns:
        SEARCH:
        old code
        REPLACE:
        new code

        <<<old>>>
        old code
        <<<new>>>
        new code
        """
        matches = self._SEARCH_REPLACE_PATTERN.findall(response)

        if not matches:
            # Try alternative pattern
            alt_pattern = re.compile(
                r'<<<old>>>\s*\n(.*?)\n<<<new>>>\s*\n(.*?)(?:\n|$)',
                re.DOTALL
            )
            matches = alt_pattern.findall(response)

        if not matches:
            return None, 0.0

        # We need file context to create a proper patch
        # Try to find file references in the response
        file_pattern = re.compile(r'(?:file|File|in|In)\s*[:\s]+([a-zA-Z0-9_/.-]+\.py)')
        file_matches = file_pattern.findall(response)

        if not file_matches:
            # Try to extract from the search content itself
            file_pattern2 = re.compile(r'^#\s*([a-zA-Z0-9_/.-]+\.py)', re.MULTILINE)
            file_matches = file_pattern2.findall(matches[0][0])

        file_path = file_matches[0] if file_matches else "unknown_file.py"

        # Create patches from search/replace pairs
        patch_parts = []

        for old_content, new_content in matches:
            old_content = old_content.strip()
            new_content = new_content.strip()

            # Create a simple patch
            diff = self._construct_simple_patch(
                file_path,
                old_content.split('\n'),
                new_content.split('\n')
            )
            patch_parts.append(diff)

        patch = '\n'.join(patch_parts)

        # Lower confidence for search/replace
        confidence = 0.6 if file_matches else 0.4

        return patch, confidence

    def _extract_edit_instructions(self, response: str) -> tuple[str | None, float]:
        """Convert natural language edit instructions to patch.

        Patterns:
        - In file.py, change "old" to "new"
        - Replace line X with Y
        - Add X after line Y
        - Delete lines X-Y
        """
        # This is more complex and would require file content access
        # For now, return low confidence result

        # Look for file references
        file_pattern = re.compile(
            r'(?:in|In|file|File)\s+([a-zA-Z0-9_/.-]+\.py)'
        )
        file_matches = file_pattern.findall(response)

        if not file_matches:
            return None, 0.0

        # Look for change patterns
        change_patterns = [
            (r'change\s+"([^"]+)"\s+to\s+"([^"]+)"', 'replace'),
            (r'replace\s+"([^"]+)"\s+with\s+"([^"]+)"', 'replace'),
            (r'add\s+"([^"]+)"\s+after\s+line\s+(\d+)', 'add_after'),
            (r'delete\s+lines?\s+(\d+)(?:\s*-\s*(\d+))?', 'delete'),
        ]

        # Extract instructions
        instructions = []
        for pattern, action in change_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for match in matches:
                instructions.append((action, match))

        if not instructions:
            return None, 0.0

        # Would need file content to create actual patch
        # Return None for now with low confidence
        logger.debug(f"Found edit instructions but need file content: {instructions}")

        return None, 0.3

    def _construct_unified_diff(
        self,
        file_path: str,
        old_content: str,
        new_content: str,
        context_lines: int = 3
    ) -> str:
        """Construct a proper unified diff from old and new content."""
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)

        # Ensure lines end with newline
        old_lines = [line if line.endswith('\n') else line + '\n' for line in old_lines]
        new_lines = [line if line.endswith('\n') else line + '\n' for line in new_lines]

        # Handle empty files
        if not old_lines:
            old_lines = ['']
        if not new_lines:
            new_lines = ['']

        # Generate unified diff
        diff_lines = list(difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f'a/{file_path}',
            tofile=f'b/{file_path}',
            n=context_lines,
            lineterm=''
        ))

        # Join lines
        return '\n'.join(diff_lines)

    def _construct_simple_patch(
        self,
        file_path: str,
        old_lines: list[str],
        new_lines: list[str],
        line_start: int = 1
    ) -> str:
        """Construct a simple patch for single-hunk changes."""
        # Calculate line counts
        old_count = len(old_lines)
        new_count = len(new_lines)

        # Build patch
        patch_lines = [
            f'--- a/{file_path}',
            f'+++ b/{file_path}',
            f'@@ -{line_start},{old_count} +{line_start},{new_count} @@'
        ]

        # Add old lines with - prefix
        for line in old_lines:
            patch_lines.append(f'-{line}')

        # Add new lines with + prefix
        for line in new_lines:
            patch_lines.append(f'+{line}')

        return '\n'.join(patch_lines)

    def validate_patch(self, patch: str) -> ValidationResult:
        """Validate patch format and content."""
        issues = []
        warnings = []

        if not patch:
            return ValidationResult(
                is_valid=False,
                issues=["Empty patch"]
            )

        lines = patch.split('\n')

        # Check for file headers
        has_file_header = False
        has_hunk_header = False
        file_pairs = []
        current_file_pair = None

        i = 0
        while i < len(lines):
            line = lines[i]

            # Check for file headers
            if line.startswith('--- '):
                if i + 1 < len(lines) and lines[i + 1].startswith('+++ '):
                    has_file_header = True
                    from_file = line[4:].strip()
                    to_file = lines[i + 1][4:].strip()
                    current_file_pair = (from_file, to_file)
                    file_pairs.append(current_file_pair)
                    i += 1
                else:
                    issues.append(f"Missing +++ line after --- at line {i + 1}")

            # Check for git-style headers
            elif line.startswith('diff --git'):
                has_file_header = True
                # Extract file names
                parts = line.split()
                if len(parts) >= 4:
                    from_file = parts[2]
                    to_file = parts[3]
                    current_file_pair = (from_file, to_file)
                    file_pairs.append(current_file_pair)

            # Check for hunk headers
            elif line.startswith('@@'):
                has_hunk_header = True
                match = self._HUNK_HEADER_PATTERN.match(line)
                if not match:
                    issues.append(f"Invalid hunk header format at line {i + 1}: {line}")
                else:
                    # Validate hunk content
                    # old_start = int(match.group(1))  # Not used currently
                    old_count = int(match.group(2) or 1)
                    # new_start = int(match.group(3))  # Not used currently
                    new_count = int(match.group(4) or 1)

                    # Count actual lines in hunk
                    j = i + 1
                    actual_old = 0
                    actual_new = 0
                    actual_context = 0

                    while (j < len(lines) and
                           not lines[j].startswith(('@@', '---', 'diff'))):
                        if lines[j].startswith('-') and not lines[j].startswith('---'):
                            actual_old += 1
                        elif (lines[j].startswith('+') and
                              not lines[j].startswith('+++')):
                            actual_new += 1
                        elif lines[j].startswith(' '):
                            actual_context += 1
                        elif lines[j].startswith('\\'):
                            # "\ No newline at end of file" markers
                            pass
                        else:
                            # Might be end of patch
                            break
                        j += 1

                    # Validate counts
                    expected_old = actual_old + actual_context
                    expected_new = actual_new + actual_context

                    if not self.strict_mode:
                        # Allow some flexibility
                        if abs(expected_old - old_count) > 1:
                            warnings.append(
                                f"Hunk at line {i + 1}: expected {old_count} old "
                                f"lines, found {expected_old}"
                            )
                        if abs(expected_new - new_count) > 1:
                            warnings.append(
                                f"Hunk at line {i + 1}: expected {new_count} new "
                                f"lines, found {expected_new}"
                            )
                    else:
                        if expected_old != old_count:
                            issues.append(
                                f"Hunk at line {i + 1}: expected {old_count} old "
                                f"lines, found {expected_old}"
                            )
                        if expected_new != new_count:
                            issues.append(
                                f"Hunk at line {i + 1}: expected {new_count} new "
                                f"lines, found {expected_new}"
                            )

            i += 1

        # Final validation
        if not has_file_header:
            issues.append("No file headers found (--- a/file +++ b/file)")
        if not has_hunk_header:
            issues.append("No hunk headers found (@@ -n,m +n,m @@)")

        # Check for actual changes
        has_changes = any(
            line.startswith(('+', '-')) and not line.startswith(('+++', '---'))
            for line in lines
        )
        if not has_changes:
            issues.append("No actual changes found (lines starting with + or -)")

        # Heuristic: patch may be truncated if it ends with a change/context line
        # and there is no explicit newline marker at the end
        if lines:
            last_line = lines[-1].strip()
            if last_line and last_line[0] in ['+', '-', ' '] and not any(
                l.startswith('\\ No newline') for l in lines[-3:]
            ):
                warnings.append("Patch may be truncated (ends with change/context line)")

        return ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            warnings=warnings
        )

    def normalize_patch(self, patch: str) -> str:
        """Normalize patch to standard format."""
        lines = patch.split('\n')
        normalized_lines = []

        for line in lines:
            # Normalize line endings
            line = line.rstrip('\r\n')

            # Normalize file paths
            if line.startswith(('--- ', '+++ ')):
                # Ensure a/ b/ prefixes
                parts = line.split(None, 1)
                if len(parts) == 2:
                    prefix = parts[0]
                    path = parts[1]

                    # Remove quotes if present
                    if path.startswith('"') and path.endswith('"'):
                        path = path[1:-1]

                    # Add a/ or b/ prefix if missing
                    if prefix == '---' and not path.startswith('a/'):
                        path = 'a/' + path.lstrip('/')
                    elif prefix == '+++' and not path.startswith('b/'):
                        path = 'b/' + path.lstrip('/')

                    line = f"{prefix} {path}"

            normalized_lines.append(line)

        # Ensure single newline at end, but avoid introducing trailing space on header lines
        result = '\n'.join(line.rstrip(' ') for line in normalized_lines)
        if not result.endswith('\n'):
            result += '\n'

        return result

    def auto_fix_patch(self, patch: str) -> str:
        """Attempt to fix common issues."""
        if not patch:
            return patch

        lines = patch.split('\n')
        fixed_lines = []

        i = 0
        while i < len(lines):
            line = lines[i]

            # Fix missing file headers
            if i == 0 and line.startswith('@@'):
                # Insert generic file headers
                fixed_lines.extend([
                    '--- a/file',
                    '+++ b/file'
                ])

            # Fix hunk headers with wrong counts
            if line.startswith('@@'):
                match = self._HUNK_HEADER_PATTERN.match(line)
                if match:
                    # Count actual lines in this hunk
                    j = i + 1
                    old_lines = 0
                    new_lines = 0
                    context_lines = 0

                    while (j < len(lines) and
                           not lines[j].startswith(('@@', '---', 'diff'))):
                        if lines[j].startswith('-') and not lines[j].startswith('---'):
                            old_lines += 1
                        elif (lines[j].startswith('+') and
                              not lines[j].startswith('+++')):
                            new_lines += 1
                        elif lines[j].startswith(' '):
                            context_lines += 1
                        elif lines[j].startswith('\\'):
                            pass
                        else:
                            break
                        j += 1

                    # Reconstruct hunk header with correct counts
                    old_start = match.group(1)
                    new_start = match.group(3)
                    old_count = old_lines + context_lines
                    new_count = new_lines + context_lines

                    # Handle single-line hunks
                    old_count_str = '' if old_count == 1 else f',{old_count}'
                    new_count_str = '' if new_count == 1 else f',{new_count}'

                    fixed_line = (
                        f'@@ -{old_start}{old_count_str} +{new_start}{new_count_str} @@'
                    )

                    # Preserve any context after @@
                    hunk_parts = line.split('@@', 2)
                    if len(hunk_parts) > 2:
                        fixed_line += ' ' + hunk_parts[2].strip()

                    line = fixed_line

            # Fix missing newline markers
            if i == len(lines) - 1 and line and not line.startswith('\\'):
                # Check if previous line was a change
                if i > 0 and lines[i-1].startswith(('+', '-', ' ')):
                    fixed_lines.append(line)
                    fixed_lines.append('\\ No newline at end of file')
                    i += 1
                    continue

            fixed_lines.append(line)
            i += 1

        return '\n'.join(fixed_lines)

    def _calculate_confidence(self, patch: str, format: PatchFormat) -> float:
        """Calculate confidence score for extracted patch."""
        if not patch:
            return 0.0

        # Base confidence by format
        format_confidence = {
            PatchFormat.UNIFIED_DIFF: 0.9,
            PatchFormat.GIT_DIFF: 0.95,
            PatchFormat.FENCED_DIFF: 0.8,
            PatchFormat.FENCED_PATCH: 0.75,
            PatchFormat.FILE_BLOCKS: 0.7,
            PatchFormat.SEARCH_REPLACE: 0.6,
            PatchFormat.EDIT_INSTRUCTIONS: 0.5,
            PatchFormat.UNKNOWN: 0.3,
        }

        confidence = format_confidence.get(format, 0.5)

        # Adjust based on validation
        validation = self.validate_patch(patch)

        if validation.is_valid:
            confidence = min(1.0, confidence + 0.1)
        else:
            # Reduce confidence based on number of issues
            confidence *= max(0.3, 1.0 - (len(validation.issues) * 0.1))

        # Adjust based on patch characteristics
        lines = patch.split('\n')

        # Small bonus for proper structure (but don't exceed 1.0)
        if any(line.startswith('--- a/') for line in lines):
            confidence = min(1.0, confidence + 0.02)
        if any(line.startswith('@@ ') for line in lines):
            confidence = min(1.0, confidence + 0.02)

        # Penalty for too short
        if len(lines) < 4:
            confidence *= 0.8

        # Penalty for no actual changes
        change_lines = sum(
            1 for line in lines
            if line.startswith(('+', '-')) and not line.startswith(('+++', '---'))
        )
        if change_lines == 0:
            confidence *= 0.5

        return confidence

    def _looks_like_unified_diff(self, text: str) -> bool:
        """Check if text looks like a unified diff."""
        return bool(self._UNIFIED_DIFF_PATTERN.search(text))

    def _looks_like_git_diff(self, text: str) -> bool:
        """Check if text looks like a git diff."""
        return bool(self._GIT_DIFF_PATTERN.search(text))
