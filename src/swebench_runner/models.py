"""Data models for SWE-bench runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Patch:
    """Represents a single patch to evaluate."""

    instance_id: str
    patch: str

    def validate(self, max_size_mb: int = 5) -> None:
        """Enhanced validation including size, encoding, and binary detection."""
        if not self.instance_id:
            raise ValueError("instance_id cannot be empty")
        if not self.patch:
            raise ValueError("patch cannot be empty")

        # Check for basic patch format
        if not self.patch.strip().startswith(("diff --git", "--- ", "+++ ")):
            raise ValueError("patch must be in unified diff format")

        # Check patch size (configurable limit)
        patch_bytes = self.patch.encode('utf-8')
        max_size = max_size_mb * 1024 * 1024  # Convert MB to bytes

        # Check against the general limit only
        if len(patch_bytes) > max_size:
            size_mb = len(patch_bytes) / (1024 * 1024)
            raise ValueError(
                f"PATCH_TOO_LARGE: {size_mb:.1f}MB exceeds {max_size_mb}MB limit"
            )

        # Check for binary file indicators
        if self._has_binary_content():
            raise ValueError("patch contains binary files which are not allowed")

    def _has_binary_content(self) -> bool:
        """Check if patch contains binary file changes."""
        lines = self.patch.split('\n')
        for line in lines:
            # Check for Git binary file indicators
            if line.startswith('GIT binary patch'):
                return True
            if 'Binary files' in line and 'differ' in line:
                return True
            # Check for common binary file extensions in diff headers
            if line.startswith(('+++', '---')):
                line_lower = line.lower()
                binary_extensions = [
                    '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.tiff',
                    '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
                    '.zip', '.tar', '.gz', '.bz2', '.7z', '.rar',
                    '.exe', '.dll', '.so', '.dylib', '.a', '.o',
                    '.mp3', '.mp4', '.avi', '.mov', '.wmv', '.flv',
                    '.db', '.sqlite', '.dat', '.bin'
                ]
                if any(ext in line_lower for ext in binary_extensions):
                    return True
        return False


@dataclass
class EvaluationResult:
    """Result of a single evaluation."""

    instance_id: str
    passed: bool
    error: Optional[str] = None  # noqa: UP045

    def __post_init__(self) -> None:
        """Validate result data."""
        if not self.instance_id:
            raise ValueError("instance_id cannot be empty")
