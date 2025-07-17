"""Data models for SWE-bench runner."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Patch:
    """Represents a single patch to evaluate."""

    instance_id: str
    patch: str

    def validate(self) -> None:
        """Basic validation for MVP."""
        if not self.instance_id:
            raise ValueError("instance_id cannot be empty")
        if not self.patch:
            raise ValueError("patch cannot be empty")

        # Check for basic patch format
        if not self.patch.strip().startswith(("diff --git", "--- ", "+++ ")):
            raise ValueError("patch must be in unified diff format")


@dataclass
class EvaluationResult:
    """Result of a single evaluation."""

    instance_id: str
    passed: bool
    error: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate result data."""
        if not self.instance_id:
            raise ValueError("instance_id cannot be empty")
