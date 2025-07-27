"""Generation components for SWE-bench runner."""

from __future__ import annotations

from .batch_processor import (
    BatchProcessor,
    BatchResult,
    BatchStats,
    CheckpointData,
    ProgressTracker,
)
from .patch_generator import GenerationResult, PatchGenerator
from .patch_validator import (
    Issue,
    IssueLevel,
    PatchValidator,
    SemanticCheckResult,
    SyntaxCheckResult,
)
from .patch_validator import (
    ValidationResult as PatchValidationResult,
)
from .prompt_builder import PromptBuilder, PromptContext, TemplateStyle
from .response_parser import ParseResult, PatchFormat, ResponseParser, ValidationResult
from .token_manager import FitStats, TokenManager, TruncationStrategy

__all__ = [
    "BatchProcessor",
    "BatchResult",
    "BatchStats",
    "CheckpointData",
    "ProgressTracker",
    "PatchGenerator",
    "GenerationResult",
    "PatchValidator",
    "PatchValidationResult",
    "IssueLevel",
    "Issue",
    "SyntaxCheckResult",
    "SemanticCheckResult",
    "PromptBuilder",
    "PromptContext",
    "TemplateStyle",
    "ResponseParser",
    "ParseResult",
    "ValidationResult",
    "PatchFormat",
    "TokenManager",
    "TruncationStrategy",
    "FitStats",
]
