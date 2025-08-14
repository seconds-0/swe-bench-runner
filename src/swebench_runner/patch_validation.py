"""Injectable patch validation service for tests and runtime.

Provides a minimal DI seam so E2E doubles can observe validation.
Runtime UX remains governed by docker_run's own checks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class PatchValidatorProtocol(Protocol):
    def validate_patch(self, patch_file: Path) -> bool: ...
    def validate_patch_content(self, content: str) -> bool: ...


_active_validator: PatchValidatorProtocol | None = None


def set_patch_validator(validator: PatchValidatorProtocol | None) -> None:
    global _active_validator
    _active_validator = validator


def try_validate_file(patch_file: Path) -> Exception | None:
    """Call injected file validator if present; ignore outcome/errors.

    Tests using doubles record side-effects (lists, errors) upon invocation.
    """
    if _active_validator is None:
        return None
    try:
        _active_validator.validate_patch(patch_file)
        return None
    except Exception as e:
        # Record the error on validator if it supports tracking (E2E doubles)
        try:
            if hasattr(_active_validator, "validation_errors") and isinstance(_active_validator.validation_errors, list):  # type: ignore[attr-defined]
                _active_validator.validation_errors.append(str(e))  # type: ignore[attr-defined]
        except Exception:
            pass
        # Return the exception so caller can tailor messaging
        return e


def try_validate_content(content: str) -> Exception | None:
    """Call injected content validator if present; ignore outcome/errors."""
    if _active_validator is None:
        return None
    try:
        _active_validator.validate_patch_content(content)
        return None
    except Exception as e:
        return e


def try_apply_patch(content: str) -> Exception | None:
    """Attempt to apply patch via validator if it supports it (for E2E doubles)."""
    if _active_validator is None:
        return None
    try:
        # Some validators implement apply_patch(content, target_dir=None)
        apply_fn = getattr(_active_validator, "apply_patch", None)
        if apply_fn is None:
            return None
        apply_fn(content)  # type: ignore[misc]
        return None
    except Exception as e:
        # Track on validator if it has a list
        try:
            if hasattr(_active_validator, "validation_errors") and isinstance(_active_validator.validation_errors, list):  # type: ignore[attr-defined]
                _active_validator.validation_errors.append(str(e))  # type: ignore[attr-defined]
        except Exception:
            pass
        return e
