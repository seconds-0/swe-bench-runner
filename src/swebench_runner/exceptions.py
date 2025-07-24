"""Exception hierarchy for SWE-bench Runner."""

from __future__ import annotations


class SWEBenchRunnerError(Exception):
    """Base exception for all SWE-bench Runner errors."""
    pass


class DatasetError(SWEBenchRunnerError):
    """Base exception for dataset operations."""
    pass


class DatasetNotFoundError(DatasetError):
    """Dataset not found or invalid dataset name."""
    pass


class DatasetAuthenticationError(DatasetError):
    """Authentication required for dataset access."""
    pass


class DatasetNetworkError(DatasetError):
    """Network error while accessing dataset."""
    pass


class DatasetValidationError(DatasetError):
    """Invalid input parameters for dataset operations."""
    pass


class RegexValidationError(DatasetValidationError):
    """Invalid or potentially dangerous regex pattern."""
    pass


class InstanceValidationError(DatasetValidationError):
    """Invalid instance ID format or specification."""
    pass
