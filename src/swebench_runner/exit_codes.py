"""Exit codes for SWE-bench Runner as specified in PRD Section 5.8.

This module centralizes all exit codes used throughout the application
to ensure consistency with the Product Requirements Document (PRD).

Exit Code Meanings:
- 0: Success - Evaluation completed successfully
- 1: General Error - Harness errors, timeouts, unknown errors
- 2: Docker Not Found - Docker not installed or daemon not running
- 3: Network Error - Network failures, registry access issues
- 4: Resource Error - Insufficient disk space, memory, or other resources

Usage:
    from swebench_runner import exit_codes

    if error_condition:
        sys.exit(exit_codes.GENERAL_ERROR)
    else:
        sys.exit(exit_codes.SUCCESS)
"""

__all__ = [
    "SUCCESS",
    "GENERAL_ERROR",
    "DOCKER_NOT_FOUND",
    "NETWORK_ERROR",
    "RESOURCE_ERROR",
    "EXIT_CODE_NAMES",
    "EXIT_CODE_DESCRIPTIONS",
    "get_exit_code_name",
    "get_exit_code_description",
]

# Exit code constants as specified in PRD Section 5.8
SUCCESS = 0              # Evaluation completed successfully
GENERAL_ERROR = 1        # Harness errors, timeouts, unknown errors
DOCKER_NOT_FOUND = 2     # Docker not installed or not running
NETWORK_ERROR = 3        # Network failures, registry access
RESOURCE_ERROR = 4       # Disk space, memory issues

# Create a mapping for debugging/logging purposes
EXIT_CODE_NAMES = {
    SUCCESS: "SUCCESS",
    GENERAL_ERROR: "GENERAL_ERROR",
    DOCKER_NOT_FOUND: "DOCKER_NOT_FOUND",
    NETWORK_ERROR: "NETWORK_ERROR",
    RESOURCE_ERROR: "RESOURCE_ERROR",
}

# Create a mapping with descriptions for error messages
EXIT_CODE_DESCRIPTIONS = {
    SUCCESS: "Evaluation completed successfully",
    GENERAL_ERROR: "General error during evaluation",
    DOCKER_NOT_FOUND: "Docker is not installed or not running",
    NETWORK_ERROR: "Network error occurred",
    RESOURCE_ERROR: "Insufficient system resources",
}


def get_exit_code_name(code: int) -> str:
    """Get the name of an exit code.

    Args:
        code: The exit code integer

    Returns:
        The name of the exit code, or "UNKNOWN" if not recognized
    """
    return EXIT_CODE_NAMES.get(code, f"UNKNOWN ({code})")


def get_exit_code_description(code: int) -> str:
    """Get a human-readable description of an exit code.

    Args:
        code: The exit code integer

    Returns:
        A description of what the exit code means
    """
    return EXIT_CODE_DESCRIPTIONS.get(code, f"Unknown exit code: {code}")
