"""Error classification utilities for SWE-bench Runner."""

from . import exit_codes


def classify_error(error_message: str) -> int:
    """Classify error message to appropriate exit code.

    Args:
        error_message: The error message to classify

    Returns:
        Appropriate exit code from exit_codes module
    """
    error_lower = error_message.lower()

    # Check network errors first (includes "failed to pull")
    if any(term in error_lower for term in [
        "network", "connection", "unreachable", "registry", "pull",
        "resolve", "dns", "connection refused", "failed to pull",
        "pull access denied"
    ]):
        return exit_codes.NETWORK_ERROR

    # Timeouts are general errors, not network errors
    elif any(term in error_lower for term in ["timeout", "timed out"]):
        return exit_codes.GENERAL_ERROR

    # Docker daemon issues
    elif "docker" in error_lower and any(term in error_lower for term in [
        "not found", "not running", "daemon"
    ]):
        return exit_codes.DOCKER_NOT_FOUND

    # Resource issues
    elif any(term in error_lower for term in [
        "disk", "space", "memory", "ram"
    ]):
        return exit_codes.RESOURCE_ERROR

    # Default to general error
    else:
        return exit_codes.GENERAL_ERROR
