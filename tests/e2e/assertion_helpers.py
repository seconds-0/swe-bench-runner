"""Assertion helper functions for E2E tests.

This module provides reusable assertion patterns for validating
CLI output against UX_Plan.md specifications.
"""

import re
import sys
from typing import Optional


def assert_exit_code(actual: int, expected: int, context: str = "") -> None:
    """Assert exit code matches expected value.

    Args:
        actual: The actual exit code returned
        expected: The expected exit code from UX_Plan
        context: Additional context for the assertion
    """
    msg = f"Expected exit code {expected}, got {actual}"
    if context:
        msg += f" ({context})"
    assert actual == expected, msg


def assert_docker_error(output: str, error_type: str = "not_running") -> None:
    """Assert Docker-related error messages are present.

    Args:
        output: Combined stdout and stderr
        error_type: Type of Docker error (not_running, permission, desktop_stopped)
    """
    output_lower = output.lower()

    # Docker must be mentioned
    assert "docker" in output_lower, "Error should mention Docker"

    if error_type == "not_running":
        assert any(term in output_lower for term in ["not running", "unreachable", "daemon"]), \
            "Should indicate Docker is not running"
        # Should suggest how to start Docker
        if "darwin" in sys.platform:
            assert any(term in output for term in ["Docker Desktop", "whale icon"]), \
                "macOS should mention Docker Desktop"
        else:
            assert any(term in output for term in ["systemctl start docker", "DOCKER_HOST"]), \
                "Linux should mention systemctl or DOCKER_HOST"

    elif error_type == "permission":
        assert "permission denied" in output_lower, "Should mention permission denied"
        assert "usermod -aG docker" in output, "Should suggest usermod command"
        assert any(term in output for term in ["$USER", "newgrp docker"]), \
            "Should include full permission fix command"
        assert "/var/run/docker.sock" in output, "Should mention Docker socket path"

    elif error_type == "desktop_stopped":
        assert any(term in output for term in ["Docker Desktop", "not running"]), \
            "Should mention Docker Desktop not running"
        assert "whale icon" in output, "Should mention whale icon for macOS"


def assert_network_error(output: str, error_type: str = "general") -> None:
    """Assert network-related error messages are present.

    Args:
        output: Combined stdout and stderr
        error_type: Type of network error (general, ghcr_blocked, rate_limit, hf_rate_limit)
    """
    output_lower = output.lower()

    if error_type == "general":
        assert any(term in output_lower for term in ["network", "connection", "internet"]), \
            "Should mention network issue"
        assert any(term in output_lower for term in ["retry", "check", "offline"]), \
            "Should suggest retry or offline mode"

    elif error_type == "ghcr_blocked":
        assert "ghcr.io" in output or "registry" in output_lower, \
            "Should mention GitHub Container Registry"
        assert any(term in output for term in ["--registry", "alternate", "docker.io"]), \
            "Should suggest alternate registry"

    elif error_type == "git_rate_limit":
        assert any(term in output_lower for term in ["rate limit", "github"]), \
            "Should mention GitHub rate limit"
        assert any(term in output_lower for term in ["retry", "wait", "attempt"]), \
            "Should mention retry behavior"

    elif error_type == "hf_rate_limit":
        assert "huggingface" in output_lower or "hf" in output_lower, \
            "Should mention HuggingFace"
        assert "rate limit" in output_lower or "quota" in output_lower, \
            "Should mention rate limit"
        assert any(term in output for term in ["--hf-token", "HUGGINGFACE_TOKEN"]), \
            "Should suggest using HF token"
        assert any(term in output for term in ["10/hour", "wait 45 min"]), \
            "Should mention quota limits"


def assert_resource_error(output: str, error_type: str = "disk_space") -> None:
    """Assert resource-related error messages are present.

    Args:
        output: Combined stdout and stderr
        error_type: Type of resource error (disk_space, memory, docker_storage)
    """
    output_lower = output.lower()

    if error_type == "disk_space":
        assert any(term in output_lower for term in ["disk", "space", "storage"]), \
            "Should mention disk space"
        assert any(term in output_lower for term in ["gb", "free", "required"]), \
            "Should mention space requirements"
        assert "swebench clean" in output or "free space" in output_lower, \
            "Should suggest how to free space"

    elif error_type == "memory":
        assert any(term in output_lower for term in ["memory", "ram", "oom"]), \
            "Should mention memory issue"
        assert any(term in output for term in ["16GB", "increase", "Docker"]), \
            "Should suggest increasing Docker memory"

    elif error_type == "docker_storage":
        assert "docker" in output_lower and "storage" in output_lower, \
            "Should mention Docker storage"
        assert "docker system prune" in output, \
            "Should suggest docker system prune command"


def assert_patch_error(output: str, error_type: str) -> None:
    """Assert patch-related error messages are present.

    Args:
        output: Combined stdout and stderr
        error_type: Type of patch error
    """
    output_lower = output.lower()

    if error_type == "invalid_schema":
        assert any(term in output_lower for term in ["schema", "field", "missing", "invalid"]), \
            "Should mention schema issue"
        assert any(term in output_lower for term in ["line", "jsonl", "format"]), \
            "Should mention file format"

    elif error_type == "too_large":
        assert any(term in output_lower for term in ["size", "large", "mb"]), \
            "Should mention size issue"
        assert any(term in output for term in ["--max-patch-size", "env limit", "--patches-dir"]), \
            "Should suggest size workaround"

    elif error_type == "conflict":
        assert any(term in output_lower for term in ["conflict", "failed", "apply"]), \
            "Should mention patch conflict"
        assert any(term in output_lower for term in ["hunk", "line", "context"]), \
            "Should provide conflict details"
        assert "logs/" in output or "patch.log" in output, \
            "Should reference log file"

    elif error_type == "encoding":
        assert any(term in output_lower for term in ["encoding", "utf-8", "utf8"]), \
            "Should mention encoding issue"
        assert "line" in output_lower, "Should mention line number"

    elif error_type == "binary":
        assert "binary" in output_lower, "Should mention binary files"
        assert "--allow-binary" in output or "not allowed" in output_lower, \
            "Should mention binary file policy"


def assert_timeout_error(output: str, instance_id: Optional[str] = None) -> None:
    """Assert timeout-related error messages are present.

    Args:
        output: Combined stdout and stderr
        instance_id: Optional instance ID that timed out
    """
    output_lower = output.lower()

    assert any(term in output_lower for term in ["timeout", "timed out"]), \
        "Should mention timeout"
    assert any(term in output_lower for term in ["minutes", "min", "seconds"]), \
        "Should mention time duration"
    assert "--timeout-mins" in output or "increase" in output_lower, \
        "Should suggest increasing timeout"

    if instance_id:
        assert instance_id in output, f"Should mention instance {instance_id}"


def assert_validation_error(output: str, error_type: str = "instance_id") -> None:
    """Assert validation error messages are present.

    Args:
        output: Combined stdout and stderr
        error_type: Type of validation error
    """
    output_lower = output.lower()

    if error_type == "instance_id":
        assert any(term in output_lower for term in ["unknown", "invalid", "instance"]), \
            "Should mention invalid instance"
        assert any(term in output_lower for term in ["dataset", "check", "valid"]), \
            "Should suggest checking dataset"


def assert_progress_format(output: str) -> None:
    """Assert progress bar follows UX_Plan format.

    Expected formats from UX_Plan 4.1:
    - [▇▇▇▇▁] 195/300 (65%) • 🟢 122 passed • 🔴 73 failed • ⏱️ 4m remaining
    - Or simpler percentage format
    """
    # Check for progress indicators
    assert any(indicator in output for indicator in ["[▇", "%", "passed", "failed"]), \
        "Should show progress indicators"

    # Check for counts if present
    if "/" in output:
        # Should have format like "195/300"
        pattern = r'\d+/\d+'
        assert re.search(pattern, output), "Should show current/total counts"

    # Check for status indicators
    if "passed" in output.lower():
        assert "failed" in output.lower(), "Should show both passed and failed counts"


def assert_contains_suggestion(output: str) -> None:
    """Assert output contains actionable suggestions.

    Suggestions should include commands, flags, or specific actions.
    """
    # Look for command-like patterns
    command_patterns = [
        r'--[\w-]+',  # CLI flags
        r'docker [\w]+',  # Docker commands
        r'swebench [\w]+',  # swebench commands
        r'systemctl',  # System commands
        r'https?://',  # URLs
        r'export \w+',  # Environment variables
    ]

    has_suggestion = any(re.search(pattern, output) for pattern in command_patterns)
    assert has_suggestion, "Error should include actionable suggestion (command, flag, or URL)"


def assert_platform_specific(output: str, platform: Optional[str] = None) -> None:
    """Assert platform-specific messages are appropriate.

    Args:
        output: Combined stdout and stderr
        platform: Platform to check for ('darwin', 'linux', or auto-detect)
    """
    if platform is None:
        platform = sys.platform

    if "darwin" in platform:
        # macOS-specific checks
        if "docker" in output.lower():
            assert any(term in output for term in ["Docker Desktop", "Applications", "whale"]), \
                "macOS Docker errors should mention Docker Desktop"
    else:
        # Linux-specific checks
        if "docker" in output.lower() and "permission" in output.lower():
            assert any(term in output for term in ["usermod", "docker group", "/var/run/docker.sock"]), \
                "Linux Docker permission errors should mention usermod or socket"

