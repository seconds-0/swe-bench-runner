"""Minimal pytest configuration for unit tests.

This configuration avoids complex fixtures that could cause import issues.
"""


import pytest


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory for tests."""
    return tmp_path


@pytest.fixture
def mock_env(monkeypatch):
    """Provide environment variable mocking."""
    # Set CI mode to avoid resource checks
    monkeypatch.setenv("CI", "true")
    monkeypatch.setenv("SWEBENCH_SKIP_RESOURCE_CHECK", "true")
    return monkeypatch
