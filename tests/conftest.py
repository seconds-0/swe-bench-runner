"""
pytest configuration and shared fixtures for swebench-runner tests.

This module provides test isolation to prevent:
1. Cache pollution in ~/.swebench during tests
2. Interactive prompts from bootstrap flow
3. Actual bootstrap operations during tests
"""

import os

import pytest


@pytest.fixture
def simple_test_env(monkeypatch):
    """
    Simple environment setup for unit tests that don't need CLI mocking.

    This fixture only sets basic environment variables without importing
    any swebench_runner modules.
    """
    # Force CI mode to skip resource checks
    monkeypatch.setenv("CI", "true")
    monkeypatch.setenv("SWEBENCH_SKIP_RESOURCE_CHECK", "true")
    # Disable keyring to prevent keychain password prompts during tests
    monkeypatch.setenv("SWEBENCH_DISABLE_KEYRING", "true")

    # Use temp cache to avoid polluting real cache
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setenv("SWEBENCH_CACHE_DIR", tmpdir)
        yield tmpdir


@pytest.fixture
def isolated_test_environment(monkeypatch, tmp_path):
    """
    Fixture that provides complete test isolation.

    NOTE: This is NOT autouse - only tests that need CLI isolation should use it.
    Unit tests should not need this fixture.

    This fixture:
    1. Isolates cache directory to prevent ~/.swebench pollution
    2. Sets CI=true to disable all interactive prompts
    3. Mocks bootstrap to prevent first-run checks
    4. Provides temp directory for test files

    Args:
        monkeypatch: pytest's monkeypatch fixture
        tmp_path: pytest's tmp_path fixture

    Returns:
        Path: Temporary directory for test use
    """
    # Create isolated cache directory
    test_cache_dir = tmp_path / ".swebench"
    test_cache_dir.mkdir(exist_ok=True)

    # Redirect cache directory
    monkeypatch.setenv("SWEBENCH_CACHE_DIR", str(test_cache_dir))

    # Force CI mode to disable interactive prompts
    monkeypatch.setenv("CI", "true")

    # Disable keyring to prevent keychain password prompts during tests
    monkeypatch.setenv("SWEBENCH_DISABLE_KEYRING", "true")

    # Lazy import to avoid import issues
    from unittest.mock import patch

    # Mock bootstrap functions to prevent first-run checks
    with patch("swebench_runner.cli.check_and_prompt_first_run", return_value=False), \
         patch("swebench_runner.cli.suggest_patches_file", return_value=None):
        yield tmp_path


@pytest.fixture
def mock_docker(mocker):
    """
    Provides a mock Docker client for tests that need Docker interaction.

    Args:
        mocker: pytest-mock's mocker fixture

    Returns:
        Mock: Configured mock Docker client
    """
    mock_client = mocker.Mock()

    # Mock common Docker operations
    mock_client.ping.return_value = True
    mock_client.images.list.return_value = []
    mock_client.containers.list.return_value = []
    mock_client.info.return_value = {
        "ServerVersion": "20.10.0",
        "OperatingSystem": "Docker Desktop",
        "MemTotal": 8 * 1024 * 1024 * 1024,  # 8GB
    }

    # Mock docker.from_env to return our mock client
    mocker.patch("docker.from_env", return_value=mock_client)

    return mock_client


@pytest.fixture
def sample_patches_file(tmp_path):
    """
    Creates a sample patches.jsonl file for testing.

    Args:
        tmp_path: pytest's tmp_path fixture

    Returns:
        Path: Path to the created patches file
    """
    patches_file = tmp_path / "patches.jsonl"
    patches_content = """{
    "instance_id": "test-001",
    "repo": "test/repo",
    "base_commit": "abc123",
    "patch": "diff --git a/test.py b/test.py\\n"
             "index 0000000..1111111 100644\\n"
             "--- a/test.py\\n+++ b/test.py\\n"
             "@@ -1 +1 @@\\n-old line\\n+new line"
}
{
    "instance_id": "test-002",
    "repo": "test/repo2",
    "base_commit": "def456",
    "patch": "diff --git a/fix.py b/fix.py\\n"
             "index 0000000..2222222 100644\\n"
             "--- a/fix.py\\n+++ b/fix.py\\n"
             "@@ -1 +1 @@\\n-bug\\n+fixed"
}"""
    patches_file.write_text(patches_content)
    return patches_file


@pytest.fixture
def cli_runner():
    """
    Provides a Click CliRunner for testing CLI commands.

    Returns:
        CliRunner: Configured Click test runner
    """
    from click.testing import CliRunner
    return CliRunner()


# Markers for Docker tests
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "docker: marks tests that require Docker "
        "(skip with SWEBENCH_SKIP_DOCKER_TESTS=true)"
    )


def pytest_collection_modifyitems(config, items):
    """Skip Docker tests if environment variable is set."""
    if os.environ.get("SWEBENCH_SKIP_DOCKER_TESTS") == "true":
        skip_docker = pytest.mark.skip(
            reason="Skipping Docker tests (SWEBENCH_SKIP_DOCKER_TESTS=true)"
        )
        for item in items:
            # Skip tests marked with @pytest.mark.docker
            if "docker" in item.keywords:
                item.add_marker(skip_docker)
            # Also skip tests with docker in their name
            if "docker" in item.name.lower():
                item.add_marker(skip_docker)
