"""Unit tests for test doubles.

This module tests the test doubles themselves to ensure they correctly
simulate various scenarios for E2E testing.
"""

from pathlib import Path

import pytest

from tests.e2e.test_doubles import (
    DockerClientDouble,
    FileSystemDouble,
    HuggingFaceDouble,
    InstanceDouble,
    NetworkDouble,
    PatchValidatorDouble,
    ProviderDouble,
    TestDoubleFactory,
    docker,
)


class TestDockerClientDouble:
    """Test DockerClientDouble scenarios."""

    def test_success_scenario(self):
        """Test normal Docker operations."""
        double = DockerClientDouble(scenario="success")

        # Test ping
        assert double.ping() is True
        assert double.ping_called is True

        # Test container run
        containers = double.containers()
        result = containers.run("test-image", "echo test")
        assert double.containers_run_called is True
        assert len(double.containers_run) == 1
        assert double.containers_run[0]["image"] == "test-image"

    def test_not_running_scenario(self):
        """Test Docker daemon not running."""
        double = DockerClientDouble(scenario="not_running")

        with pytest.raises(docker.errors.DockerException) as exc_info:
            double.ping()
        assert "Cannot connect to the Docker daemon" in str(exc_info.value)
        assert double.ping_called is True

    def test_permission_denied_scenario(self):
        """Test Docker permission denied."""
        double = DockerClientDouble(scenario="permission_denied")

        with pytest.raises(PermissionError) as exc_info:
            double.ping()
        assert "Permission denied" in str(exc_info.value)
        assert double.ping_called is True

    def test_oom_during_run_scenario(self):
        """Test out of memory during container run."""
        double = DockerClientDouble(scenario="oom_during_run")

        containers = double.containers()
        with pytest.raises(docker.errors.APIError) as exc_info:
            containers.run("test-image", "test-command")
        assert "OOM" in str(exc_info.value)
        assert double.containers_run_called is True
        assert double.containers_run[0]["failed"] == "oom"

    def test_container_timeout_scenario(self):
        """Test container operation timeout."""
        double = DockerClientDouble(scenario="container_timeout")

        containers = double.containers()
        with pytest.raises(TimeoutError) as exc_info:
            containers.run("test-image", "test-command")
        assert "timed out" in str(exc_info.value)
        assert double.containers_run[0]["failed"] == "timeout"

    def test_stale_image_scenario(self):
        """Test stale Docker image detection."""
        double = DockerClientDouble(scenario="stale_image")

        assert double.image_age_days == 200
        images = double.images()
        image = images.get("test-image")
        assert "Created" in image.attrs


class TestNetworkDouble:
    """Test NetworkDouble scenarios."""

    def test_success_scenario(self):
        """Test normal network operations."""
        double = NetworkDouble(scenario="success")

        response = double.get("https://example.com")
        assert response["status_code"] == 200
        assert len(double.requests_made) == 1

    def test_connection_error_scenario(self):
        """Test connection refused."""
        double = NetworkDouble(scenario="connection_error")

        with pytest.raises(ConnectionError) as exc_info:
            double.get("https://example.com")
        assert "Connection refused" in str(exc_info.value)

    def test_timeout_scenario(self):
        """Test request timeout."""
        double = NetworkDouble(scenario="timeout")

        with pytest.raises(TimeoutError) as exc_info:
            double.get("https://example.com")
        assert "timeout" in str(exc_info.value).lower()

    def test_ghcr_blocked_scenario(self):
        """Test GitHub Container Registry blocked."""
        double = NetworkDouble(scenario="ghcr_blocked")

        with pytest.raises(Exception) as exc_info:
            double.get("https://ghcr.io/v2/")
        assert "ghcr.io blocked" in str(exc_info.value)

    def test_git_rate_limit_scenario(self):
        """Test GitHub API rate limit."""
        double = NetworkDouble(scenario="git_rate_limit")

        with pytest.raises(Exception) as exc_info:
            double.get("https://api.github.com/repos/test")
        assert "rate limit" in str(exc_info.value).lower()
        assert double.rate_limit_hits == 1


class TestPatchValidatorDouble:
    """Test PatchValidatorDouble scenarios."""

    def test_success_scenario(self):
        """Test valid patch validation."""
        double = PatchValidatorDouble(scenario="success")

        assert double.validate_patch(Path("test.patch")) is True
        assert len(double.patches_validated) == 1
        assert len(double.validation_errors) == 0

    def test_invalid_schema_scenario(self):
        """Test invalid patch schema."""
        double = PatchValidatorDouble(scenario="invalid_schema")

        with pytest.raises(ValueError) as exc_info:
            double.validate_patch(Path("test.patch"))
        assert "Invalid patch schema" in str(exc_info.value)
        assert len(double.validation_errors) == 1

    def test_too_large_scenario(self):
        """Test patch file too large."""
        double = PatchValidatorDouble(scenario="too_large")

        with pytest.raises(ValueError) as exc_info:
            double.validate_patch(Path("test.patch"))
        assert "too large" in str(exc_info.value)
        assert double.get_patch_size(Path("test.patch")) == 2 * 1024 * 1024

    def test_encoding_error_scenario(self):
        """Test patch encoding error."""
        double = PatchValidatorDouble(scenario="encoding_error")

        with pytest.raises(UnicodeDecodeError):
            double.validate_patch(Path("test.patch"))
        assert len(double.validation_errors) == 1

    def test_apply_failed_scenario(self):
        """Test patch application failure."""
        double = PatchValidatorDouble(scenario="apply_failed")

        with pytest.raises(RuntimeError) as exc_info:
            double.apply_patch("patch content", Path("/tmp"))
        assert "Failed to apply patch" in str(exc_info.value)


class TestFileSystemDouble:
    """Test FileSystemDouble scenarios."""

    def test_success_scenario(self):
        """Test normal filesystem operations."""
        double = FileSystemDouble(scenario="success")

        # Test write
        double.write_file(Path("test.txt"), "content")
        assert len(double.files_written) == 1

        # Test read
        content = double.read_file(Path("test.txt"))
        assert content == "Mock file content"
        assert len(double.files_read) == 1

        # Test disk usage
        usage = double.get_disk_usage()
        assert usage["free"] == 500000000

    def test_disk_full_scenario(self):
        """Test disk full errors."""
        double = FileSystemDouble(scenario="disk_full")

        with pytest.raises(OSError) as exc_info:
            double.write_file(Path("test.txt"), "content")
        assert "No space left" in str(exc_info.value)

        usage = double.get_disk_usage()
        assert usage["free"] == 1  # Only 1 byte free

    def test_permission_denied_scenario(self):
        """Test permission denied errors."""
        double = FileSystemDouble(scenario="permission_denied")

        with pytest.raises(PermissionError) as exc_info:
            double.write_file(Path("/etc/passwd"), "content")
        assert "Permission denied" in str(exc_info.value)

    def test_cache_corrupted_scenario(self):
        """Test corrupted cache files."""
        double = FileSystemDouble(scenario="cache_corrupted")

        with pytest.raises(ValueError) as exc_info:
            double.read_file(Path("cache/data.json"))
        assert "corrupted" in str(exc_info.value)


class TestInstanceDouble:
    """Test InstanceDouble scenarios."""

    def test_success_scenario(self):
        """Test normal instance operations."""
        double = InstanceDouble(scenario="success")

        assert double.validate_instance("django-123") is True
        result = double.run_instance("django-123", timeout=30)
        assert result["status"] == "success"
        assert len(double.instances_run) == 1

    def test_timeout_scenario(self):
        """Test instance evaluation timeout."""
        double = InstanceDouble(scenario="timeout")

        with pytest.raises(TimeoutError) as exc_info:
            double.run_instance("django-123", timeout=30)
        assert "timed out after 30 minutes" in str(exc_info.value)

    def test_flaky_scenario(self):
        """Test flaky test detection."""
        double = InstanceDouble(scenario="flaky")

        # First two attempts should fail
        with pytest.raises(RuntimeError) as exc_info:
            double.run_instance("test-123", timeout=30)
        assert "attempt 1/3" in str(exc_info.value)

        with pytest.raises(RuntimeError) as exc_info:
            double.run_instance("test-123", timeout=30)
        assert "attempt 2/3" in str(exc_info.value)

        # Third attempt should succeed
        result = double.run_instance("test-123", timeout=30)
        assert result["status"] == "success"
        assert double.flaky_attempts == 3

    def test_invalid_id_scenario(self):
        """Test invalid instance ID."""
        double = InstanceDouble(scenario="invalid_id")

        with pytest.raises(ValueError) as exc_info:
            double.validate_instance("fake-123")
        assert "Invalid instance ID" in str(exc_info.value)


class TestHuggingFaceDouble:
    """Test HuggingFaceDouble scenarios."""

    def test_success_scenario(self):
        """Test normal HuggingFace operations."""
        double = HuggingFaceDouble(scenario="success")

        dataset = double.load_dataset("test-dataset")
        assert "train" in dataset
        assert len(double.datasets_loaded) == 1

        assert double.login("test-token") is True

    def test_rate_limit_scenario(self):
        """Test HuggingFace rate limit."""
        double = HuggingFaceDouble(scenario="rate_limit")

        with pytest.raises(Exception) as exc_info:
            double.load_dataset("test-dataset")
        assert "Rate limit exceeded" in str(exc_info.value)

    def test_auth_error_scenario(self):
        """Test authentication failure."""
        double = HuggingFaceDouble(scenario="auth_error")

        with pytest.raises(ValueError) as exc_info:
            double.login("invalid-token")
        assert "Invalid token" in str(exc_info.value)


class TestProviderDouble:
    """Test ProviderDouble scenarios."""

    def test_success_scenario(self):
        """Test normal provider operations."""
        double = ProviderDouble(provider="openai", scenario="success")

        result = double.generate("Test prompt", model="gpt-4")
        assert "diff --git" in result
        assert len(double.requests_made) == 1
        assert double.tokens_used > 0

    def test_rate_limit_scenario(self):
        """Test provider rate limit."""
        double = ProviderDouble(provider="anthropic", scenario="rate_limit")

        with pytest.raises(Exception) as exc_info:
            double.generate("Test prompt")
        assert "anthropic: Rate limit exceeded" in str(exc_info.value)

    def test_context_overflow_scenario(self):
        """Test context length exceeded."""
        double = ProviderDouble(scenario="context_overflow")

        with pytest.raises(ValueError) as exc_info:
            double.generate("Very long prompt" * 10000)
        assert "Context length exceeded" in str(exc_info.value)


class TestTestDoubleFactory:
    """Test the TestDoubleFactory."""

    def test_create_docker_double(self):
        """Test creating Docker double."""
        double = TestDoubleFactory.create_docker_double("not_running")
        assert isinstance(double, DockerClientDouble)
        assert double.scenario == "not_running"

    def test_create_all_doubles(self):
        """Test creating all doubles."""
        doubles = TestDoubleFactory.create_all_doubles("success")

        assert "docker" in doubles
        assert "network" in doubles
        assert "filesystem" in doubles
        assert "patch" in doubles
        assert "instance" in doubles
        assert "huggingface" in doubles
        assert "openai" in doubles
        assert "anthropic" in doubles

        # Check all have correct scenario
        assert doubles["docker"].scenario == "success"
        assert doubles["network"].scenario == "success"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
