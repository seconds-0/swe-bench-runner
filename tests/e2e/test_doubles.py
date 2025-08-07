"""Test doubles for E2E testing.

This module provides clean test doubles to replace environment variable mocking.
Each double is designed to simulate specific behaviors and track interactions
for verification in tests.
"""

import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

# Simulate docker module structure when needed
try:
    import docker
    from docker.errors import APIError, DockerException
except ImportError:
    # Create mock classes for testing without docker installed
    class DockerException(Exception):
        pass

    class APIError(Exception):
        pass

    class docker:
        errors = type('errors', (), {
            'DockerException': DockerException,
            'APIError': APIError
        })()


class DockerClientDouble:
    """Test double for Docker client with scenario-based behaviors."""

    def __init__(self, scenario: str = "success"):
        """Initialize Docker client double.

        Args:
            scenario: Test scenario to simulate. Options:
                - "success": Normal operation
                - "not_running": Docker daemon not running
                - "permission_denied": Permission denied error
                - "desktop_stopped": Docker Desktop stopped (macOS)
                - "oom": Out of memory error (deprecated, use oom_during_run)
                - "oom_during_run": OOM error during container.run()
                - "storage_full": Disk storage full (deprecated, use storage_full_during_run)
                - "storage_full_during_run": Storage full during container operations
                - "network_error": Network connectivity issue
                - "container_limit": Too many containers (deprecated, use container_limit_exceeded)
                - "container_limit_exceeded": Too many containers (100+)
                - "stale_image": Image is old/stale
                - "container_timeout": Container operation timeout
        """
        self.scenario = scenario
        self.ping_called = False
        self.containers_run = []
        self.images_pulled = []
        self.containers_list = []
        self.containers_run_called = False
        self.image_age_days = 200 if scenario == "stale_image" else 1

    def ping(self) -> bool:
        """Simulate Docker ping operation."""
        self.ping_called = True

        if self.scenario == "not_running":
            raise docker.errors.DockerException("Cannot connect to the Docker daemon")
        elif self.scenario == "permission_denied":
            raise PermissionError("Permission denied: /var/run/docker.sock")
        elif self.scenario == "desktop_stopped":
            raise docker.errors.DockerException("Docker Desktop is stopped")
        elif self.scenario == "network_error":
            raise ConnectionError("Network error connecting to Docker")

        return True

    def containers(self) -> MagicMock:
        """Return mock containers interface."""
        mock = MagicMock()

        if self.scenario in ["container_limit", "container_limit_exceeded"]:
            # Simulate too many containers (100+)
            self.containers_list = [{"Id": f"container_{i}", "Status": "running"} for i in range(101)]

        mock.list.return_value = self.containers_list
        mock.run = self._container_run
        return mock

    def images(self) -> MagicMock:
        """Return mock images interface."""
        mock = MagicMock()
        mock.pull = self._image_pull
        mock.get = self._image_get
        return mock

    def _container_run(self, image: str, command: str = None, **kwargs) -> dict[str, Any]:
        """Simulate container run."""
        self.containers_run_called = True

        # Handle OOM scenarios - both old and new names
        if self.scenario in ["oom", "oom_during_run"]:
            # Track the attempt before raising
            container_info = {
                "image": image,
                "command": command,
                "kwargs": kwargs,
                "failed": "oom"
            }
            self.containers_run.append(container_info)
            raise docker.errors.APIError("Container killed due to OOM")

        # Handle storage full scenarios - both old and new names
        elif self.scenario in ["storage_full", "storage_full_during_run"]:
            # Track the attempt before raising
            container_info = {
                "image": image,
                "command": command,
                "kwargs": kwargs,
                "failed": "storage_full"
            }
            self.containers_run.append(container_info)
            raise docker.errors.APIError("No space left on device")

        # Handle container limit scenarios
        elif self.scenario in ["container_limit", "container_limit_exceeded"]:
            # Track the attempt before raising
            container_info = {
                "image": image,
                "command": command,
                "kwargs": kwargs,
                "failed": "container_limit"
            }
            self.containers_run.append(container_info)
            raise docker.errors.APIError("Cannot create container: too many containers (limit: 100)")

        # Handle container timeout scenario
        elif self.scenario == "container_timeout":
            # Track the attempt before raising
            container_info = {
                "image": image,
                "command": command,
                "kwargs": kwargs,
                "failed": "timeout"
            }
            self.containers_run.append(container_info)
            # Simulate timeout by raising a timeout error
            raise TimeoutError("Container operation timed out after 30 minutes")

        # Normal operation
        container_info = {
            "image": image,
            "command": command,
            "kwargs": kwargs
        }
        self.containers_run.append(container_info)

        # Return mock container object with additional methods
        mock_container = MagicMock()
        mock_container.id = f"container_{len(self.containers_run)}"
        mock_container.status = "running"

        # Add logs method for containers that might need it
        mock_container.logs.return_value = b"Container logs here\n"

        # Add wait method for containers
        mock_container.wait.return_value = {"StatusCode": 0}

        return mock_container

    def _image_pull(self, repository: str, tag: str = None) -> None:
        """Simulate image pull."""
        if self.scenario == "network_error":
            raise docker.errors.APIError("Network error pulling image")

        self.images_pulled.append(f"{repository}:{tag or 'latest'}")

    def _image_get(self, name: str) -> MagicMock:
        """Simulate image get for stale image checks."""
        mock_image = MagicMock()
        if self.scenario == "stale_image":
            # Return an image with old creation date
            import datetime
            old_date = datetime.datetime.now() - datetime.timedelta(days=self.image_age_days)
            mock_image.attrs = {"Created": old_date.isoformat()}
        else:
            # Return a recent image
            import datetime
            mock_image.attrs = {"Created": datetime.datetime.now().isoformat()}
        return mock_image


class HuggingFaceDouble:
    """Test double for HuggingFace dataset operations."""

    def __init__(self, scenario: str = "success"):
        """Initialize HuggingFace double.

        Args:
            scenario: Test scenario to simulate. Options:
                - "success": Normal operation
                - "rate_limit": Rate limit exceeded
                - "network_error": Network connectivity issue
                - "auth_error": Authentication failure
                - "dataset_not_found": Dataset doesn't exist
        """
        self.scenario = scenario
        self.datasets_loaded = []
        self.auth_attempts = []

    def load_dataset(self, name: str, **kwargs) -> dict[str, Any]:
        """Simulate dataset loading."""
        self.datasets_loaded.append({"name": name, "kwargs": kwargs})

        if self.scenario == "rate_limit":
            raise Exception("Rate limit exceeded: 10 requests per hour")
        elif self.scenario == "network_error":
            raise ConnectionError("Failed to reach HuggingFace Hub")
        elif self.scenario == "auth_error":
            raise ValueError("Invalid authentication token")
        elif self.scenario == "dataset_not_found":
            raise FileNotFoundError(f"Dataset {name} not found")

        # Return mock dataset
        return {
            "train": [{"instance_id": f"test_{i}"} for i in range(5)],
            "test": [{"instance_id": f"test_{i}"} for i in range(3)]
        }

    def login(self, token: str) -> bool:
        """Simulate HuggingFace login."""
        self.auth_attempts.append(token)

        if self.scenario == "auth_error":
            raise ValueError("Invalid token format")

        return True


class ProviderDouble:
    """Test double for AI model providers (OpenAI, Anthropic, etc.)."""

    def __init__(self, provider: str = "openai", scenario: str = "success"):
        """Initialize provider double.

        Args:
            provider: Provider name (openai, anthropic, ollama)
            scenario: Test scenario to simulate. Options:
                - "success": Normal operation
                - "rate_limit": Rate limit exceeded
                - "auth_error": Authentication failure
                - "timeout": Request timeout
                - "invalid_model": Model not found
                - "context_overflow": Context too long
        """
        self.provider = provider
        self.scenario = scenario
        self.requests_made = []
        self.tokens_used = 0

    def generate(self, prompt: str, model: str = None, **kwargs) -> str:
        """Simulate text generation."""
        request_info = {
            "prompt": prompt,
            "model": model,
            "kwargs": kwargs,
            "timestamp": time.time()
        }
        self.requests_made.append(request_info)

        if self.scenario == "rate_limit":
            raise Exception(f"{self.provider}: Rate limit exceeded")
        elif self.scenario == "auth_error":
            raise ValueError(f"{self.provider}: Invalid API key")
        elif self.scenario == "timeout":
            time.sleep(0.1)  # Small delay
            raise TimeoutError(f"{self.provider}: Request timeout")
        elif self.scenario == "invalid_model":
            raise ValueError(f"Model {model} not found")
        elif self.scenario == "context_overflow":
            raise ValueError("Context length exceeded maximum")

        # Simulate token usage
        self.tokens_used += len(prompt.split()) * 2

        # Return mock patch
        return """diff --git a/test.py b/test.py
@@ -1,3 +1,4 @@
+# Fixed by AI
 def test():
     pass"""

    def get_usage(self) -> dict[str, int]:
        """Get token usage statistics."""
        return {
            "prompt_tokens": self.tokens_used // 2,
            "completion_tokens": self.tokens_used // 2,
            "total_tokens": self.tokens_used
        }


class NetworkDouble:
    """Test double for network operations."""

    def __init__(self, scenario: str = "success"):
        """Initialize network double.

        Args:
            scenario: Test scenario to simulate. Options:
                - "success": Normal operation
                - "connection_error": Connection refused
                - "timeout": Request timeout
                - "dns_error": DNS resolution failure
                - "ghcr_blocked": GitHub Container Registry blocked
                - "git_rate_limit": GitHub API rate limit exceeded
                - "hf_rate_limit": HuggingFace Hub rate limit exceeded
                - "general_failure": General network failure
        """
        self.scenario = scenario
        self.requests_made = []
        self.rate_limit_hits = 0

    def get(self, url: str, **kwargs) -> dict[str, Any]:
        """Simulate HTTP GET request."""
        self.requests_made.append({"method": "GET", "url": url, "kwargs": kwargs})

        if self.scenario == "connection_error":
            raise ConnectionError("Connection refused")
        elif self.scenario == "timeout":
            raise TimeoutError("Request timeout after 30s")
        elif self.scenario == "dns_error":
            raise Exception("DNS resolution failed")
        elif self.scenario == "ghcr_blocked" and "ghcr.io" in url:
            raise Exception("Access to ghcr.io blocked by firewall")
        elif self.scenario == "git_rate_limit" and ("github.com" in url or "api.github.com" in url):
            self.rate_limit_hits += 1
            raise Exception("API rate limit exceeded (60 requests per hour for unauthenticated requests)")
        elif self.scenario == "hf_rate_limit" and "huggingface.co" in url:
            self.rate_limit_hits += 1
            raise Exception("Rate limit exceeded: 10 requests per hour")
        elif self.scenario == "general_failure":
            raise ConnectionError("Network unreachable")

        return {
            "status_code": 200,
            "content": b"Mock response",
            "headers": {"Content-Type": "application/json"}
        }

    def post(self, url: str, data: Any = None, json: Any = None, **kwargs) -> dict[str, Any]:
        """Simulate HTTP POST request."""
        self.requests_made.append({
            "method": "POST",
            "url": url,
            "data": data,
            "json": json,
            "kwargs": kwargs
        })

        # Apply same error scenarios as GET
        if self.scenario == "connection_error":
            raise ConnectionError("Connection refused")
        elif self.scenario == "timeout":
            raise TimeoutError("Request timeout after 30s")
        elif self.scenario == "git_rate_limit" and ("github.com" in url or "api.github.com" in url):
            self.rate_limit_hits += 1
            raise Exception("API rate limit exceeded")
        elif self.scenario == "general_failure":
            raise ConnectionError("Network unreachable")

        return {
            "status_code": 200,
            "content": b"Mock POST response",
            "headers": {"Content-Type": "application/json"}
        }


class FileSystemDouble:
    """Test double for file system operations."""

    def __init__(self, scenario: str = "success"):
        """Initialize file system double.

        Args:
            scenario: Test scenario to simulate. Options:
                - "success": Normal operation
                - "disk_full": No space left on device
                - "permission_denied": Permission denied
                - "cache_corrupted": Cache directory corrupted
        """
        self.scenario = scenario
        self.files_written = []
        self.files_read = []
        self.dirs_created = []

    def write_file(self, path: Path, content: str) -> None:
        """Simulate file write."""
        self.files_written.append({"path": str(path), "content": content})

        if self.scenario == "disk_full":
            raise OSError("No space left on device")
        elif self.scenario == "permission_denied":
            raise PermissionError(f"Permission denied: {path}")

    def read_file(self, path: Path) -> str:
        """Simulate file read."""
        self.files_read.append(str(path))

        if self.scenario == "cache_corrupted" and "cache" in str(path):
            raise ValueError("Cache file corrupted")
        elif self.scenario == "permission_denied":
            raise PermissionError(f"Permission denied: {path}")

        return "Mock file content"

    def mkdir(self, path: Path, **kwargs) -> None:
        """Simulate directory creation."""
        self.dirs_created.append(str(path))

        if self.scenario == "disk_full":
            raise OSError("No space left on device")
        elif self.scenario == "permission_denied":
            raise PermissionError(f"Permission denied: {path}")

    def get_disk_usage(self, path: Path = Path("/")) -> dict[str, int]:
        """Get disk usage statistics."""
        if self.scenario == "disk_full":
            return {
                "total": 1000000000,  # 1GB
                "used": 999999999,    # Almost full
                "free": 1             # 1 byte free
            }

        return {
            "total": 1000000000,  # 1GB
            "used": 500000000,    # 500MB
            "free": 500000000     # 500MB
        }


class PatchValidatorDouble:
    """Test double for patch validation operations."""

    def __init__(self, scenario: str = "success"):
        """Initialize patch validator double.

        Args:
            scenario: Test scenario to simulate. Options:
                - "success": Normal operation
                - "invalid_schema": Invalid patch schema/format
                - "too_large": Patch file too large (>1MB)
                - "too_large_env": Patch too large for environment variable
                - "encoding_error": Patch has encoding issues
                - "binary": Patch contains binary content
                - "conflict": Patch has merge conflicts
                - "apply_failed": Patch application fails
        """
        self.scenario = scenario
        self.patches_validated = []
        self.patches_applied = []
        self.validation_errors = []

    def validate_patch(self, patch_file: Path) -> bool:
        """Simulate patch validation."""
        self.patches_validated.append(str(patch_file))

        if self.scenario == "invalid_schema":
            error = "Invalid patch schema: missing 'instance_id' field"
            self.validation_errors.append(error)
            raise ValueError(error)
        elif self.scenario == "too_large":
            error = f"Patch file too large: {patch_file} is 2MB (max: 1MB)"
            self.validation_errors.append(error)
            raise ValueError(error)
        elif self.scenario == "too_large_env":
            error = "Patch content too large for environment variable (>128KB)"
            self.validation_errors.append(error)
            raise ValueError(error)
        elif self.scenario == "encoding_error":
            error = "Encoding error in patch: invalid UTF-8 at line 42"
            self.validation_errors.append(error)
            raise UnicodeDecodeError('utf-8', b'\\xff', 0, 1, 'invalid start byte')
        elif self.scenario == "binary":
            error = "Binary content detected in patch"
            self.validation_errors.append(error)
            raise ValueError(error)
        elif self.scenario == "conflict":
            error = "Merge conflict markers found in patch"
            self.validation_errors.append(error)
            raise ValueError(error)

        return True

    def apply_patch(self, patch_content: str, target_dir: Path = None) -> bool:
        """Simulate patch application."""
        patch_info = {
            "content_preview": patch_content[:100] if patch_content else None,
            "target_dir": str(target_dir) if target_dir else None
        }
        self.patches_applied.append(patch_info)

        if self.scenario == "apply_failed":
            error = "Failed to apply patch: hunk #1 FAILED at 42"
            self.validation_errors.append(error)
            raise RuntimeError(error)
        elif self.scenario == "conflict":
            error = "Patch conflicts with existing changes"
            self.validation_errors.append(error)
            raise RuntimeError(error)

        return True

    def get_patch_size(self, patch_file: Path) -> int:
        """Get patch file size in bytes."""
        if self.scenario == "too_large":
            return 2 * 1024 * 1024  # 2MB
        elif self.scenario == "too_large_env":
            return 256 * 1024  # 256KB
        return 1024  # 1KB for normal patches

    def validate_patch_content(self, content: str) -> bool:
        """Validate patch content directly."""
        if self.scenario == "encoding_error":
            raise UnicodeDecodeError('utf-8', b'\\xff', 0, 1, 'invalid start byte')
        elif self.scenario == "binary":
            raise ValueError("Binary content detected")
        elif self.scenario == "invalid_schema":
            raise ValueError("Invalid JSON structure")
        return True


class InstanceDouble:
    """Test double for instance-level operations."""

    def __init__(self, scenario: str = "success"):
        """Initialize instance double.

        Args:
            scenario: Test scenario to simulate. Options:
                - "success": Normal operation
                - "timeout": Instance evaluation timeout
                - "flaky": Flaky test detection
                - "invalid_id": Invalid instance ID
        """
        self.scenario = scenario
        self.instances_validated = []
        self.instances_run = []
        self.flaky_attempts = 0

    def validate_instance(self, instance_id: str) -> bool:
        """Validate instance ID."""
        self.instances_validated.append(instance_id)

        if self.scenario == "invalid_id":
            if instance_id.startswith("fake"):
                raise ValueError(f"Invalid instance ID: {instance_id}")

        return True

    def run_instance(self, instance_id: str, timeout: int = 30) -> dict[str, Any]:
        """Run instance evaluation."""
        self.instances_run.append({"id": instance_id, "timeout": timeout})

        if self.scenario == "timeout":
            if instance_id == "django-123":
                raise TimeoutError(f"Instance {instance_id} timed out after {timeout} minutes")

        elif self.scenario == "flaky":
            self.flaky_attempts += 1
            if self.flaky_attempts < 3:
                # Fail the first 2 attempts
                raise RuntimeError(f"Test failed (attempt {self.flaky_attempts}/3)")
            # Succeed on 3rd attempt

        return {"status": "success", "tests_passed": True}


class TestDoubleFactory:
    """Factory for creating and managing test doubles."""

    @staticmethod
    def create_docker_double(scenario: str = "success") -> DockerClientDouble:
        """Create a Docker client double."""
        return DockerClientDouble(scenario)

    @staticmethod
    def create_huggingface_double(scenario: str = "success") -> HuggingFaceDouble:
        """Create a HuggingFace double."""
        return HuggingFaceDouble(scenario)

    @staticmethod
    def create_provider_double(
        provider: str = "openai",
        scenario: str = "success"
    ) -> ProviderDouble:
        """Create a provider double."""
        return ProviderDouble(provider, scenario)

    @staticmethod
    def create_network_double(scenario: str = "success") -> NetworkDouble:
        """Create a network double."""
        return NetworkDouble(scenario)

    @staticmethod
    def create_filesystem_double(scenario: str = "success") -> FileSystemDouble:
        """Create a file system double."""
        return FileSystemDouble(scenario)

    @staticmethod
    def create_patch_double(scenario: str = "success") -> PatchValidatorDouble:
        """Create a patch validator double."""
        return PatchValidatorDouble(scenario)

    @staticmethod
    def create_instance_double(scenario: str = "success") -> InstanceDouble:
        """Create an instance double."""
        return InstanceDouble(scenario)

    @staticmethod
    def create_all_doubles(scenario: str = "success") -> dict[str, Any]:
        """Create all doubles with the same scenario.

        Returns:
            Dictionary with all test doubles
        """
        return {
            "docker": TestDoubleFactory.create_docker_double(scenario),
            "huggingface": TestDoubleFactory.create_huggingface_double(scenario),
            "openai": TestDoubleFactory.create_provider_double("openai", scenario),
            "anthropic": TestDoubleFactory.create_provider_double("anthropic", scenario),
            "network": TestDoubleFactory.create_network_double(scenario),
            "filesystem": TestDoubleFactory.create_filesystem_double(scenario),
            "patch": TestDoubleFactory.create_patch_double(scenario),
            "instance": TestDoubleFactory.create_instance_double(scenario)
        }
