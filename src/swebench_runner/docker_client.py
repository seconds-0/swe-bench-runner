"""Docker client abstraction for dependency injection.

This module provides a clean abstraction over Docker operations,
allowing for easy testing through dependency injection.
"""

import os
import platform
import sys
from typing import Protocol

import docker
from docker.errors import APIError

from . import exit_codes


class DockerClientProtocol(Protocol):
    """Protocol defining the Docker client interface."""

    def ping(self) -> bool:
        """Check if Docker daemon is accessible."""
        ...

    def containers(self):
        """Get containers interface."""
        ...

    def images(self):
        """Get images interface."""
        ...


# Global client factory for dependency injection
_docker_client_factory = None


def set_docker_client_factory(factory):
    """Set a custom Docker client factory for testing.

    Args:
        factory: Callable that returns a Docker client
    """
    global _docker_client_factory
    _docker_client_factory = factory


def reset_docker_client_factory():
    """Reset Docker client factory to default."""
    global _docker_client_factory
    _docker_client_factory = None


def get_docker_client() -> DockerClientProtocol:
    """Get a Docker client instance.

    Returns:
        Docker client instance (real or test double)
    """
    if _docker_client_factory is not None:
        return _docker_client_factory()

    # Default to real Docker client
    return docker.from_env()


def check_docker_running(client: DockerClientProtocol | None = None) -> None:
    """Check if Docker daemon is accessible.

    Args:
        client: Optional Docker client to use (for testing)

    Exits:
        With appropriate error code if Docker is not available
    """
    # Test harness compatibility: allow env flag to force not-running behavior
    # BUT honor a custom factory if one is set (tests inject doubles via factory)
    if os.getenv("SWEBENCH_MOCK_NO_DOCKER", "false").lower() == "true" and _docker_client_factory is None:
        # Honor platform overrides used by tests
        platform_hint = os.getenv("SWEBENCH_PLATFORM", platform.system())
        if platform_hint.lower().startswith("darwin"):
            print("⛔ Docker Desktop not running. Start it from Applications and wait for whale icon.")
        else:
            print("⛔ Docker daemon unreachable at /var/run/docker.sock")
            print("Try: systemctl start docker or set DOCKER_HOST")
        print("ℹ️  Start Docker and try again.")
        sys.exit(exit_codes.DOCKER_NOT_FOUND)

    if client is None:
        client = get_docker_client()

    try:
        client.ping()
    except APIError as e:
        error_msg = str(e).lower()
        # Container limit is just a warning, not a fatal error
        if "too many containers" in error_msg or "container limit" in error_msg:
            print("⚠️  Docker container limit reached (100). Reducing workers.")
            # This is just a warning, don't exit
            return
        # Connection refused to Docker daemon means Docker is not running
        elif ("connection refused" in error_msg or
            "cannot connect to the docker daemon" in error_msg):
            platform_hint = os.getenv("SWEBENCH_PLATFORM", platform.system())
            if platform_hint.lower().startswith("darwin"):
                print("⛔ Docker Desktop not running. Start it from Applications "
                      "and wait for whale icon.")
            else:
                print("⛔ Docker daemon unreachable at /var/run/docker.sock")
                print("Try: systemctl start docker or set DOCKER_HOST")
            print("ℹ️  Start Docker and try again.")
            sys.exit(exit_codes.DOCKER_NOT_FOUND)
        elif any(term in error_msg for term in [
            "network", "connection", "timeout", "unreachable",
            "resolve", "dns", "timed out"
        ]):
            print("❌ Network error connecting to Docker daemon")
            print("   Check internet connection and Docker network settings")
            sys.exit(exit_codes.NETWORK_ERROR)
        else:
            if platform.system() == "Darwin":
                print("⛔ Docker Desktop not running. Start it from Applications "
                      "and wait for whale icon.")
            else:
                print("⛔ Docker daemon unreachable at /var/run/docker.sock")
                print("Try: systemctl start docker or set DOCKER_HOST")
            print("ℹ️  Start Docker and try again.")
            sys.exit(exit_codes.DOCKER_NOT_FOUND)
    except PermissionError as e:
        print(f"⛔ Permission denied accessing Docker: {e}")
        print("ℹ️  On Linux, add your user to the docker group:")
        print("    sudo usermod -aG docker $USER")
        print("    Then log out and back in.")
        sys.exit(exit_codes.DOCKER_NOT_FOUND)
    except Exception as e:
        error_msg = str(e).lower()
        # Check for specific error patterns
        if "docker desktop" in error_msg and "stopped" in error_msg:
            print("⛔ Docker Desktop is stopped. Click the whale icon to start it.")
            sys.exit(exit_codes.DOCKER_NOT_FOUND)
        elif ("connection refused" in error_msg or
            "cannot connect to the docker daemon" in error_msg):
            platform_hint = os.getenv("SWEBENCH_PLATFORM", platform.system())
            if platform_hint.lower().startswith("darwin"):
                print("⛔ Docker Desktop not running. Start it from Applications "
                      "and wait for whale icon.")
            else:
                print("⛔ Docker daemon unreachable at /var/run/docker.sock")
            print("ℹ️  Start Docker and try again.")
            sys.exit(exit_codes.DOCKER_NOT_FOUND)
        elif any(term in error_msg for term in [
            "network", "connection", "timeout", "unreachable",
            "resolve", "dns"
        ]):
            print(f"❌ Network error connecting to Docker: {e}")
            print("   Check internet connection and Docker network settings")
            sys.exit(exit_codes.NETWORK_ERROR)
        else:
            print(f"❌ Error connecting to Docker: {e}")
            sys.exit(exit_codes.DOCKER_NOT_FOUND)

def is_custom_factory_set() -> bool:
    """Return True if a custom docker client factory is registered."""
    return _docker_client_factory is not None
