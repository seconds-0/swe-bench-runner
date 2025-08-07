"""Unit tests for docker_client.py - focus on connection error handling.

Tests specifically for:
- Docker connection error handling
- Platform-specific error messages
- Permission errors
- Network errors vs Docker errors
- Error classification for proper exit codes
"""

import platform
import sys
from unittest.mock import Mock, patch, MagicMock

import pytest
from docker.errors import APIError

from swebench_runner import exit_codes
from swebench_runner.docker_client import (
    check_docker_running,
    get_docker_client,
    set_docker_client_factory,
    reset_docker_client_factory,
    DockerClientProtocol
)


class TestDockerClientConnectionErrors:
    """Test Docker connection error handling."""

    def test_detects_docker_not_running(self):
        """Detect when Docker daemon is not running.
        
        Why: Most common user issue - forgetting to start Docker.
        """
        mock_client = Mock(spec=DockerClientProtocol)
        mock_client.ping.side_effect = APIError("Cannot connect to the Docker daemon")
        
        with pytest.raises(SystemExit) as exc_info:
            check_docker_running(mock_client)
        
        assert exc_info.value.code == exit_codes.DOCKER_NOT_FOUND

    def test_detects_permission_denied(self):
        """Detect Docker permission errors.
        
        Why: Linux users often need to be added to docker group.
        """
        mock_client = Mock(spec=DockerClientProtocol)
        mock_client.ping.side_effect = PermissionError("Permission denied")
        
        with pytest.raises(SystemExit) as exc_info:
            check_docker_running(mock_client)
        
        assert exc_info.value.code == exit_codes.DOCKER_NOT_FOUND

    def test_distinguishes_network_errors(self):
        """Distinguish network errors from Docker errors.
        
        Why: Network errors may be transient, Docker errors need Docker started.
        """
        mock_client = Mock(spec=DockerClientProtocol)
        mock_client.ping.side_effect = APIError("Network timeout")
        
        with pytest.raises(SystemExit) as exc_info:
            check_docker_running(mock_client)
        
        assert exc_info.value.code == exit_codes.NETWORK_ERROR

    def test_handles_docker_desktop_stopped(self):
        """Detect Docker Desktop stopped state.
        
        Why: macOS users may have Docker Desktop app closed.
        """
        mock_client = Mock(spec=DockerClientProtocol)
        mock_client.ping.side_effect = Exception("Docker Desktop is stopped")
        
        with pytest.raises(SystemExit) as exc_info:
            check_docker_running(mock_client)
        
        assert exc_info.value.code == exit_codes.DOCKER_NOT_FOUND


class TestDockerClientPlatformMessages:
    """Test platform-specific error messages."""

    @patch('platform.system')
    def test_macos_shows_docker_desktop_message(self, mock_system, capsys):
        """Show Docker Desktop message on macOS.
        
        Why: macOS users need to start Docker Desktop app.
        """
        mock_system.return_value = "Darwin"
        
        mock_client = Mock(spec=DockerClientProtocol)
        mock_client.ping.side_effect = APIError("Connection refused")
        
        with pytest.raises(SystemExit):
            check_docker_running(mock_client)
        
        captured = capsys.readouterr()
        assert "Docker Desktop" in captured.out or "Applications" in captured.out

    @patch('platform.system')
    def test_linux_shows_socket_message(self, mock_system, capsys):
        """Show docker.sock message on Linux.
        
        Why: Linux users need to check systemd service.
        """
        mock_system.return_value = "Linux"
        
        mock_client = Mock(spec=DockerClientProtocol)
        mock_client.ping.side_effect = APIError("Connection refused")
        
        with pytest.raises(SystemExit):
            check_docker_running(mock_client)
        
        captured = capsys.readouterr()
        assert "docker.sock" in captured.out or "daemon" in captured.out

    def test_permission_error_shows_usermod_command(self, capsys):
        """Show usermod command for permission errors.
        
        Why: Linux users need specific command to fix permissions.
        """
        mock_client = Mock(spec=DockerClientProtocol)
        mock_client.ping.side_effect = PermissionError("Permission denied")
        
        with pytest.raises(SystemExit):
            check_docker_running(mock_client)
        
        captured = capsys.readouterr()
        assert "usermod" in captured.out or "docker group" in captured.out


class TestDockerClientFactory:
    """Test Docker client factory for dependency injection."""

    def test_default_factory_returns_real_client(self):
        """Default factory should return real Docker client.
        
        Why: Production code needs real Docker client.
        """
        reset_docker_client_factory()
        
        with patch('docker.from_env') as mock_from_env:
            mock_from_env.return_value = Mock(spec=DockerClientProtocol)
            
            client = get_docker_client()
            
            mock_from_env.assert_called_once()
            assert client is not None

    def test_can_inject_custom_factory(self):
        """Allow injecting custom factory for testing.
        
        Why: Tests need to inject test doubles.
        """
        mock_client = Mock(spec=DockerClientProtocol)
        custom_factory = lambda: mock_client
        
        set_docker_client_factory(custom_factory)
        
        client = get_docker_client()
        
        assert client is mock_client
        
        # Clean up
        reset_docker_client_factory()

    def test_reset_factory_restores_default(self):
        """Reset should restore default behavior.
        
        Why: Tests shouldn't affect each other.
        """
        # Set custom factory
        set_docker_client_factory(lambda: Mock())
        
        # Reset
        reset_docker_client_factory()
        
        # Should use default
        with patch('docker.from_env') as mock_from_env:
            mock_from_env.return_value = Mock(spec=DockerClientProtocol)
            
            client = get_docker_client()
            
            mock_from_env.assert_called_once()


class TestDockerClientErrorParsing:
    """Test parsing of various Docker error messages."""

    def test_parses_connection_refused_error(self):
        """Parse connection refused errors correctly.
        
        Why: Most common error when Docker not running.
        """
        errors = [
            "Connection refused",
            "Cannot connect to the Docker daemon",
            "Is the docker daemon running?",
        ]
        
        for error_msg in errors:
            mock_client = Mock(spec=DockerClientProtocol)
            mock_client.ping.side_effect = APIError(error_msg)
            
            with pytest.raises(SystemExit) as exc_info:
                check_docker_running(mock_client)
            
            assert exc_info.value.code == exit_codes.DOCKER_NOT_FOUND

    def test_parses_network_timeout_errors(self):
        """Parse network timeout errors.
        
        Why: Network issues need different handling than Docker issues.
        """
        errors = [
            "network timeout",
            "connection timed out",
            "DNS resolution failed",
            "unreachable",
        ]
        
        for error_msg in errors:
            mock_client = Mock(spec=DockerClientProtocol)
            mock_client.ping.side_effect = APIError(error_msg)
            
            with pytest.raises(SystemExit) as exc_info:
                check_docker_running(mock_client)
            
            assert exc_info.value.code == exit_codes.NETWORK_ERROR

    def test_handles_unexpected_errors_safely(self):
        """Handle unexpected errors without crashing.
        
        Why: New Docker versions may have different error messages.
        """
        mock_client = Mock(spec=DockerClientProtocol)
        mock_client.ping.side_effect = Exception("Unexpected error format")
        
        with pytest.raises(SystemExit) as exc_info:
            check_docker_running(mock_client)
        
        # Should still exit with Docker error
        assert exc_info.value.code == exit_codes.DOCKER_NOT_FOUND


class TestDockerClientIntegration:
    """Test integration with docker_run module."""

    def test_docker_run_uses_client_abstraction(self):
        """docker_run should use docker_client abstraction.
        
        Why: Enables test double injection for E2E tests.
        """
        from swebench_runner.docker_run import check_docker_running as run_check
        
        # Inject test double
        mock_client = Mock(spec=DockerClientProtocol)
        mock_client.ping.return_value = True
        set_docker_client_factory(lambda: mock_client)
        
        # Should use injected client
        run_check()  # Should not raise
        
        # Clean up
        reset_docker_client_factory()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])