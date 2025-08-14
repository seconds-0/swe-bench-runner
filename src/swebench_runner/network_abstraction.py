"""Injectable network abstraction for E2E tests and runtime checks.

Provides a minimal DI seam so doubles can simulate registry/git/network
conditions without relying on external connectivity.
"""

from __future__ import annotations

from typing import Any, Protocol


class NetworkClientProtocol(Protocol):
    def get(self, url: str, **kwargs: Any) -> dict[str, Any]: ...


_active_network_client: NetworkClientProtocol | None = None


def set_network_client(client: NetworkClientProtocol | None) -> None:
    global _active_network_client
    _active_network_client = client


def check_ghcr_access() -> None:
    """Raise if GHCR access is blocked (used by E2E ghcr_blocked)."""
    if _active_network_client is None:
        return
    _active_network_client.get("https://ghcr.io/v2/")


def check_github_rate_limit() -> None:
    """Raise if GitHub API rate limit is exceeded (used by E2E git_rate_limit)."""
    if _active_network_client is None:
        return
    _active_network_client.get("https://api.github.com/rate_limit")


def check_general_connectivity() -> None:
    """Raise on general network failures."""
    if _active_network_client is None:
        return
    _active_network_client.get("https://example.com/")
