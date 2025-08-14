"""Injectable instance abstraction for E2E tests.

Allows tests to validate instance IDs and simulate timeouts/flaky behavior.
"""

from __future__ import annotations

from typing import Any, Protocol


class InstanceClientProtocol(Protocol):
    def validate_instance(self, instance_id: str) -> bool: ...
    def run_instance(self, instance_id: str, timeout: int = 30) -> dict[str, Any]: ...


_active_instance_client: InstanceClientProtocol | None = None


def set_instance_client(client: InstanceClientProtocol | None) -> None:
    global _active_instance_client
    _active_instance_client = client


def get_instance_client() -> InstanceClientProtocol | None:
    """Expose the active instance client (for E2E detection only)."""
    return _active_instance_client


def validate_instance_id(instance_id: str) -> None:
    if _active_instance_client is None:
        return
    try:
        _active_instance_client.validate_instance(instance_id)
    except Exception as e:
        # Re-raise explicitly for clarity in callers/tests
        raise e

def record_instance_timeout(instance_id: str, timeout_mins: int) -> None:
    if _active_instance_client is None:
        return
    try:
        _active_instance_client.run_instance(instance_id, timeout=timeout_mins)
    except Exception as e:
        # Propagate so callers can classify/exit appropriately
        raise e
