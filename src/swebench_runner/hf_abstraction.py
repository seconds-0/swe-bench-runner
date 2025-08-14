"""HuggingFace dataset abstraction for tests and runtime.

Allows E2E tests to inject a lightweight client that simulates rate limits
and other behaviors without depending on external libraries.
"""

from __future__ import annotations

from typing import Any, Protocol


class HFClientProtocol(Protocol):
    def load_dataset(self, name: str, split: str, download_mode: str) -> Any: ...
    def login(self, token: str) -> bool: ...


_active_client: HFClientProtocol | None = None


def set_hf_client(client: HFClientProtocol | None) -> None:
    global _active_client
    _active_client = client


def load_dataset(name: str, split: str, download_mode: str) -> Any:
    """Load dataset via injected client if present, else via datasets library."""
    if _active_client is not None:
        return _active_client.load_dataset(name, split=split, download_mode=download_mode)

    # Fallback to real library
    try:
        from datasets import load_dataset as _real_load_dataset  # type: ignore
    except Exception as e:
        # Surface import issues to caller; CLI classifies appropriately
        raise e
    return _real_load_dataset(name, split=split, download_mode=download_mode)
