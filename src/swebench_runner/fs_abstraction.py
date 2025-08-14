"""Filesystem abstraction seam for E2E doubles.

Allows tests to inject a filesystem double that can simulate disk usage
and errors without relying on the real machine state.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Protocol


class FileSystemProtocol(Protocol):
    def get_disk_usage(self, path: Path = Path("/")) -> dict[str, int]: ...


_active_fs: FileSystemProtocol | None = None


def set_filesystem(fs: FileSystemProtocol | None) -> None:
    global _active_fs
    _active_fs = fs


def get_free_disk_gb(path: Path | str = ".") -> float:
    """Return free disk space in GB for the given path, using injected FS if any."""
    if _active_fs is not None:
        try:
            usage = _active_fs.get_disk_usage(Path(path))
            free_bytes = int(usage.get("free", 0))
            return free_bytes / (1024 ** 3)
        except Exception:
            # Fall back to real usage on any error
            pass
    usage = shutil.disk_usage(str(path))
    return usage.free / (1024 ** 3)
