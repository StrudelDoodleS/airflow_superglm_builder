"""Small cross-platform advisory file lock for trusted-host workflows."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Iterator


def _is_windows() -> bool:
    return sys.platform == "win32"


def _ensure_lock_byte(handle: BinaryIO) -> None:
    handle.seek(0)
    if handle.read(1) == b"":
        handle.seek(0)
        handle.write(b"\0")
    handle.seek(0)


def _acquire(handle: BinaryIO) -> None:
    if _is_windows():
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _release(handle: BinaryIO) -> None:
    if _is_windows():
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def exclusive_file_lock(path: str | Path) -> Iterator[BinaryIO]:
    """Hold an exclusive advisory lock on one sentinel file."""
    lock_path = Path(path).expanduser().resolve()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    with os.fdopen(descriptor, "r+b", buffering=0) as handle:
        _ensure_lock_byte(handle)
        _acquire(handle)
        try:
            yield handle
        finally:
            _release(handle)
