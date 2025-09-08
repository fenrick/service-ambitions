# SPDX-License-Identifier: MIT
"""Utilities for safe, resumable file writes.

This module centralises helper functions used to atomically update output files
and track processed service identifiers across CLI commands.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List

import logfire


def read_lines(path: Path) -> List[str]:
    """Return lines from ``path`` if it exists.

    Args:
        path: File to read.

    Returns:
        A list of lines without trailing newlines. Missing files yield an empty
        list.
    """
    with logfire.span("fs.read_lines", attributes={"path": str(path)}):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
            logfire.debug("Read lines", path=str(path), count=len(lines))
            return lines
        except FileNotFoundError:
            logfire.debug("File not found when reading lines", path=str(path))
            return []


def atomic_write(path: Path, lines: Iterable[str]) -> None:
    """Write ``lines`` to ``path`` atomically.

    Args:
        path: Destination file to replace.
        lines: Iterable of lines to write without trailing newlines.

    The function writes to ``path`` with a ``.tmp`` suffix, flushes and
    syncs the temporary file to disk, then performs :func:`os.replace` to
    ensure the final file is updated atomically.
    """
    with logfire.span("fs.atomic_write", attributes={"path": str(path)}):
        tmp_path = Path(f"{path}.tmp")
        # Ensure the destination directory exists before attempting the write.
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with open(tmp_path, "w", encoding="utf-8") as handle:
            for line in lines:
                handle.write(f"{line}\n")
                count += 1
            # Ensure data is written to disk before the atomic replace
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        logfire.debug("Atomic write complete", path=str(path), lines=count, bytes=size)


__all__ = ["atomic_write", "read_lines"]
