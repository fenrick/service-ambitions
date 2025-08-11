"""Utilities for safe, resumable file writes.

This module centralises helper functions used to atomically update output files
and track processed service identifiers across CLI commands.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List


@logfire.instrument()
def read_lines(path: Path) -> List[str]:
    """Return lines from ``path`` if it exists.

    Args:
        path: File to read.

    Returns:
        A list of lines without trailing newlines. Missing files yield an empty
        list.
    """

    try:
        return path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:  # pragma: no cover - best effort
        return []


@logfire.instrument()
def atomic_write(path: Path, lines: Iterable[str]) -> None:
    """Write ``lines`` to ``path`` atomically.

    Args:
        path: Destination file to replace.
        lines: Iterable of lines to write without trailing newlines.

    The function writes to ``path`` with a ``.tmp`` suffix then performs
    :func:`os.replace` to ensure the final file is updated atomically.
    """

    tmp_path = Path(f"{path}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(f"{line}\n")
    os.replace(tmp_path, path)


__all__ = ["atomic_write", "read_lines"]
