"""Project-wide constants and default paths.

This module centralises small constants that are imported across the
application. Keep this file minimal and free of side effects.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

DEFAULT_CACHE_DIR = (
    Path(os.path.expandvars(os.environ.get("XDG_CACHE_HOME", tempfile.gettempdir())))
    / "service-ambitions"
)

__all__ = ["DEFAULT_CACHE_DIR"]
