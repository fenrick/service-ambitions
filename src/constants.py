from __future__ import annotations

import os
from pathlib import Path

DEFAULT_CACHE_DIR = (
    Path(os.path.expandvars(os.environ.get("XDG_CACHE_HOME", "/tmp")))
    / "service-ambitions"
)

__all__ = ["DEFAULT_CACHE_DIR"]
