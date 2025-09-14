# SPDX-License-Identifier: MIT
"""Tests for importing modules without Poetry bootstrap."""

from __future__ import annotations

import os
import subprocess
import sys

os.environ.setdefault("SA_OPENAI_API_KEY", "test-key")


def test_modules_import_without_poetry():
    """Ensure src modules import when ``PYTEST_PYPROJECT=0``."""
    env = {
        **os.environ,
        "PYTEST_PYPROJECT": "0",
        "SA_OPENAI_API_KEY": "test-key",
        "OPENAI_API_KEY": "test-key",
    }
    code = (
        "import tests.conftest as c; "
        "import models as m; "
        "import sys; from pathlib import Path; "
        "p=Path.cwd()/'src'; "
        "assert str(p) in sys.path and m.__file__.startswith(str(p))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
