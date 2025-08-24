# SPDX-License-Identifier: MIT
from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path

from telemetry import (
    has_quarantines,
    print_summary,
    record_mapping_set,
    record_quarantine,
    reset,
)


def test_summary_and_quarantine(monkeypatch) -> None:
    """Metrics aggregate and report a summary."""

    reset()
    record_mapping_set(
        "applications",
        features=2,
        mapped_ids=4,
        unknown_ids=1,
        retries=1,
        latency=0.5,
        tokens=100,
    )
    record_quarantine(Path("q/unknown.json"))
    buf = StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    print_summary()
    output = buf.getvalue()
    assert "applications" in output
    assert "features=2" in output
    assert has_quarantines()
    reset()
