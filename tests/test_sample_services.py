# SPDX-License-Identifier: MIT
from pathlib import Path

from pydantic_core import from_json


def test_sample_services_jsonl_is_valid() -> None:
    """All lines in sample-services.jsonl should be valid JSON."""
    lines = Path("sample-services.jsonl").read_text(encoding="utf-8").splitlines()
    for line in lines:
        assert from_json(line) is not None
