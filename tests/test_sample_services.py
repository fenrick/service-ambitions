# SPDX-License-Identifier: MIT
import json
from pathlib import Path


def test_sample_services_jsonl_is_valid() -> None:
    """All lines in sample-services.jsonl should be valid JSON."""
    lines = Path("sample-services.jsonl").read_text(encoding="utf-8").splitlines()
    for line in lines:
        assert json.loads(line) is not None
