# SPDX-License-Identifier: MIT
"""Tests for `_extract_service_id`."""

from io_utils.service_loader import _extract_service_id


def test_extract_service_id_from_partial_json() -> None:
    """Service ID can be parsed from truncated JSON lines."""
    line = '{"service_id": "svc1", "name": "Service"'
    assert _extract_service_id(line) == "svc1"


def test_extract_service_id_missing() -> None:
    """Missing identifiers return ``None`` for partial lines."""
    line = '{"name": "Service"'
    assert _extract_service_id(line) is None
