# SPDX-License-Identifier: MIT
"""Tests for schema migration utilities."""

import pytest

from migrations.schema_migration import migrate_record


def test_migrate_from_1_0_to_1_1() -> None:
    """1.0 → 1.1 adds defaults and renames fields."""
    source = {
        "schema_version": "1.0",
        "service": {},
        "plateaus": [
            {
                "plateau": 1,
                "plateau_name": "p1",
                "service_description": "desc",
            }
        ],
    }

    migrated = migrate_record("1.0", "1.1", source)

    assert migrated["meta"]["schema_version"] == "1.1"
    plateau = migrated["plateaus"][0]
    assert "service_description" not in plateau
    assert plateau["description"] == "desc"
    assert plateau["features"] == []


def test_rejects_unsupported_versions() -> None:
    """Unrecognised version pairs raise ``ValueError``."""
    with pytest.raises(ValueError):
        migrate_record("0.9", "1.0", {})
