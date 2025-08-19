"""Utilities for migrating JSONL records between schema versions."""

from __future__ import annotations

from typing import Any, Dict


def migrate_record(payload: Dict[str, Any], source: str, target: str) -> Dict[str, Any]:
    """Return ``payload`` upgraded from ``source`` to ``target``.

    Args:
        payload: A JSON-serialisable mapping representing one record.
        source: The schema version currently used by ``payload``.
        target: The desired schema version.

    The current implementation performs a minimal migration by updating the
    ``schema_version`` field. It can be extended to handle structural changes
    between revisions.

    Returns:
        The migrated record.
    """

    if source == target:
        return payload

    upgraded = {**payload}
    upgraded["schema_version"] = target
    return upgraded


__all__ = ["migrate_record"]
