"""Schema version migration utilities."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict


def migrate_record(
    from_version: str, to_version: str, data: Dict[str, Any]
) -> Dict[str, Any]:
    """Migrate a record between schema versions.

    Args:
        from_version: Version of the input ``data``.
        to_version: Desired schema version. ``1.0`` → ``1.x`` is supported.
        data: Mapping conforming to ``from_version`` of the schema.

    Returns:
        The migrated record. A deep copy is returned to avoid mutating ``data``.

    Raises:
        ValueError: If ``from_version`` and ``to_version`` are not supported.

    Examples:
        >>> migrate_record(
        ...     "1.0",
        ...     "1.1",
        ...     {
        ...         "schema_version": "1.0",
        ...         "service": {},
        ...         "plateaus": [
        ...             {
                "plateau": 1,
                "plateau_name": "p1",
                "service_description": "d",
            }
        ],
        ...     },
        ... )
        {
            'meta': {
                'schema_version': '1.1',
                'run_id': '',
                'seed': None,
                'models': {},
                'web_search': False,
                'mapping_types': [],
                'created': '2024-01-01T00:00:00+00:00',
            },
            'service': {},
            'plateaus': [
                {
                    'plateau': 1,
                    'plateau_name': 'p1',
                    'description': 'd',
                    'features': [],
                }
            ],
        }
    """

    if from_version == to_version:
        # Already on the requested version; return a copy to preserve immutability.
        return deepcopy(data)

    if from_version == "1.0" and to_version.startswith("1."):
        # Perform the 1.0 → 1.x migration.
        migrated = deepcopy(data)
        migrated.pop("schema_version", None)
        migrated["meta"] = {
            "schema_version": to_version,
            "run_id": "",
            "seed": None,
            "models": {},
            "web_search": False,
            "mapping_types": [],
            "created": datetime.now(timezone.utc).isoformat(),
        }
        for plateau in migrated.get("plateaus", []):
            if "service_description" in plateau:
                plateau["description"] = plateau.pop("service_description")
            plateau.setdefault("features", [])
        return migrated

    # Any other version combination is unsupported.
    raise ValueError(f"Unsupported schema migration: {from_version} → {to_version}")
