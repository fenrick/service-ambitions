"""Utilities for migrating service JSONL files to the latest schema."""

from __future__ import annotations

import json
from pathlib import Path


def migrate_record(data: dict) -> dict:
    """Return ``data`` converted from 1.0 layout to the 1.x schema.

    The migration performs a best-effort rename of legacy fields:

    * ``id`` → ``service_id``
    * ``jobs`` → ``jobs_to_be_done`` with string jobs wrapped in objects
    * feature ``id`` → ``feature_id``
    * ``customer`` → ``customer_type``
    """

    result: dict = {
        "service_id": data.get("service_id") or data.get("id"),
        "name": data.get("name") or data.get("service"),
        "description": data.get("description", ""),
    }

    if "customer_type" in data:
        result["customer_type"] = data["customer_type"]
    elif "customer" in data:
        result["customer_type"] = data["customer"]

    jobs = data.get("jobs_to_be_done") or data.get("jobs") or []
    result["jobs_to_be_done"] = [
        job if isinstance(job, dict) else {"name": job} for job in jobs
    ]

    features = data.get("features", [])
    migrated_features = []
    for feat in features:
        migrated_features.append(
            {
                "feature_id": feat.get("feature_id") or feat.get("id"),
                "name": feat.get("name"),
                "description": feat.get("description", ""),
            }
        )
    result["features"] = migrated_features
    return result


def migrate_jsonl(input_path: Path, output_path: Path) -> int:
    """Migrate services in ``input_path`` writing results to ``output_path``.

    Args:
        input_path: Location of the legacy JSONL file.
        output_path: Destination for migrated records.

    Returns:
        Number of records written to ``output_path``.
    """

    count = 0
    with (
        Path(input_path).open("r", encoding="utf-8") as src,
        Path(output_path).open("w", encoding="utf-8") as dst,
    ):
        for line in src:
            if not line.strip():
                continue
            migrated = migrate_record(json.loads(line))
            dst.write(json.dumps(migrated) + "\n")
            count += 1
    return count


__all__ = ["migrate_record", "migrate_jsonl"]
