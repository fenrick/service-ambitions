# SPDX-License-Identifier: MIT
"""Utilities for writing quarantined payloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import logfire
from pydantic_core import from_json, to_json

from observability import telemetry

MANIFEST = "manifest.json"
ALLOWED_KINDS = {"json_parse_error", "unknown_ids", "schema_mismatch", "timeout"}


class QuarantineWriter:
    """Persist invalid payloads and maintain a manifest."""

    def __init__(self, base_dir: Path | str = Path("quarantine")) -> None:
        self.base_dir = Path(base_dir)

    def write(self, set_name: str, service_id: str, kind: str, payload: Any) -> Path:
        """Persist ``payload`` and update manifest.

        Parameters
        ----------
        set_name:
            Mapping set or processing stage name.
        service_id:
            Identifier of the service associated with ``payload``.
        kind:
            Nature of the quarantine, such as ``json_parse_error`` or ``unknown_ids``.
        payload:
            Offending data to store.

        Returns
        -------
        Path
            Location of the written payload file.
        """

        if kind not in ALLOWED_KINDS:
            raise ValueError(f"Unsupported quarantine kind: {kind}")

        qdir = self.base_dir / service_id / set_name
        qdir.mkdir(parents=True, exist_ok=True)

        index = sum(1 for _ in qdir.glob(f"{kind}_*.json")) + 1
        file_path = qdir / f"{kind}_{index}.json"

        if isinstance(payload, str):
            file_path.write_text(payload, encoding="utf-8")
        else:
            file_path.write_text(
                to_json(payload, indent=2).decode("utf-8"),
                encoding="utf-8",
            )

        manifest_path = qdir / MANIFEST
        if manifest_path.exists():
            manifest = from_json(manifest_path.read_text(encoding="utf-8"))
        else:
            manifest = {}
        entry = manifest.setdefault(kind, {"count": 0, "examples": []})
        entry["count"] += 1
        if len(entry["examples"]) < 3:
            entry["examples"].append(payload)
        manifest_path.write_text(
            to_json(manifest, indent=2).decode("utf-8"),
            encoding="utf-8",
        )

        logfire.warning(
            "Quarantined payload",
            path=str(file_path),
            kind=kind,
            service_id=service_id,
            set_name=set_name,
        )

        telemetry.record_quarantine(file_path)
        return file_path


__all__ = ["QuarantineWriter"]
