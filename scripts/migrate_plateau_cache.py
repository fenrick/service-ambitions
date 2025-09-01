#!/usr/bin/env python3
"""Migrate cache entries into plateau-specific directories."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

from pydantic_core import from_json


def _load_feature_plateaus(files: Iterable[Path]) -> dict[str, dict[str, str]]:
    """Return mapping of service to feature plateau levels."""

    service_map: dict[str, dict[str, str]] = {}
    for file in files:
        try:
            lines = file.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                data = from_json(line)
            except ValueError:
                # Skip invalid JSON lines
                continue
            service = data.get("service", {}).get("service_id")
            if not service:
                continue
            feature_map = service_map.setdefault(service, {})
            for plateau in data.get("plateaus", []):
                level = plateau.get("plateau")
                if level is None:
                    continue
                for feat in plateau.get("features", []):
                    fid = feat.get("feature_id")
                    if fid:
                        feature_map[fid] = str(level)
    return service_map


def _move_entries(
    cache_root: Path, context: str, service: str, feature_map: dict[str, str]
) -> None:
    """Relocate feature and mapping caches using ``feature_map``."""

    base = cache_root / context / service
    for fid, plateau in feature_map.items():
        for kind in ("features", "mappings"):
            src = base / kind / "unknown" / fid
            if src.exists():
                dest = base / kind / plateau / fid
                dest.parent.mkdir(parents=True, exist_ok=True)
                src.rename(dest)
    for kind in ("features", "mappings"):
        unknown_dir = base / kind / "unknown"
        if unknown_dir.exists() and not any(unknown_dir.iterdir()):
            unknown_dir.rmdir()


def migrate(output_root: Path, cache_root: Path) -> None:
    """Migrate cache directories based on evolution output."""

    for context_dir in output_root.iterdir():
        if not context_dir.is_dir():
            continue
        context = context_dir.name
        outputs = list(context_dir.glob("*.jsonl"))
        if not outputs:
            continue
        service_map = _load_feature_plateaus(outputs)
        for service, feature_map in service_map.items():
            _move_entries(cache_root, context, service, feature_map)


def main() -> None:
    """CLI entrypoint."""

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output")
    cache = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".cache")
    migrate(out, cache)


if __name__ == "__main__":
    main()
