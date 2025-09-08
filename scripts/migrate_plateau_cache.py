#!/usr/bin/env python3
"""Migrate cache entries into plateau-specific directories."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import logfire
from pydantic_core import from_json

from constants import DEFAULT_CACHE_DIR


def _service_id_from_record(data: dict[str, object]) -> str | None:
    """Return service identifier from a JSON record, if present."""
    svc = data.get("service")
    if isinstance(svc, dict):
        sid = svc.get("service_id")
        return str(sid) if isinstance(sid, (str, int)) else None
    return None


def _plateaus_from_record(data: dict[str, object]) -> list[dict[str, object]]:
    """Return list of plateau dictionaries from a JSON record."""
    raw = data.get("plateaus")
    if not isinstance(raw, list):
        return []
    return [p for p in raw if isinstance(p, dict)]


def _update_feature_map_from_plateaus(
    feature_map: dict[str, str], plateaus: list[dict[str, object]], *, service: str
) -> None:
    """Populate ``feature_map`` from plateau entries."""
    for plateau in plateaus:
        level = plateau.get("plateau")
        if level is None:
            continue
        feats = plateau.get("features")
        if not isinstance(feats, list):
            continue
        for feat in feats:
            if not isinstance(feat, dict):
                continue
            fid = feat.get("feature_id")
            if not fid:
                continue
            feature_map[str(fid)] = str(level)
            logfire.debug(
                "Recorded feature plateau",
                service=service,
                feature=str(fid),
                plateau=str(level),
            )


def _parse_plateau_file(file: Path) -> dict[str, dict[str, str]]:
    """Return per-service feature->plateau mapping extracted from ``file``."""
    try:
        lines = file.read_text(encoding="utf-8").splitlines()
    except OSError:
        logfire.warning("Unable to read plateau output", path=str(file))
        return {}
    logfire.debug("Parsing plateau file", path=str(file))
    service_map: dict[str, dict[str, str]] = {}
    for line in lines:
        try:
            data = from_json(line)
        except ValueError:
            logfire.warning("Invalid JSON line", path=str(file))
            continue
        if not isinstance(data, dict):
            continue
        service = _service_id_from_record(data)
        if not service:
            continue
        feature_map = service_map.setdefault(service, {})
        plateaus = _plateaus_from_record(data)
        _update_feature_map_from_plateaus(feature_map, plateaus, service=service)
    return service_map


def _load_feature_plateaus(files: Iterable[Path]) -> dict[str, dict[str, str]]:
    """Return mapping of service to feature plateau levels."""
    service_map: dict[str, dict[str, str]] = {}
    for file in files:
        per_file = _parse_plateau_file(file)
        for svc, fmap in per_file.items():
            service_map.setdefault(svc, {}).update(fmap)
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
                logfire.debug(
                    "Moved cache entry",
                    service=service,
                    feature=fid,
                    plateau=plateau,
                    kind=kind,
                )
    for kind in ("features", "mappings"):
        unknown_dir = base / kind / "unknown"
        if unknown_dir.exists() and not any(unknown_dir.iterdir()):
            unknown_dir.rmdir()
            logfire.debug("Removed empty unknown directory", service=service, kind=kind)


def migrate(output_root: Path, cache_root: Path) -> None:
    """Migrate cache directories based on evolution output."""
    for context_dir in output_root.iterdir():
        if not context_dir.is_dir():
            continue
        context = context_dir.name
        outputs = list(context_dir.glob("*.jsonl"))
        if not outputs:
            continue
        logfire.debug("Processing context", context=context)
        service_map = _load_feature_plateaus(outputs)
        for service, feature_map in service_map.items():
            logfire.debug("Relocating cache entries", context=context, service=service)
            _move_entries(cache_root, context, service, feature_map)


def main() -> None:
    """CLI entrypoint."""
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output")
    cache = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_CACHE_DIR
    migrate(out, cache)


if __name__ == "__main__":
    main()
