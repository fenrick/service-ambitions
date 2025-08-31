#!/usr/bin/env python3
"""Migrate legacy cache layout to context-aware structure."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def migrate(root: Path, context: str) -> None:
    """Rewrite caches under ``root`` into context/service hierarchy."""

    for service_dir in root.iterdir():
        if not service_dir.is_dir():
            continue
        service = service_dir.name
        for file in service_dir.rglob("*.json"):
            try:
                data = json.load(file.open("r", encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            rel = file.relative_to(service_dir)
            parts = rel.parts
            if parts and parts[0] in {"features", "mappings"}:
                rel = Path(parts[0]) / "unknown" / Path(*parts[1:])
            new_path = root / context / service / rel
            new_path.parent.mkdir(parents=True, exist_ok=True)
            with new_path.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, separators=(",", ":"))
            file.unlink()


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: migrate_cache.py <context> [cache_dir]", file=sys.stderr)
        raise SystemExit(1)
    context = sys.argv[1]
    root = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".cache")
    migrate(root, context)


if __name__ == "__main__":
    main()
