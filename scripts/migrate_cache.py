#!/usr/bin/env python3
"""Migrate legacy cache layout to context-aware structure."""
from __future__ import annotations

import sys
from pathlib import Path

from pydantic_core import from_json, to_json


def migrate(root: Path, context: str) -> None:
    """Rewrite caches under ``root`` into context/service hierarchy."""

    for service_dir in root.iterdir():
        if not service_dir.is_dir():
            # Skip non-directory entries in the cache root
            continue
        service = service_dir.name
        for file in service_dir.rglob("*.json"):
            try:
                with file.open("r", encoding="utf-8") as fh:
                    data = from_json(fh.read())
            except ValueError:
                # Ignore files containing invalid JSON
                continue
            rel = file.relative_to(service_dir)
            parts = rel.parts
            if parts and parts[0] in {"features", "mappings"}:
                # Insert "unknown" placeholder for feature/mapping caches
                rel = Path(parts[0]) / "unknown" / Path(*parts[1:])
            new_path = root / context / service / rel
            new_path.parent.mkdir(parents=True, exist_ok=True)
            with new_path.open("w", encoding="utf-8") as fh:
                fh.write(to_json(data).decode("utf-8"))
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
