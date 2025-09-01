"""Tests for migrate_plateau_cache script."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any, cast


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "migrate_plateau_cache", Path("scripts/migrate_plateau_cache.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_migrate_moves_feature_and_mapping(tmp_path: Path) -> None:
    module = cast(Any, _load_module())
    output = tmp_path / "output" / "ctx"
    cache = tmp_path / ".cache" / "ctx" / "svc"
    output.mkdir(parents=True)
    cache_feat = cache / "features" / "unknown" / "f1"
    cache_feat.mkdir(parents=True)
    (cache_feat / "data.json").write_text("{}", encoding="utf-8")
    cache_map = cache / "mappings" / "unknown" / "f1" / "apps"
    cache_map.mkdir(parents=True)
    (cache_map / "x.json").write_text("{}", encoding="utf-8")
    record = {
        "service": {"service_id": "svc"},
        "plateaus": [{"plateau": 1, "features": [{"feature_id": "f1"}]}],
    }
    with (output / "svc.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(record))
    module.migrate(tmp_path / "output", tmp_path / ".cache")
    assert (cache / "features" / "1" / "f1" / "data.json").exists()
    assert not (cache / "features" / "unknown").exists()
    assert (cache / "mappings" / "1" / "f1" / "apps" / "x.json").exists()
    assert not (cache / "mappings" / "unknown").exists()
