# SPDX-License-Identifier: MIT
"""End-to-end tests for the CLI mapping subcommand."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

from core import mapping
from models import Contribution, MappingSet

cli = importlib.import_module("cli.main")


def _settings() -> SimpleNamespace:
    """Return minimal settings for the mapping CLI."""
    return SimpleNamespace(
        log_level="INFO",
        logfire_token=None,
        diagnostics=False,
        strict_mapping=False,
        strict=False,
        model="openai:gpt-5",
        models=None,
        use_local_cache=True,
        cache_mode="off",
        cache_dir=Path(".cache"),
        prompt_dir=Path("prompts"),
        mapping_data_dir=Path("tests/fixtures/catalogue"),
        mapping_sets=[
            MappingSet(
                name="Applications",
                file="applications.json",
                field="applications",
            ),
            MappingSet(
                name="Technologies",
                file="technologies.json",
                field="technologies",
            ),
        ],
    )


def _stub_map_set(*args, **kwargs):
    """Return deterministic mappings for test features."""
    session, set_name, _, features, *rest = args
    mapped = []
    for feat in features:
        mappings = dict(feat.mappings)
        if set_name == "applications":
            mapping_id = {"F1": "app1", "F2": "app2"}.get(feat.feature_id)
        else:
            mapping_id = {"F1": "tech1", "F2": "tech2"}.get(feat.feature_id)
        if mapping_id:
            mappings.setdefault(set_name, []).append(Contribution(item=mapping_id))
        mapped.append(feat.model_copy(update={"mappings": mappings}))
    return mapped


async def _stub_map_set_async(*args, **kwargs):
    return _stub_map_set(*args, **kwargs)


def test_cli_map_matches_golden(monkeypatch, tmp_path) -> None:
    """The map subcommand produces the locked golden output."""
    monkeypatch.setattr(mapping, "map_set", _stub_map_set_async)
    monkeypatch.setattr(cli, "load_settings", lambda _path=None: _settings())
    monkeypatch.setattr(cli, "_configure_logging", lambda *a, **k: None)

    input_file = Path("tests/fixtures/mapping_services.jsonl")
    output_file = tmp_path / "out.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main",
            "map",
            "--input-file",
            str(input_file),
            "--output-file",
            str(output_file),
        ],
    )

    cli.main()

    expected = Path("tests/golden/mapping_run.jsonl").read_text(encoding="utf-8")
    assert output_file.read_text(encoding="utf-8") == expected
