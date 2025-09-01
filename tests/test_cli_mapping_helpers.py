# SPDX-License-Identifier: MIT
"""Unit tests for mapping CLI helper functions."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import cli_mapping
import mapping
from models import Contribution, MappingSet, ServiceEvolution


def _settings() -> SimpleNamespace:
    """Return minimal settings for helper tests."""

    return SimpleNamespace(
        diagnostics=False,
        strict_mapping=False,
        mapping_data_dir=Path("tests/fixtures/catalogue"),
        mapping_sets=[
            MappingSet(
                name="Applications", file="applications.json", field="applications"
            ),
            MappingSet(
                name="Technologies", file="technologies.json", field="technologies"
            ),
        ],
    )


def _stub_map_set(*args, **kwargs):
    """Return deterministic mappings for test features."""

    session, set_name, _, features = args[:4]
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


def test_load_catalogue_invokes_loader(monkeypatch) -> None:
    """Catalogue loading delegates to loader helpers."""

    calls = {}

    def fake_configure(path):
        calls["configure"] = path

    def fake_load(mapping_sets):
        calls["load"] = mapping_sets
        return {"applications": []}, "hash"

    monkeypatch.setattr(cli_mapping, "configure_mapping_data_dir", fake_configure)
    monkeypatch.setattr(cli_mapping, "load_mapping_items", fake_load)

    settings = SimpleNamespace(mapping_data_dir=Path("cat"), mapping_sets=[1])
    items, catalogue_hash = cli_mapping.load_catalogue(None, settings)

    assert calls["configure"] == Path("cat")
    assert calls["load"] == [1]
    assert items == {"applications": []}
    assert catalogue_hash == "hash"


@pytest.mark.asyncio
async def test_remap_features_populates_mappings(monkeypatch) -> None:
    """Feature remapping populates plateau mappings."""

    settings = _settings()
    items, catalogue_hash = cli_mapping.load_catalogue(None, settings)

    text = Path("tests/fixtures/mapping_services.jsonl").read_text(encoding="utf-8")
    evolutions = [
        ServiceEvolution.model_validate_json(line)
        for line in text.splitlines()
        if line.strip()
    ]

    monkeypatch.setattr(mapping, "map_set", _stub_map_set_async)

    await cli_mapping.remap_features(evolutions, items, settings, "off", catalogue_hash)

    plateau = evolutions[0].plateaus[0]
    apps = plateau.mappings["applications"][0]
    techs = plateau.mappings["technologies"][0]
    assert apps.id == "app1"
    assert techs.id == "tech1"


@pytest.mark.asyncio
async def test_write_output_matches_golden(monkeypatch, tmp_path) -> None:
    """Writing mapped output produces the locked golden file."""

    settings = _settings()
    items, catalogue_hash = cli_mapping.load_catalogue(None, settings)
    text = Path("tests/fixtures/mapping_services.jsonl").read_text(encoding="utf-8")
    evolutions = [
        ServiceEvolution.model_validate_json(line)
        for line in text.splitlines()
        if line.strip()
    ]
    monkeypatch.setattr(mapping, "map_set", _stub_map_set_async)
    await cli_mapping.remap_features(evolutions, items, settings, "off", catalogue_hash)

    out_path = tmp_path / "out.jsonl"
    cli_mapping.write_output(evolutions, out_path)

    expected = Path("tests/golden/mapping_run.jsonl").read_text(encoding="utf-8")
    assert out_path.read_text(encoding="utf-8") == expected
