# SPDX-License-Identifier: MIT
import asyncio
import json
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic_core import from_json

from core import mapping
from core.canonical import canonicalise_record
from core.conversation import ConversationSession
from core.mapping import MapSetParams
from io_utils import loader
from models import MappingResponse, MappingSet, ServiceEvolution


class DummySession:
    """Session returning queued responses for ``ask_async``."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def derive(self) -> "DummySession":
        return self

    async def ask_async(self, prompt: str) -> MappingResponse:
        resp = self._responses.pop(0)
        return MappingResponse.model_validate_json(resp)


def _load_evolutions() -> list[ServiceEvolution]:
    text = Path("tests/fixtures/mapping_services.jsonl").read_text(encoding="utf-8")
    return [
        ServiceEvolution.model_validate_json(line)
        for line in text.splitlines()
        if line.strip()
    ]


def _params(**kwargs: Any) -> MapSetParams:
    return MapSetParams(
        service_name="svc", service_description="desc", plateau=1, **kwargs
    )


def test_mapping_run_matches_golden(tmp_path) -> None:
    """End-to-end mapping run matches locked golden output."""

    loader.configure_mapping_data_dir("tests/fixtures/catalogue")
    mapping.render_set_prompt = lambda *a, **k: "PROMPT"
    sets = [
        MappingSet(name="Applications", file="applications.json", field="applications"),
        MappingSet(name="Technologies", file="technologies.json", field="technologies"),
    ]
    items, catalogue_hash = loader.load_mapping_items(sets)
    evolutions = _load_evolutions()
    features = [f for evo in evolutions for p in evo.plateaus for f in p.features]
    session_apps = DummySession(
        [
            json.dumps(
                {
                    "features": [
                        {"feature_id": "F1", "applications": [{"item": "app1"}]},
                        {"feature_id": "F2", "applications": [{"item": "app2"}]},
                    ]
                }
            )
        ]
    )
    mapped = asyncio.run(
        mapping.map_set(
            cast(ConversationSession, session_apps),
            "applications",
            items["applications"],
            features,
            _params(catalogue_hash=catalogue_hash),
        )
    )
    session_tech = DummySession(
        [
            json.dumps(
                {
                    "features": [
                        {"feature_id": "F1", "technologies": [{"item": "tech1"}]},
                        {"feature_id": "F2", "technologies": [{"item": "tech2"}]},
                    ]
                }
            )
        ]
    )
    mapped = asyncio.run(
        mapping.map_set(
            cast(ConversationSession, session_tech),
            "technologies",
            items["technologies"],
            mapped,
            _params(catalogue_hash=catalogue_hash),
        )
    )
    by_id = {f.feature_id: f for f in mapped}
    for evo in evolutions:
        for plateau in evo.plateaus:
            mapped_feats = [by_id[f.feature_id] for f in plateau.features]
            plateau.mappings = {
                "applications": mapping.group_features_by_mapping(
                    mapped_feats, "applications", items["applications"]
                ),
                "technologies": mapping.group_features_by_mapping(
                    mapped_feats, "technologies", items["technologies"]
                ),
            }
    out = tmp_path / "out.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for evo in evolutions:
            record = canonicalise_record(evo.model_dump(mode="json"))
            fh.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
    expected = Path("tests/golden/mapping_run.jsonl").read_text(encoding="utf-8")
    assert out.read_text(encoding="utf-8") == expected


def test_default_mode_quarantines_unknown_ids(monkeypatch, tmp_path) -> None:
    """Unknown IDs are dropped and quarantined when strict mode is off."""

    loader.configure_mapping_data_dir("tests/fixtures/catalogue")
    mapping.render_set_prompt = lambda *a, **k: "PROMPT"
    sets = [
        MappingSet(name="Applications", file="applications.json", field="applications")
    ]
    items, catalogue_hash = loader.load_mapping_items(sets)
    evolutions = _load_evolutions()
    features = [f for evo in evolutions for p in evo.plateaus for f in p.features]
    session = DummySession(
        [
            json.dumps(
                {
                    "features": [
                        {"feature_id": "F1", "applications": [{"item": "app1"}]},
                        {"feature_id": "F2", "applications": [{"item": "appX"}]},
                    ]
                }
            )
        ]
    )
    monkeypatch.chdir(tmp_path)
    paths: list[Path] = []
    monkeypatch.setattr(
        "observability.telemetry.record_quarantine", lambda p: paths.append(p.resolve())
    )
    mapped = asyncio.run(
        mapping.map_set(
            cast(ConversationSession, session),
            "applications",
            items["applications"],
            features,
            _params(catalogue_hash=catalogue_hash),
        )
    )
    assert mapped[0].mappings["applications"][0].item == "app1"
    assert mapped[1].mappings["applications"] == []
    qfile = Path("quarantine/unknown/applications/unknown_ids_1.json")
    assert from_json(qfile.read_text()) == ["appX"]
    assert paths == [qfile.resolve()]


def test_strict_mapping_raises_on_unknown_ids(monkeypatch, tmp_path) -> None:
    """Strict mapping mode raises when the agent invents IDs."""

    loader.configure_mapping_data_dir("tests/fixtures/catalogue")
    mapping.render_set_prompt = lambda *a, **k: "PROMPT"
    sets = [
        MappingSet(name="Applications", file="applications.json", field="applications")
    ]
    items, catalogue_hash = loader.load_mapping_items(sets)
    evolutions = _load_evolutions()
    features = [f for evo in evolutions for p in evo.plateaus for f in p.features]
    session = DummySession(
        [
            json.dumps(
                {
                    "features": [
                        {"feature_id": "F1", "applications": [{"item": "app1"}]},
                        {"feature_id": "F2", "applications": [{"item": "appX"}]},
                    ]
                }
            )
        ]
    )
    monkeypatch.chdir(tmp_path)
    with pytest.raises(mapping.MappingError):
        asyncio.run(
            mapping.map_set(
                cast(ConversationSession, session),
                "applications",
                items["applications"],
                features,
                _params(strict=True, catalogue_hash=catalogue_hash),
            )
        )
    qfile = Path("quarantine/unknown/applications/unknown_ids_1.json")
    assert from_json(qfile.read_text()) == ["appX"]
