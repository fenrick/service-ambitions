# SPDX-License-Identifier: MIT
"""Unit tests for reverse CLI helper functions."""

from __future__ import annotations

import json
from types import SimpleNamespace

from cli.main import _rebuild_mapping_cache, _reconstruct_feature_cache
from core.mapping import build_cache_key, cache_path
from models import (
    FeatureMappingRef,
    MappingFeatureGroup,
    MappingSet,
    MaturityScore,
    PlateauFeature,
    PlateauResult,
)
from runtime.environment import RuntimeEnv


def _settings(tmp_path):
    return SimpleNamespace(
        cache_dir=tmp_path / ".cache",
        context_id="unknown",
        mapping_sets=[
            MappingSet(
                name="Applications", file="applications.json", field="applications"
            )
        ],
        model="gpt-5",
        diagnostics=False,
        prompt_dir=tmp_path,
    )


def test_reconstruct_feature_cache_writes_grouped(tmp_path) -> None:
    RuntimeEnv.reset()
    settings = _settings(tmp_path)
    RuntimeEnv.initialize(settings)
    score = MaturityScore(level=1, label="Initial", justification="j")
    feat = PlateauFeature(
        feature_id="F1",
        name="Feature1",
        description="Desc1",
        score=score,
        customer_type="learners",
        mappings={},
    )
    plateau = PlateauResult(
        plateau=1,
        plateau_name="alpha",
        service_description="desc",
        features=[feat],
        mappings={},
    )
    _reconstruct_feature_cache("svc", plateau)
    cache_file = settings.cache_dir / "unknown" / "svc" / "1" / "features.json"
    assert cache_file.exists()
    data = json.loads(cache_file.read_text(encoding="utf-8"))
    assert data["features"]["learners"][0]["name"] == "Feature1"


def test_rebuild_mapping_cache_writes_entries(tmp_path) -> None:
    RuntimeEnv.reset()
    settings = _settings(tmp_path)
    RuntimeEnv.initialize(settings)
    score = MaturityScore(level=1, label="Initial", justification="j")
    feat = PlateauFeature(
        feature_id="F1",
        name="Feature1",
        description="Desc1",
        score=score,
        customer_type="learners",
        mappings={},
    )
    group = MappingFeatureGroup(
        id="app1",
        name="App1",
        mappings=[FeatureMappingRef(feature_id="F1", description="Desc1")],
    )
    plateau = PlateauResult(
        plateau=1,
        plateau_name="alpha",
        service_description="desc",
        features=[feat],
        mappings={"applications": [group]},
    )
    _rebuild_mapping_cache("svc", plateau, settings, "0" * 64)
    assert plateau.mappings == {}
    key = build_cache_key(
        settings.model, "applications", "0" * 64, [feat], settings.diagnostics
    )
    cache_file = cache_path("svc", 1, "applications", key)
    assert cache_file.exists()
    data = json.loads(cache_file.read_text(encoding="utf-8"))
    assert data["features"][0]["feature_id"] == "F1"
