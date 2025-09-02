# SPDX-License-Identifier: MIT
"""Tests for canonicalise_record helper."""

import json
from datetime import datetime, timezone

from core.canonical import canonicalise_record
from models import ServiceMeta


def test_canonicalise_record_sorts_features_and_mappings() -> None:
    record = {
        "service": {
            "features": [
                {
                    "feature_id": "b",
                    "mappings": {"apps": [{"item": "2"}, {"item": "1"}]},
                },
                {
                    "feature_id": "a",
                    "mappings": {"apps": [{"item": "3"}, {"item": "2"}]},
                },
            ]
        },
        "plateaus": [
            {
                "features": [
                    {
                        "feature_id": "2",
                        "mappings": {"tech": [{"item": "b"}, {"item": "a"}]},
                    },
                    {
                        "feature_id": "1",
                        "mappings": {"tech": [{"item": "d"}, {"item": "c"}]},
                    },
                ]
            }
        ],
        "meta": {},
    }
    result = canonicalise_record(record)
    assert [f["feature_id"] for f in result["service"]["features"]] == ["a", "b"]
    assert [
        c["item"] for c in result["service"]["features"][0]["mappings"]["apps"]
    ] == [
        "2",
        "3",
    ]
    plateau_feats = result["plateaus"][0]["features"]
    assert [f["feature_id"] for f in plateau_feats] == ["1", "2"]
    assert [c["item"] for c in plateau_feats[0]["mappings"]["tech"]] == ["c", "d"]
    assert result["meta"]["seed"] == 0
    assert result["meta"]["context_window"] == 0
    assert result["meta"]["diagnostics"] is False
    assert result["meta"]["catalogue_hash"] == ""


def test_canonicalise_record_handles_datetime() -> None:
    meta = ServiceMeta(run_id="r1", created=datetime(2024, 1, 1, tzinfo=timezone.utc))
    record = {"meta": meta.model_dump(mode="json")}
    canonical = canonicalise_record(record)
    assert isinstance(canonical["meta"]["created"], str)
    json.dumps(canonical, separators=(",", ":"), sort_keys=True)
