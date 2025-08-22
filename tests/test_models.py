"""Unit tests for Pydantic models."""

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from models import (
    Contribution,
    MappingResponse,
    MaturityScore,
    PlateauFeature,
    PlateauResult,
    ServiceEvolution,
    ServiceInput,
    ServiceMeta,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_service_evolution_contains_plateaus() -> None:
    """Constructing a ServiceEvolution should retain nested models."""

    service = ServiceInput(
        service_id="svc-1",
        name="svc",
        customer_type="retail",
        description="desc",
        jobs_to_be_done=[{"name": "job"}],
    )
    feature = PlateauFeature(
        feature_id="f1",
        name="Feat",
        description="D",
        score=MaturityScore(level=3, label="Defined", justification="std"),
        customer_type="learners",
    )
    plateau = PlateauResult(
        plateau=1,
        plateau_name="Foundational",
        service_description="desc",
        features=[feature],
    )

    meta = ServiceMeta(
        run_id="run",
        seed=None,
        models={},
        web_search=False,
        mapping_types=[],
        created=datetime.now(timezone.utc),
    )
    evolution = ServiceEvolution(meta=meta, service=service, plateaus=[plateau])

    assert evolution.plateaus[0].features[0].name == "Feat"


def test_plateau_feature_validates_score() -> None:
    """Scores must lie within the inclusive range [0, 1]."""

    with pytest.raises(ValidationError):
        PlateauFeature(
            feature_id="f1",
            name="Feat",
            description="D",
            score=MaturityScore(level=6, label="Invalid", justification="bad"),
            customer_type="learners",
        )


def test_contribution_requires_item() -> None:
    """An ``item`` identifier is mandatory for a contribution."""
    with pytest.raises(ValidationError):
        Contribution()  # type: ignore[call-arg]


def test_contribution_defaults_weight() -> None:
    """Omitting ``contribution`` should default the weight to ``None``."""
    result = Contribution(item="INF-1")
    assert result.contribution is None


def test_contribution_allows_any_weight() -> None:
    """Contribution weights outside the previous range are accepted."""
    result = Contribution(item="INF-1", contribution=1.5)
    assert result.contribution == 1.5


def test_mapping_response_handles_nested_mappings() -> None:
    """Nested ``mappings`` keys should be flattened into the mappings field."""
    payload = {
        "features": [
            {
                "feature_id": "f1",
                "mappings": {
                    "data": [{"item": "INF-1", "contribution": 0.5}],
                },
            }
        ]
    }

    result = MappingResponse.model_validate(payload)

    assert result.features[0].mappings["data"][0].item == "INF-1"


def test_mapping_response_flattens_duplicate_keys() -> None:
    """Duplicate mapping type keys should flatten to the inner list."""
    payload = {
        "features": [
            {
                "feature_id": "f1",
                "applications": {
                    "applications": [{"item": "APP-1", "contribution": 0.5}]
                },
            }
        ]
    }

    result = MappingResponse.model_validate(payload)

    assert result.features[0].mappings["applications"][0].item == "APP-1"


def test_mapping_response_allows_unbounded_items() -> None:
    """Mapping lists should accept arbitrary numbers of entries."""
    payload = {
        "features": [
            {
                "feature_id": "f1",
                "data": [{"item": "INF-1", "contribution": 0.5} for _ in range(6)],
            }
        ]
    }

    result = MappingResponse.model_validate(payload)
    assert len(result.features[0].mappings["data"]) == 6


def test_mapping_response_accepts_none_contribution() -> None:
    """Contributions with ``None`` weights should be preserved."""
    payload = {
        "features": [
            {
                "feature_id": "f1",
                "data": [{"item": "INF-1", "contribution": None}],
            }
        ]
    }

    result = MappingResponse.model_validate(payload)
    assert result.features[0].mappings["data"][0].contribution is None
