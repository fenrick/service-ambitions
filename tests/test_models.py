"""Unit tests for Pydantic models."""

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from models import (
    Contribution,
    FeatureItem,
    MappingResponse,
    MaturityScore,
    PlateauFeature,
    PlateauResult,
    ServiceEvolution,
    ServiceInput,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_service_evolution_contains_plateaus(meta_factory) -> None:
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

    evolution = ServiceEvolution(
        meta=meta_factory(), service=service, plateaus=[plateau]
    )

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


def test_feature_item_accepts_plateau_numbers() -> None:
    """Feature IDs should include plateau numbers."""

    item = FeatureItem(
        feature_id="FEAT-2-learners-sample",
        name="n",
        description="d",
        score=MaturityScore(level=1, label="Initial", justification="j"),
    )

    assert item.feature_id.startswith("FEAT-2")


def test_feature_item_rejects_invalid_ids() -> None:
    """Nonconforming feature IDs should raise ``ValidationError``."""

    with pytest.raises(ValidationError):
        FeatureItem(
            feature_id="FEAT-X-learners-test",
            name="n",
            description="d",
            score=MaturityScore(level=1, label="Initial", justification="j"),
        )


def test_contribution_requires_fields() -> None:
    """Missing fields should trigger a ``ValidationError``."""
    with pytest.raises(ValidationError):
        Contribution()


def test_contribution_enforces_range() -> None:
    """Contribution weights must lie within [0.1, 1.0]."""
    with pytest.raises(ValidationError):
        Contribution(item="INF-1", contribution=1.5)


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


def test_mapping_response_limits_items() -> None:
    """Mapping lists should contain at most five entries."""
    payload = {
        "features": [
            {
                "feature_id": "f1",
                "data": [
                    {"item": "INF-1", "contribution": 0.5},
                    {"item": "INF-1", "contribution": 0.5},
                    {"item": "INF-1", "contribution": 0.5},
                    {"item": "INF-1", "contribution": 0.5},
                    {"item": "INF-1", "contribution": 0.5},
                    {"item": "INF-1", "contribution": 0.5},
                ],
            }
        ]
    }
    with pytest.raises(ValidationError):
        MappingResponse.model_validate(payload)
