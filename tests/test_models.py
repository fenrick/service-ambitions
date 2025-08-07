"""Unit tests for Pydantic models."""

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models import (  # noqa: E402  pylint: disable=wrong-import-position
    Contribution,
    PlateauFeature,
    PlateauResult,
    ServiceEvolution,
    ServiceInput,
)


def test_service_evolution_contains_plateaus() -> None:
    """Constructing a ServiceEvolution should retain nested models."""

    service = ServiceInput(name="svc", customer_type="retail", description="desc")
    feature = PlateauFeature(
        feature_id="f1",
        name="Feat",
        description="D",
        score=0.75,
        customer_type="learners",
    )
    plateau = PlateauResult(plateau=1, service_description="desc", features=[feature])

    evolution = ServiceEvolution(service=service, plateaus=[plateau])

    assert evolution.plateaus[0].features[0].name == "Feat"


def test_plateau_feature_validates_score() -> None:
    """Scores must lie within the inclusive range [0, 1]."""

    with pytest.raises(ValidationError):
        PlateauFeature(
            feature_id="f1",
            name="Feat",
            description="D",
            score=2.0,
            customer_type="learners",
        )


def test_contribution_requires_fields() -> None:
    """Missing fields should trigger a ``ValidationError``."""
    with pytest.raises(ValidationError):
        Contribution()  # type: ignore[call-arg]
