"""Unit tests for Pydantic models."""

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models import (  # noqa: E402  pylint: disable=wrong-import-position
    PlateauFeature,
    PlateauResult,
    ServiceEvolution,
    ServiceInput,
)


def test_service_evolution_contains_results() -> None:
    """Constructing a ServiceEvolution should retain nested models."""

    service = ServiceInput(name="svc", description="desc")
    feature = PlateauFeature(feature_id="f1", name="Feat", description="D")
    result = PlateauResult(feature=feature, score=0.75)

    evolution = ServiceEvolution(service=service, results=[result])

    assert evolution.results[0].feature.name == "Feat"


def test_plateau_result_validates_score() -> None:
    """Scores must lie within the inclusive range [0, 1]."""

    feature = PlateauFeature(feature_id="f1", name="Feat", description="D")

    with pytest.raises(ValidationError):
        PlateauResult(feature=feature, score=2.0)
