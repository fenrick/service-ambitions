"""Tests for pydantic_code utilities."""

from __future__ import annotations

from pydantic import BaseModel

from pydantic_code import from_json, to_json


class SampleModel(BaseModel):
    """Simple model for round-trip tests."""

    value: int
    name: str


def test_round_trip_json() -> None:
    """Objects should round-trip to JSON and back."""

    model = SampleModel(value=1, name="alpha")
    data = to_json(model)
    result = from_json(data, SampleModel)
    assert result == model
