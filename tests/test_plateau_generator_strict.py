"""Tests for strict mode in PlateauGenerator."""

from __future__ import annotations

from typing import cast

import pytest

from conversation import ConversationSession
from plateau_generator import PlateauGenerator
from test_plateau_generator import DummySession


def test_validate_roles_strict_raises() -> None:
    """Invalid role payloads should raise when strict mode is enabled."""

    session = DummySession([])
    generator = PlateauGenerator(
        cast(ConversationSession, session), required_count=2, strict=True
    )
    role_data = {"learners": []}
    with pytest.raises(ValueError):
        generator._validate_roles(role_data)


def test_enforce_min_features_respects_strict(monkeypatch) -> None:
    """_enforce_min_features emits warnings or raises based on strict mode."""

    session = DummySession([])
    generator = PlateauGenerator(
        cast(ConversationSession, session), required_count=2, strict=False
    )
    warnings: list[str] = []
    monkeypatch.setattr(
        "plateau_generator.logfire.warning", lambda msg, **_: warnings.append(msg)
    )
    generator._enforce_min_features(
        {"learners": [], "academics": [], "professional_staff": []}
    )
    assert warnings  # warning emitted in non-strict mode

    strict_gen = PlateauGenerator(
        cast(ConversationSession, session), required_count=2, strict=True
    )
    with pytest.raises(ValueError):
        strict_gen._enforce_min_features(
            {"learners": [], "academics": [], "professional_staff": []}
        )
