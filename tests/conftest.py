"""Test configuration for service-ambitions.

Ensures OpenAI calls are stubbed and a deterministic API key is present.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable
from unittest.mock import AsyncMock

import pytest

from models import SCHEMA_VERSION, EvolutionMeta


@pytest.fixture(autouse=True)
def _mock_openai(monkeypatch):
    """Provide dummy credentials and block outbound OpenAI requests."""

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    try:
        import openai

        if hasattr(openai, "AsyncClient"):
            monkeypatch.setattr(
                openai.AsyncClient,
                "responses",
                AsyncMock(
                    side_effect=RuntimeError(
                        "OpenAI network access is disabled during tests"
                    )
                ),
                raising=False,
            )
    except Exception:
        pass


@pytest.fixture
def meta_factory() -> Callable[..., EvolutionMeta]:
    """Return a factory that builds ``EvolutionMeta`` instances for tests."""

    def _factory(**overrides: Any) -> EvolutionMeta:
        data: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime(2000, 1, 1, tzinfo=timezone.utc),
            "generator": "tests",
        }
        data.update(overrides)
        return EvolutionMeta(**data)

    return _factory
