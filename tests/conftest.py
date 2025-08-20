"""Test configuration for service-ambitions.

Ensures OpenAI calls are stubbed and a deterministic API key is present.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest


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
