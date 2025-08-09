from __future__ import annotations

from types import SimpleNamespace

import pytest


class DummyAgent:
    """Simple stand-in for :class:`pydantic_ai.Agent`."""

    def __init__(
        self,
        model=None,
        instructions: str | None = None,
    ) -> None:  # pragma: no cover
        self.model = model
        self.instructions = instructions

    async def run(self, prompt: str, output_type=None):  # pragma: no cover - stub
        output = output_type.model_construct() if output_type else None
        return SimpleNamespace(output=output, new_messages=lambda: [])


class DummyModel:
    """Placeholder model used to avoid network calls."""

    def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - trivial
        return


@pytest.fixture(autouse=True)
def _mock_openai_pydantic_ai(monkeypatch):
    """Prevent external API calls during tests."""

    # Replace OpenAI client with no-op implementation.
    def _openai_client(*_, **__):  # pragma: no cover - simple stub
        return SimpleNamespace()

    monkeypatch.setattr("openai.OpenAI", _openai_client, raising=False)

    # Stub out Pydantic-AI components that could reach the network.
    monkeypatch.setattr("pydantic_ai.Agent", DummyAgent)
    monkeypatch.setattr("pydantic_ai.models.openai.OpenAIResponsesModel", DummyModel)

    # Ensure modules importing Agent directly use the stub.
    import generator

    monkeypatch.setattr(generator, "Agent", DummyAgent)
