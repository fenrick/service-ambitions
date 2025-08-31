# SPDX-License-Identifier: MIT
"""Test configuration for service-ambitions.

Ensures OpenAI calls are stubbed and a deterministic API key is present.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
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


class _DummySpan:
    def __enter__(self):
        return SimpleNamespace(set_attribute=lambda *a, **k: None)

    def __exit__(self, *args):
        return None


dummy_logfire = SimpleNamespace(
    metric_counter=lambda name: SimpleNamespace(add=lambda *a, **k: None),
    span=lambda name, attributes=None: _DummySpan(),
    info=lambda *a, **k: None,
    warning=lambda *a, **k: None,
    error=lambda *a, **k: None,
    debug=lambda *a, **k: None,
    exception=lambda *a, **k: None,
    force_flush=lambda: None,
)
sys.modules.setdefault("logfire", dummy_logfire)  # type: ignore[arg-type]

dummy_pydantic = SimpleNamespace(BaseModel=object)
sys.modules.setdefault("pydantic", dummy_pydantic)  # type: ignore[arg-type]
sys.modules.setdefault(
    "pydantic_ai",
    SimpleNamespace(Agent=object, messages=SimpleNamespace(ModelMessage=object)),
)  # type: ignore[arg-type]
sys.modules.setdefault("pydantic_ai.models", SimpleNamespace(Model=object))  # type: ignore[arg-type]
sys.modules.setdefault(
    "pydantic_ai.models.openai",
    SimpleNamespace(OpenAIResponsesModel=object, OpenAIResponsesModelSettings=object),
)  # type: ignore[arg-type]


class _DummyTqdm:  # pragma: no cover - simple progress bar stub
    def __init__(self, *args, **kwargs) -> None:
        pass

    def update(self, *args, **kwargs) -> None:
        return None

    def close(self) -> None:
        return None


sys.modules.setdefault("tqdm", SimpleNamespace(tqdm=_DummyTqdm))  # type: ignore[arg-type]


class DummyAgent:
    """Agent echoing prompts for deterministic output."""

    def __init__(
        self, model: object | None = None, instructions: str | None = None
    ) -> None:
        self.model = model
        self.instructions = instructions

    async def run(self, user_prompt: str, output_type: type) -> SimpleNamespace:
        """Return predictable payload for the supplied prompt."""

        return SimpleNamespace(
            output=SimpleNamespace(model_dump=lambda: {"service": user_prompt}),
            usage=lambda: SimpleNamespace(total_tokens=1),
        )


@pytest.fixture()
def dummy_agent() -> type[DummyAgent]:
    """Provide deterministic :class:`DummyAgent` for agent-dependent tests."""

    return DummyAgent
