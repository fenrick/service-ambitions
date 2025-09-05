# SPDX-License-Identifier: MIT
"""Test configuration for service-ambitions.

Ensures OpenAI calls are stubbed and a deterministic API key is present.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest


@pytest.fixture(autouse=True)
def _mock_openai(monkeypatch):
    """Provide dummy credentials and block outbound OpenAI requests."""

    monkeypatch.setenv("SA_OPENAI_API_KEY", "test-key")
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


@pytest.fixture(autouse=True)
def _clear_prompt_cache():
    """Ensure prompt cache is empty before and after each test."""
    from io_utils import loader

    try:
        loader.clear_prompt_cache()
    except RuntimeError:
        pass
    yield
    try:
        loader.clear_prompt_cache()
    except RuntimeError:
        pass


@pytest.fixture(autouse=True)
def _init_runtime_env():
    """Initialise a default runtime environment for tests."""

    from runtime.environment import RuntimeEnv
    from runtime.settings import load_settings

    RuntimeEnv.reset()
    RuntimeEnv.initialize(load_settings())
    yield
    RuntimeEnv.reset()


class _DummySpan:
    def __enter__(self):
        return SimpleNamespace(set_attribute=lambda *a, **k: None)

    def __exit__(self, *args):
        return None


# Minimal stub to satisfy libraries expecting ``logfire.LogfireSpan``.
class _DummyLogfireSpan:  # pragma: no cover - simple span placeholder
    pass


dummy_logfire = cast(Any, ModuleType("logfire"))
dummy_logfire.metric_counter = lambda name: SimpleNamespace(add=lambda *a, **k: None)
dummy_logfire.span = lambda name, attributes=None: _DummySpan()
dummy_logfire.info = lambda *a, **k: None
dummy_logfire.warning = lambda *a, **k: None
dummy_logfire.error = lambda *a, **k: None
dummy_logfire.debug = lambda *a, **k: None
dummy_logfire.exception = lambda *a, **k: None
dummy_logfire.force_flush = lambda: None
dummy_logfire.LogfireSpan = _DummyLogfireSpan
sys.modules.setdefault("logfire", dummy_logfire)

# Minimal ``pydantic_ai`` stub to avoid heavyweight dependencies.
dummy_pydantic_ai = cast(Any, ModuleType("pydantic_ai"))


class _DummyAgent:
    def __class_getitem__(cls, _):  # pragma: no cover - simple generic stub
        return cls


dummy_pydantic_ai.Agent = _DummyAgent


class _DummyModelRequest:
    def __init__(self, parts=None, **kwargs) -> None:
        self.parts = parts


class _DummyUserPromptPart:
    def __init__(self, content=None, **kwargs) -> None:
        self.content = content


dummy_pydantic_ai.messages = SimpleNamespace(
    ModelRequest=_DummyModelRequest, UserPromptPart=_DummyUserPromptPart
)
sys.modules.setdefault("pydantic_ai", dummy_pydantic_ai)
dummy_pydantic_models = cast(Any, ModuleType("pydantic_ai.models"))
dummy_pydantic_models.Model = SimpleNamespace
sys.modules.setdefault("pydantic_ai.models", dummy_pydantic_models)
dummy_openai_models = cast(Any, ModuleType("pydantic_ai.models.openai"))


class _DummyOpenAIResponsesModel:
    def __init__(self, *args, **kwargs) -> None:
        self._settings = kwargs.get("settings")
        self._provider = kwargs.get("provider")


dummy_openai_models.OpenAIResponsesModel = _DummyOpenAIResponsesModel
dummy_openai_models.OpenAIResponsesModelSettings = SimpleNamespace
sys.modules.setdefault("pydantic_ai.models.openai", dummy_openai_models)
dummy_pydantic_providers = cast(Any, ModuleType("pydantic_ai.providers"))
dummy_pydantic_providers.Provider = SimpleNamespace
sys.modules.setdefault("pydantic_ai.providers", dummy_pydantic_providers)
dummy_openai_providers = cast(Any, ModuleType("pydantic_ai.providers.openai"))


class _DummyOpenAIProvider:
    def __init__(self, api_key=None, **_kwargs) -> None:
        self._client = SimpleNamespace(api_key=api_key)


dummy_openai_providers.OpenAIProvider = _DummyOpenAIProvider
sys.modules.setdefault("pydantic_ai.providers.openai", dummy_openai_providers)
dummy_model_factory = cast(Any, ModuleType("models.factory"))
dummy_model_factory.ModelFactory = SimpleNamespace
sys.modules.setdefault("models.factory", dummy_model_factory)


class _DummyTqdm:  # pragma: no cover - simple progress bar stub
    def __init__(self, *args, **kwargs) -> None:
        pass

    def update(self, *args, **kwargs) -> None:
        return None

    def close(self) -> None:
        return None


dummy_tqdm = cast(Any, ModuleType("tqdm"))
dummy_tqdm.tqdm = _DummyTqdm
sys.modules.setdefault("tqdm", dummy_tqdm)

# Ensure real modules are loaded before test-specific stubs override them.
try:  # noqa: SIM105 - best effort import for optional dependencies
    import generation.plateau_generator as plateau_generator  # noqa: E402,F401
except Exception:  # pragma: no cover - optional import
    plateau_generator = cast(
        Any,
        SimpleNamespace(
            PlateauGenerator=object,
            default_plateau_map=lambda: {},
            default_plateau_names=lambda: [],
        ),
    )
    sys.modules.setdefault("generation.plateau_generator", plateau_generator)
import io_utils.service_loader as service_loader  # noqa: E402,F401
import models  # noqa: E402,F401
import runtime.settings as settings  # noqa: E402,F401
from core import mapping  # noqa: E402,F401
from observability import telemetry  # noqa: E402,F401


class DummyAgent:
    """Agent echoing prompts for deterministic output."""

    def __init__(
        self,
        model: object | None = None,
        instructions: str | None = None,
        output_type: type | None = None,
    ) -> None:
        self.model = model
        self.instructions = instructions
        self.output_type = output_type

    async def run(self, user_prompt: str) -> SimpleNamespace:
        """Return predictable payload for the supplied prompt."""

        return SimpleNamespace(
            output=SimpleNamespace(model_dump=lambda: {"service": user_prompt}),
            usage=lambda: SimpleNamespace(total_tokens=1),
        )


@pytest.fixture()
def dummy_agent() -> type[DummyAgent]:
    """Provide deterministic :class:`DummyAgent` for agent-dependent tests."""

    return DummyAgent
