"""Factory for constructing stage-specific OpenAI models."""

from __future__ import annotations

from typing import Dict

from pydantic_ai.models import Model

from generator import build_model
from models import ReasoningConfig, StageModels


class ModelFactory:
    """Build and cache models for different generation stages."""

    def __init__(
        self,
        default_model: str,
        api_key: str,
        *,
        stage_models: StageModels | None = None,
        reasoning: ReasoningConfig | None = None,
        seed: int | None = None,
        web_search: bool = False,
    ) -> None:
        self._default = default_model
        self._api_key = api_key
        self._stage_models = stage_models or StageModels()
        self._reasoning = reasoning
        self._seed = seed
        self._web_search = web_search
        self._cache: Dict[str, Model] = {}

    def model_name(self, stage: str, override: str | None = None) -> str:
        """Return the resolved model name for ``stage``."""

        if override:
            return override
        return getattr(self._stage_models, stage, None) or self._default

    def get(self, stage: str, override: str | None = None) -> Model:
        """Return a model instance for ``stage``."""

        name = self.model_name(stage, override)
        model = self._cache.get(name)
        if model is None:
            use_search = self._web_search if stage == "search" else False
            model = build_model(
                name,
                self._api_key,
                seed=self._seed,
                reasoning=self._reasoning,
                web_search=use_search,
            )
            self._cache[name] = model
        return model


__all__ = ["ModelFactory"]
