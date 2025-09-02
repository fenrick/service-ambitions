# SPDX-License-Identifier: MIT
"""Runtime environment singleton for shared settings and state."""

from __future__ import annotations

from threading import Lock
from typing import TYPE_CHECKING, Any

import logfire

from utils import (
    FileMappingLoader,
    FilePromptLoader,
    MappingLoader,
    PromptLoader,
)

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from models import ServiceMeta
    from settings import Settings


class RuntimeEnv:
    """Thread-safe singleton storing application settings and shared state."""

    _instance: "RuntimeEnv" | None = None
    _lock = Lock()

    def __init__(self, settings: "Settings") -> None:
        """Initialise the runtime environment."""

        self.settings = settings
        self.state: dict[str, Any] = {}
        # Debug logging helps diagnose configuration loading problems.
        logfire.debug("RuntimeEnv created", settings=str(settings))

    @property
    def run_meta(self) -> "ServiceMeta | None":
        """Return metadata describing the current run."""

        return self.state.get("run_meta")

    @run_meta.setter
    def run_meta(self, meta: "ServiceMeta | None") -> None:
        """Persist run metadata for later access."""

        if meta is None:
            self.state.pop("run_meta", None)
        else:
            self.state["run_meta"] = meta

    @property
    def prompt_loader(self) -> PromptLoader:
        """Return the active prompt loader, creating a default if needed."""

        loader = self.state.get("prompt_loader")
        if loader is None:
            loader = FilePromptLoader(self.settings.prompt_dir)
            self.state["prompt_loader"] = loader
        return loader

    @prompt_loader.setter
    def prompt_loader(self, loader: PromptLoader) -> None:
        """Persist ``loader`` for later retrieval."""

        self.state["prompt_loader"] = loader

    @property
    def mapping_loader(self) -> MappingLoader:
        """Return the active mapping loader, creating a default if needed."""

        loader = self.state.get("mapping_loader")
        if loader is None:
            loader = FileMappingLoader(self.settings.mapping_data_dir)
            self.state["mapping_loader"] = loader
        return loader

    @mapping_loader.setter
    def mapping_loader(self, loader: MappingLoader) -> None:
        """Persist ``loader`` for later retrieval."""

        self.state["mapping_loader"] = loader

    @classmethod
    def initialize(cls, settings: "Settings") -> "RuntimeEnv":
        """Initialise and return the runtime environment.

        Args:
            settings: Validated application settings.

        Returns:
            The active :class:`RuntimeEnv` instance.
        """
        with logfire.span("runtime_env.initialize"):
            with cls._lock:
                logfire.info(
                    "Initialising runtime environment",
                    context_id=getattr(settings, "context_id", None),
                )
                cls._instance = cls(settings)
                return cls._instance

    @classmethod
    def instance(cls) -> "RuntimeEnv":
        """Return the current runtime environment.

        Raises:
            RuntimeError: If :meth:`initialize` was not called.
        """
        inst = cls._instance
        if inst is None:
            logfire.error("RuntimeEnv accessed before initialisation")
            raise RuntimeError("RuntimeEnv has not been initialised")
        return inst

    @classmethod
    def reset(cls) -> None:
        """Clear the active runtime environment and cached state.

        Useful for tests needing a fresh configuration or for scenarios where
        the application must reload settings at runtime. Loader caches and run
        metadata are cleared to avoid stale data in subsequent initialisations.
        """
        with logfire.span("runtime_env.reset"):
            with cls._lock:
                logfire.info("Resetting runtime environment")
                inst = cls._instance
                if inst is not None:
                    inst.run_meta = None  # remove any persisted run metadata
                    prompt = inst.state.get("prompt_loader")
                    if prompt is not None:
                        # Discard memoised prompts to force reloads.
                        prompt.clear_cache()
                    mapping = inst.state.get("mapping_loader")
                    if mapping is not None:
                        # Discard cached mapping data to force reloads.
                        mapping.clear_cache()
                cls._instance = None


__all__ = ["RuntimeEnv"]
