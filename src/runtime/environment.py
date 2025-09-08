# SPDX-License-Identifier: MIT
"""Runtime environment singleton for shared settings and state."""

from __future__ import annotations

from threading import Lock
from typing import TYPE_CHECKING

import logfire

from utils import (
    FileMappingLoader,
    FilePromptLoader,
    MappingLoader,
    PromptLoader,
)
from llm.queue import LLMQueue

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from models import ServiceMeta
    from runtime.settings import Settings


class RuntimeEnv:
    """Thread-safe singleton storing application settings and shared state."""

    _instance: "RuntimeEnv" | None = None
    _lock = Lock()

    def __init__(self, settings: "Settings") -> None:
        """Initialise the runtime environment."""
        self.settings = settings
        self._state_lock = Lock()
        self._run_meta: "ServiceMeta | None" = None
        self._prompt_loader: PromptLoader = FilePromptLoader(settings.prompt_dir)
        self._mapping_loader: MappingLoader = FileMappingLoader(
            settings.mapping_data_dir
        )
        # Optional global LLM queue (feature-flagged)
        self._llm_queue: LLMQueue | None = None
        if getattr(settings, "llm_queue_enabled", False):
            self._llm_queue = LLMQueue(getattr(settings, "llm_queue_concurrency", 3))
        # Debug logging helps diagnose configuration loading problems.
        logfire.debug("RuntimeEnv created", settings=str(settings))

    @property
    def run_meta(self) -> "ServiceMeta | None":
        """Return metadata describing the current run."""
        with self._state_lock:
            return self._run_meta

    @run_meta.setter
    def run_meta(self, meta: "ServiceMeta | None") -> None:
        """Persist run metadata for later access."""
        with self._state_lock:
            self._run_meta = meta

    @property
    def prompt_loader(self) -> PromptLoader:
        """Return the active prompt loader."""
        return self._prompt_loader

    @prompt_loader.setter
    def prompt_loader(self, loader: PromptLoader) -> None:
        """Persist ``loader`` for later retrieval."""
        with self._state_lock:
            self._prompt_loader = loader

    @property
    def mapping_loader(self) -> MappingLoader:
        """Return the active mapping loader."""
        return self._mapping_loader

    @mapping_loader.setter
    def mapping_loader(self, loader: MappingLoader) -> None:
        """Persist ``loader`` for later retrieval."""
        with self._state_lock:
            self._mapping_loader = loader

    @property
    def llm_queue(self) -> LLMQueue | None:
        """Return the global LLM queue if enabled."""
        return self._llm_queue

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
                    inst.prompt_loader.clear_cache()
                    inst.mapping_loader.clear_cache()
                cls._instance = None


__all__ = ["RuntimeEnv"]
