# SPDX-License-Identifier: MIT
"""Runtime environment singleton for shared settings and state."""

from __future__ import annotations

from threading import Lock
from typing import TYPE_CHECKING, Any

import logfire

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
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


__all__ = ["RuntimeEnv"]
