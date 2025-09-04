"""Error handling abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod

import logfire


class ErrorHandler(ABC):
    """Interface for reporting errors.

    Implementations should avoid raising further exceptions and should emit
    concise diagnostics suitable for production logs.
    """

    @abstractmethod
    def handle(self, message: str, exc: Exception | None = None) -> None:
        """Record ``message`` with optional ``exc`` context."""


class LoggingErrorHandler(ErrorHandler):
    """Error handler that logs via ``logfire``."""

    def handle(self, message: str, exc: Exception | None = None) -> None:
        """Log an error message with optional exception context.

        Args:
            message: Description of the error to record.
            exc: Exception instance providing additional context.

        Returns:
            None.
        """
        if exc:
            logfire.error(f"{message}: {exc}")
        else:
            logfire.error(message)
