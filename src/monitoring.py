"""Helpers for enabling Pydantic Logfire telemetry.

The functions in this module configure monitoring only when an API token is
present so that the rest of the application can remain oblivious to telemetry
concerns.
"""

from __future__ import annotations

import json
import logging
import os

# Default log file used across the application
LOG_FILE_NAME = "service.log"


class JsonFormatter(logging.Formatter):
    """Simple structured logging formatter."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        return json.dumps(payload)


def init_logfire(token: str | None = None) -> None:
    """Configure Logfire if a token is available.

    Args:
        token: Logfire API token. Falls back to ``LOGFIRE_TOKEN`` environment
            variable when ``None``.

    The function is a no-op when the token is missing. When a token is provided
    the Logfire SDK is configured, common libraries are instrumented and a
    logging handler is attached to the root logger replacing any existing
    handlers to prevent duplicate output.
    """

    import logfire

    key = token or os.getenv("LOGFIRE_TOKEN")
    if key is None:
        # Persist logs locally when no telemetry token is supplied.
        handler = logging.FileHandler(LOG_FILE_NAME)
        handler.setFormatter(JsonFormatter())
        logging.getLogger().handlers = [handler]
        return

    logfire.configure(token=key, service_name="service-ambition-generator")
    logfire.instrument_system_metrics(base="full")

    # Dynamically enable available instrumentation helpers on the logfire module.
    for name in (
        "instrument_pydantic_ai",
        "instrument_pydantic",
        "instrument_openai",
    ):
        instrument = getattr(logfire, name, None)
        if instrument:
            instrument()

    logfire.info("Logfire telemetry enabled")
