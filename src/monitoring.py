"""Helpers for enabling Pydantic Logfire telemetry.

The functions in this module are intentionally lightweight wrappers around the
optional `logfire` package. They configure monitoring only when both an API
token and the dependency are present so that the rest of the application can
remain oblivious to telemetry concerns.
"""

from __future__ import annotations

import importlib
import json
import logging
import os

logger = logging.getLogger(__name__)

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


def _configure_json_logging(level: int) -> None:
    """Attach a JSON formatter to the root logger."""

    root_logger = logging.getLogger()
    handler = logging.FileHandler(LOG_FILE_NAME)
    handler.setLevel(level)
    handler.setFormatter(JsonFormatter())
    root_logger.handlers.clear()
    root_logger.addHandler(handler)


def init_logfire(token: str | None = None) -> None:
    """Configure Logfire if a token is available.

    Args:
        token: Logfire API token. Falls back to ``LOGFIRE_TOKEN`` environment
            variable when ``None``.

    The function is a no-op when either the token is missing or the optional
    ``logfire`` dependency is not installed. When both are present the Logfire
    SDK is configured, common libraries are instrumented and a logging handler
    is attached to the root logger replacing any existing handlers to prevent
    duplicate output.
    """

    # Use the explicit token if provided, otherwise fall back to the environment.
    root_logger = logging.getLogger()
    # Preserve existing level and formatter when swapping handlers.
    level = root_logger.level
    formatter = root_logger.handlers[0].formatter if root_logger.handlers else None

    key = token or os.getenv("LOGFIRE_TOKEN")
    if not key:
        logger.debug("LOGFIRE_TOKEN not set; skipping Logfire setup")
        _configure_json_logging(level)
        return

    try:
        # ``logfire`` is an optional dependency; import lazily to avoid
        # requiring it for users that do not enable telemetry.
        logfire = importlib.import_module("logfire")
    except ImportError:  # pragma: no cover - depends on optional package
        logger.warning("logfire package not installed; skipping Logfire setup")
        _configure_json_logging(level)
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

    installer = getattr(logfire, "install_auto_tracing", None)
    if installer:
        installer([], min_duration=0)

    handler_cls = getattr(logfire, "LogfireLoggingHandler", None)
    if handler_cls:
        # Replace any existing handlers to prevent duplicate log output.
        root_logger.handlers.clear()
        handler = handler_cls()
        handler.setLevel(level)
        if formatter:
            handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    logger.info("Logfire telemetry enabled")
