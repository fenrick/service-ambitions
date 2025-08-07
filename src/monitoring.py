"""Helpers for enabling Pydantic Logfire telemetry."""

from __future__ import annotations

import importlib
import logging
import os

logger = logging.getLogger(__name__)


def init_logfire(service: str | None = None, token: str | None = None) -> None:
    """Configure Logfire if a token is available.

    Args:
        service: Optional service name to associate with traces.
        token: Logfire API token. Falls back to ``LOGFIRE_TOKEN`` env var.

    When the token is provided and the ``logfire`` package is installed this
    function configures the Logfire SDK, instruments Pydantic, Pydantic AI,
    OpenAI and system metrics, and attaches a Logfire logging handler to the
    root logger, replacing existing handlers to avoid duplicate output. If either
    condition is not met the setup is skipped.
    """

    key = token or os.getenv("LOGFIRE_TOKEN")
    if not key:
        logger.debug("LOGFIRE_TOKEN not set; skipping Logfire setup")
        return

    try:
        logfire = importlib.import_module("logfire")
    except ImportError:  # pragma: no cover - depends on optional package
        logger.warning("logfire package not installed; skipping Logfire setup")
        return

    logfire.configure(token=key, service_name=service)
    logfire.instrument_system_metrics(base='full')

    for name in (
        "instrument_pydantic_ai",
        "instrument_pydantic",
        "instrument_openai",
    ):
        instrument = getattr(logfire, name, None)
        if instrument:
            instrument()

    handler_cls = getattr(logfire, "LogfireLoggingHandler", None)
    if handler_cls:
        root_logger = logging.getLogger()
        root_logger.handlers.clear()  # avoid duplicate output from existing handlers
        root_logger.addHandler(handler_cls())

    logger.info(
        "Logfire telemetry enabled%s",
        f" for service {service}" if service else "",
    )
