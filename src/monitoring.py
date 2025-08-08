"""Helpers for enabling Pydantic Logfire telemetry.

The functions in this module are intentionally lightweight wrappers around the
optional `logfire` package. They configure monitoring only when both an API
token and the dependency are present so that the rest of the application can
remain oblivious to telemetry concerns.
"""

from __future__ import annotations

import importlib
import logging
import os

logger = logging.getLogger(__name__)


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
    key = token or os.getenv("LOGFIRE_TOKEN")
    if not key:
        # Without a token no monitoring can be configured.
        logger.debug("LOGFIRE_TOKEN not set; skipping Logfire setup")
        return

    try:
        # ``logfire`` is an optional dependency; import lazily to avoid
        # requiring it for users that do not enable telemetry.
        logfire = importlib.import_module("logfire")
    except ImportError:  # pragma: no cover - depends on optional package
        logger.warning("logfire package not installed; skipping Logfire setup")
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

    handler_cls = getattr(logfire, "LogfireLoggingHandler", None)
    if handler_cls:
        root_logger = logging.getLogger()

        # Replace any existing handlers to prevent duplicate log output.
        root_logger.handlers.clear()
        root_logger.addHandler(handler_cls())

    logger.info("Logfire telemetry enabled")
