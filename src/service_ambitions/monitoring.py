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
    function configures the Logfire SDK. If either condition is not met the
    setup is skipped.
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
    logger.info(
        "Logfire telemetry enabled%s",
        f" for service {service}" if service else "",
    )
