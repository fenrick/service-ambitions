# SPDX-License-Identifier: MIT
"""Helpers for enabling Pydantic Logfire telemetry."""

from __future__ import annotations

import os
from typing import Literal

import logfire


def _mask_token(value: str | None) -> str | None:
    """Return a masked representation of ``value`` for safe logging."""

    if not value:
        return None
    return f"{value[:4]}..."


def init_logfire(
    token: str | None = None,
    min_log_level: Literal[
        "fatal", "error", "warn", "notice", "info", "debug", "trace"
    ] = "warn",
) -> None:
    """Configure Logfire and enable instrumentation.

    Args:
        token: Optional Logfire API token. If omitted, ``SA_LOGFIRE_TOKEN`` from the
            environment is used. Missing tokens keep telemetry local.
        min_log_level: Minimum level for console and telemetry output.
    """

    key = token or os.getenv("SA_LOGFIRE_TOKEN")
    masked = _mask_token(key)
    if key and hasattr(logfire, "add_masking_rule"):
        logfire.add_masking_rule(key)
    logfire.debug("Configuring logfire", token=masked)
    logfire.configure(
        token=key,
        service_name="service-ambition-generator",
        console=logfire.ConsoleOptions(
            min_log_level=min_log_level,
            show_project_link=False,
            verbose=True,
        ),
        min_level=min_log_level,
    )

    logfire.instrument_system_metrics(base="full")
    for name in ("instrument_pydantic_ai", "instrument_pydantic", "instrument_openai"):
        instrument = getattr(logfire, name, None)
        if instrument:
            instrument()
