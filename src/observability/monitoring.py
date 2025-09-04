# SPDX-License-Identifier: MIT
"""Helpers for enabling Pydantic Logfire telemetry."""

from __future__ import annotations

import os
from typing import Literal

import logfire


def init_logfire(
    token: str | None = None,
    min_log_level: Literal[
        "fatal", "error", "warn", "notice", "info", "debug", "trace"
    ] = "warn",
    json_logs: bool = False,
) -> None:
    """Configure Logfire and enable instrumentation.

    Args:
        token: Optional Logfire API token. If omitted, ``LOGFIRE_TOKEN`` from the
            environment is used. Missing tokens keep telemetry local.
        min_log_level: Minimum level for console and telemetry output.
        json_logs: Emit console logs as structured JSON when ``True``.
    """

    key = token or os.getenv("LOGFIRE_TOKEN")
    if json_logs:
        console = logfire.ConsoleOptions(
            min_log_level=min_log_level,
            show_project_link=False,
            verbose=True,
            format="json",
        )  # type: ignore[call-arg]
    else:
        console = logfire.ConsoleOptions(
            min_log_level=min_log_level,
            show_project_link=False,
            verbose=True,
        )
    logfire.configure(
        token=key,
        service_name="service-ambition-generator",
        console=console,
        min_level=min_log_level,
    )

    logfire.instrument_system_metrics(base="full")
    for name in ("instrument_pydantic_ai", "instrument_pydantic", "instrument_openai"):
        instrument = getattr(logfire, name, None)
        if instrument:
            instrument()
