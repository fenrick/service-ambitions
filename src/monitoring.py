# SPDX-License-Identifier: MIT
"""Helpers for enabling Pydantic Logfire telemetry.

Logfire operates locally without an API token; the token is only required for
publishing telemetry to the cloud. These helpers configure the SDK and, when
requested, instrument the application regardless of token availability so the
rest of the application can remain oblivious to monitoring details.
"""

from __future__ import annotations

import os

# Default log file used across the application
LOG_FILE_NAME = "service.log"


def init_logfire(token: str | None = None, diagnostics: bool = False) -> None:
    """Configure the Logfire SDK and optionally enable diagnostics.

    Args:
        token: Optional Logfire API token. When omitted the ``LOGFIRE_TOKEN``
            environment variable is used. A missing token means logs remain
            local but the SDK still operates.
        diagnostics: When ``True``, enable system metrics and available
            instrumentation helpers to collect diagnostic telemetry.
    """

    import logfire

    key = token or os.getenv("LOGFIRE_TOKEN")
    logfire.configure(token=key, service_name="service-ambition-generator")

    if diagnostics:  # Enable instrumentation only when diagnostics are requested
        logfire.instrument_system_metrics(base="full")

        # Dynamically enable available instrumentation helpers on the logfire module.
        for name in (
            "instrument_pydantic_ai",
            "instrument_pydantic",
            "instrument_openai",
        ):
            instrument = getattr(logfire, name, None)
            if instrument:  # Skip helpers missing from the current Logfire build
                instrument()

        logfire.info("Logfire telemetry enabled")
