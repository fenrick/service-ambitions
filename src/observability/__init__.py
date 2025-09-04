"""Telemetry and monitoring helpers for the service generator.

Exports:
    init_logfire: Configure Pydantic Logfire instrumentation.
    record_mapping_set: Record metrics for a mapping set.
    record_quarantine: Track creation of quarantine files.
    print_summary: Output a summary of collected metrics.
    has_quarantines: Indicate whether quarantined files exist.
    reset: Clear stored metrics and quarantine paths.
"""

from .monitoring import init_logfire
from .telemetry import (
    has_quarantines,
    print_summary,
    record_mapping_set,
    record_quarantine,
    reset,
)

__all__ = [
    "init_logfire",
    "record_mapping_set",
    "record_quarantine",
    "print_summary",
    "has_quarantines",
    "reset",
]
