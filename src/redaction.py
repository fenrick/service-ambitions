"""Utility helpers for sanitising sensitive data in logs."""

from __future__ import annotations

import re

# Common PII patterns which should be fully masked before logging or
# persistence.  Each pattern is substituted with ``<redacted>`` prior to the
# generic numeric replacement applied at the end of :func:`redact_pii`.
_PII_PATTERNS: list[re.Pattern[str]] = [
    # Email addresses such as ``user@example.com``
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    # UUID strings like ``123e4567-e89b-12d3-a456-426614174000``
    re.compile(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        re.IGNORECASE,
    ),
    # Simple identifiers combining letters and digits e.g. ``AB1234567``
    re.compile(r"[A-Z]{1,3}\d{3,}", re.IGNORECASE),
]


def redact_pii(text: str) -> str:
    """Return ``text`` with obvious personal identifiers removed.

    The function masks common PII patterns such as email addresses and
    alphanumeric identifiers before replacing any remaining digits with
    ``*`` characters.

    Args:
        text: Raw string potentially containing PII.

    Returns:
        Redacted string with sensitive content masked.
    """

    for pattern in _PII_PATTERNS:
        text = pattern.sub("<redacted>", text)
    return re.sub(r"\d", "*", text)


__all__ = ["redact_pii"]
