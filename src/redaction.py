"""Utility helpers for sanitising sensitive data in logs."""

from __future__ import annotations

import re


def redact_pii(text: str) -> str:
    """Return ``text`` with obvious personal identifiers removed.

    This implementation naively replaces all digits with ``*`` characters.

    Args:
        text: Raw string potentially containing PII.

    Returns:
        Redacted string with numeric characters masked.
    """

    return re.sub(r"\d", "*", text)


__all__ = ["redact_pii"]
