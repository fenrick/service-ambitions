# SPDX-License-Identifier: MIT
"""Utilities for generating and validating short feature codes.

These codes map human-readable feature descriptions to concise, stable
identifiers. Codes are deterministic: the same canonical string always
produces the same code. A registry tracks all generated codes for
validation and auditing.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from uuid import NAMESPACE_URL, uuid5


@dataclass
class ShortCodeRegistry:
    """Generate and validate deterministic 6 character codes.

    The registry maintains a mapping of generated codes to their canonical
    source strings so callers can verify that returned codes are known and
    trace them back to their original content.
    """

    mapping: dict[str, str] = field(default_factory=dict)

    def generate(self, canonical: str) -> str:
        """Return a 6 character alphanumeric code for ``canonical``.

        Args:
            canonical: String used as the canonical source for the code.

        Returns:
            Deterministic 6 character uppercase code.
        """
        # UUIDv5 provides a reproducible 128-bit value for the input string.
        uuid_val = uuid5(NAMESPACE_URL, canonical)
        # Base32 encode and take the first 6 characters without padding.
        code = base64.b32encode(uuid_val.bytes).decode("ascii").rstrip("=")[:6]
        self.mapping[code] = canonical
        return code

    def validate(self, code: str) -> bool:
        """Return ``True`` when ``code`` is known to the registry."""
        return code in self.mapping
