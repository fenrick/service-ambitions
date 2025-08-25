# SPDX-License-Identifier: MIT
"""Tests for :mod:`shortcode`."""

from shortcode import ShortCodeRegistry


def test_generate_is_deterministic() -> None:
    registry = ShortCodeRegistry()
    first = registry.generate("canon")
    second = registry.generate("canon")
    assert first == second
    assert registry.validate(first)


def test_validate_unknown_code() -> None:
    registry = ShortCodeRegistry()
    assert not registry.validate("ABC123")
