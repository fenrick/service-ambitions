"""Utilities for JSON serialization using Pydantic."""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import TypeAdapter
from pydantic_core import to_json as _to_json

T = TypeVar("T")


def to_json(obj: Any) -> str:
    """Convert a Python object to a JSON string.

    Args:
        obj: The Python object to serialise.

    Returns:
        The JSON representation of ``obj``.
    """

    return _to_json(obj).decode("utf-8")


def from_json(data: str, schema: type[T]) -> T:
    """Parse ``data`` into an instance of ``schema``.

    Args:
        data: The JSON string to deserialise.
        schema: The Pydantic model or other schema type expected.

    Returns:
        An instance of ``schema`` derived from ``data``.
    """

    return TypeAdapter(schema).validate_json(data)


__all__ = ["to_json", "from_json"]
