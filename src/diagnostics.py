"""Helpers for validating generated output files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Type, cast

import logfire as _logfire
from pydantic import BaseModel

logfire = cast(Any, _logfire)


@logfire.instrument()
def validate_jsonl(path: Path, model: Type[BaseModel]) -> int:
    """Validate JSON records in ``path`` against ``model``.

    Args:
        path: Location of the JSONL file to inspect.
        model: Pydantic model used for validation of each line.

    Returns:
        The number of valid lines processed.

    Raises:
        ValueError: If any line fails validation.
    """

    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                model.model_validate_json(line)
            except Exception as exc:  # noqa: PERF203
                raise ValueError(f"Line {idx} invalid: {exc}") from exc
            count += 1
    return count


__all__ = ["validate_jsonl"]
