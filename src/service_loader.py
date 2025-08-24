# SPDX-License-Identifier: MIT
"""Iterators for service definition files.

This module provides :class:`ServiceLoader` and :func:`load_services` for
reading newline-delimited JSON service definitions and yielding validated
:class:`~models.ServiceInput` instances.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Generator, Iterator

import logfire
from pydantic import TypeAdapter

from models import ServiceInput

TOTAL_LINES = logfire.metric_counter("services_total_lines")
VALID_SERVICES = logfire.metric_counter("services_valid")
QUARANTINED_LINES = logfire.metric_counter("services_quarantined")

SERVICES_FILE_NOT_FOUND = "Services file not found"
SERVICE_ID_ATTR = "service.id"


def _extract_service_id(line: str) -> str | None:
    """Return service identifier from ``line`` when available."""

    try:
        return json.loads(line).get("service_id")
    except Exception:
        return None


def _process_line(
    line: str,
    line_number: int,
    path_obj: Path,
    adapter: TypeAdapter[ServiceInput],
) -> ServiceInput | None:
    """Return validated service or ``None`` for invalid entries."""

    if not line:
        logfire.debug(
            "Skipping blank line",
            file_path=str(path_obj),
            line_number=line_number,
        )
        return None
    try:
        service = adapter.validate_json(line)
        VALID_SERVICES.add(1)
        return service
    except Exception as exc:
        QUARANTINED_LINES.add(1)
        quarantine_dir = path_obj.parent / "quarantine"
        quarantine_dir.mkdir(exist_ok=True)
        service_id = _extract_service_id(line)
        filename = f"{service_id or line_number}.json"
        quarantine_path = quarantine_dir / filename
        quarantine_path.write_text(line, encoding="utf-8")
        logfire.error(
            "Invalid service entry",
            file_path=str(path_obj),
            line_number=line_number,
            service_id=service_id,
            quarantine_path=str(quarantine_path),
            error=str(exc),
        )
        return None


def _load_service_entries(path: Path | str) -> Generator[ServiceInput, None, None]:
    """Yield services from ``path`` while validating each JSON line."""

    path_obj = Path(path)
    adapter = TypeAdapter(ServiceInput)
    try:
        with path_obj.open("r", encoding="utf-8") as file:
            for line_number, raw_line in enumerate(file, start=1):
                TOTAL_LINES.add(1)
                service = _process_line(
                    raw_line.strip(), line_number, path_obj, adapter
                )
                if service is not None:
                    yield service
    except FileNotFoundError:
        logfire.error(SERVICES_FILE_NOT_FOUND, file_path=str(path_obj))
        raise FileNotFoundError(
            f"{SERVICES_FILE_NOT_FOUND}. Please create a {path_obj} file in the"
            " current directory."
        ) from None
    except Exception as exc:
        logfire.error(
            "Error reading services file",
            file_path=str(path_obj),
            error=str(exc),
        )
        raise RuntimeError(
            f"An error occurred while reading the services file: {exc}"
        ) from exc


class ServiceLoader:
    """Iterator and context manager for service definitions."""

    def __init__(self, path: Path | str) -> None:
        self._path = path

    def __iter__(self) -> Iterator[ServiceInput]:
        return _load_service_entries(self._path)

    def __enter__(self) -> Iterator[ServiceInput]:
        return self.__iter__()

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def load_services(path: Path | str) -> ServiceLoader:
    """Return an iterable over services defined in ``path``."""

    return ServiceLoader(path)


__all__ = ["load_services", "ServiceLoader"]
