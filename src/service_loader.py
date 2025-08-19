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


def _load_service_entries(path: Path | str) -> Generator[ServiceInput, None, None]:
    """Yield services from ``path`` while validating each JSON line."""

    path_obj = Path(path)
    with logfire.span("Calling service_loader.load_services"):
        adapter = TypeAdapter(ServiceInput)
        try:
            with path_obj.open("r", encoding="utf-8") as file:
                for line_number, raw_line in enumerate(file, start=1):
                    TOTAL_LINES.add(1)
                    with logfire.span("service_loader.read_line") as span:
                        span.set_attribute("file.path", str(path_obj))
                        span.set_attribute("line.number", line_number)

                        line = raw_line.strip()
                        if not line:
                            span.set_attribute("service.id", None)
                            logfire.debug(
                                "Skipping blank line",
                                file_path=str(path_obj),
                                line_number=line_number,
                            )
                            continue

                        try:
                            service = adapter.validate_json(line)
                            span.set_attribute("service.id", service.service_id)
                            VALID_SERVICES.add(1)
                            yield service
                        except Exception as exc:
                            QUARANTINED_LINES.add(1)
                            quarantine_dir = path_obj.parent / "quarantine"
                            quarantine_dir.mkdir(exist_ok=True)

                            service_id: str | None = None
                            try:
                                # Attempt to extract a service identifier from the JSON
                                service_id = json.loads(line).get("service_id")
                            except Exception:
                                pass  # Parsing failed; fall back to line number

                            span.set_attribute("service.id", service_id)
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
                            continue  # Keep processing subsequent services
        except FileNotFoundError:
            logfire.error("Services file not found", file_path=str(path_obj))
            raise FileNotFoundError(
                "Services file not found. Please create a %s file in the current"
                " directory." % path_obj
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
