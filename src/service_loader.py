"""Iterators for service definition files.

This module provides :class:`ServiceLoader` and :func:`load_services` for
reading newline-delimited JSON service definitions and yielding validated
:class:`~models.ServiceInput` instances.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator, Iterator

import logfire
from pydantic import TypeAdapter

from models import ServiceInput


def _load_service_entries(path: Path | str) -> Generator[ServiceInput, None, None]:
    """Yield services from ``path`` while validating each JSON line."""

    path_obj = Path(path)
    with logfire.span("Calling service_loader.load_services"):
        adapter = TypeAdapter(ServiceInput)
        try:
            with path_obj.open("r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue  # Skip blank lines
                    try:
                        yield adapter.validate_json(line)
                    except Exception as exc:
                        logfire.error(f"Invalid service entry in {path_obj}: {exc}")
                        raise RuntimeError("Invalid service definition") from exc
        except FileNotFoundError:
            logfire.error(f"Services file not found: {path_obj}")
            raise FileNotFoundError(
                "Services file not found. Please create a %s file in the current"
                " directory." % path_obj
            ) from None
        except Exception as exc:
            logfire.error(f"Error reading services file {path_obj}: {exc}")
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
