"""Mapping catalogue abstractions."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence, Tuple, TypeVar

import logfire
from pydantic import TypeAdapter
from pydantic_core import to_json

from models import MappingItem, MappingSet


class MappingLoader(ABC):
    """Interface for loading mapping catalogues.

    Implementations should minimise disk access and reuse loaded data where
    possible as catalogues are static. Loading a small catalogue should complete
    within tens of milliseconds.
    """

    @abstractmethod
    def load(
        self, sets: Sequence[MappingSet]
    ) -> tuple[dict[str, list[MappingItem]], str]:
        """Return mapping data for ``sets`` and a combined hash."""


class FileMappingLoader(MappingLoader):
    """Load mapping items from JSON files on disk."""

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir

    def load(
        self, sets: Sequence[MappingSet]
    ) -> tuple[dict[str, list[MappingItem]], str]:  # noqa: D401
        key: Tuple[Tuple[str, str], ...] = tuple((s.file, s.field) for s in sets)
        with logfire.span(
            "mapping_loader.load",
            attributes={"files": [f for f, _ in key]},
        ):
            if not self._data_dir.is_dir():
                raise FileNotFoundError(
                    f"Mapping data directory not found: {self._data_dir}"
                )
            data: dict[str, list[MappingItem]] = {}
            digests: list[str] = []
            for file, field in key:
                path = self._data_dir / file
                items = _read_json_file(path, list[MappingItem])
                ordered, digest = _compile_catalogue_for_set(items)
                data[field] = ordered
                digests.append(f"{field}:{digest}")
            combined = "|".join(sorted(digests))
            digest = hashlib.sha256(combined.encode("utf-8")).hexdigest()
            logfire.debug(
                "Loaded mapping sets",
                count=len(data),
                digest=digest,
            )
            return data, digest


T = TypeVar("T")


def _read_json_file(path: Path, schema: type[T]) -> T:
    with logfire.span("mapping_loader.read_json", attributes={"path": str(path)}):
        adapter = TypeAdapter(schema)
        with path.open("r", encoding="utf-8") as fh:
            data = adapter.validate_json(fh.read())
        logfire.debug("Read mapping file", path=str(path))
        return data


def _sanitize(value: str) -> str:
    return value.replace("\n", " ").replace("\t", " ")


def _compile_catalogue_for_set(
    items: Sequence[MappingItem],
) -> tuple[list[MappingItem], str]:
    ordered = sorted(items, key=lambda item: item.id)
    canonical = [
        {
            "id": _sanitize(i.id),
            "name": _sanitize(i.name),
            "description": _sanitize(i.description),
        }
        for i in ordered
    ]
    try:
        serialised = to_json(canonical, sort_keys=True).decode("utf-8")  # type: ignore[call-arg]
    except TypeError:  # pragma: no cover - legacy pydantic-core
        serialised = to_json(canonical).decode("utf-8")
    digest = hashlib.sha256(serialised.encode("utf-8")).hexdigest()
    return ordered, digest
