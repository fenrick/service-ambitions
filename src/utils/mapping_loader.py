"""Mapping catalogue abstractions with memoisation.

Caching is keyed by ``(file, field)`` tuples so repeat loads avoid disk I/O.
Warm hits complete in microseconds instead of the tens of milliseconds needed
for a cold read.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence, Tuple, TypeAlias, TypeVar, cast

import logfire
from pydantic import TypeAdapter
from pydantic_core import to_json

from models import MappingDatasetFile, MappingItem, MappingSet

# Readability aliases
ItemList: TypeAlias = list[MappingItem]
CatalogueMap: TypeAlias = dict[str, ItemList]


class MappingLoader(ABC):
    """Interface for loading mapping catalogues.

    Implementations should minimise disk access and reuse loaded data where
    possible as catalogues are static. Loading a small catalogue should complete
    within tens of milliseconds.
    """

    @abstractmethod
    def load(self, sets: Sequence[MappingSet]) -> tuple[CatalogueMap, str]:
        """Return mapping data for ``sets`` and a combined hash."""

    @abstractmethod
    def clear_cache(self) -> None:
        """Reset any memoised mapping data."""

    # Optional: auto-discover object-form datasets in a directory
    def discover(
        self,
    ) -> tuple[CatalogueMap, str]:  # pragma: no cover - interface default
        raise NotImplementedError


class FileMappingLoader(MappingLoader):
    """Load mapping items from JSON files on disk with caching."""

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._cache: dict[tuple[str, str], tuple[ItemList, str]] = {}

    def load(self, sets: Sequence[MappingSet]) -> tuple[CatalogueMap, str]:
        """Return mapping data and a combined hash for ``sets``.

        Args:
            sets: Collection of mapping definitions specifying which files and
                fields to load.

        Returns:
            Two-item tuple containing:
                * A mapping of field names to their corresponding list of
                  ``MappingItem`` objects.
                * A SHA256 digest summarising the loaded sets.

        Raises:
            FileNotFoundError: If the data directory or a mapping file is missing.
            pydantic.ValidationError: If a mapping file contains invalid data.
        """
        key: Tuple[Tuple[str, str], ...] = tuple((s.file, s.field) for s in sets)
        with logfire.span(
            "mapping_loader.load",
            attributes={"files": [f for f, _ in key]},
        ):
            if not self._data_dir.is_dir():
                raise FileNotFoundError(
                    f"Mapping data directory not found: {self._data_dir}"
                )
            data: CatalogueMap = {}
            digests: list[str] = []
            for file, field in key:
                cache_key = (file, field)
                if cache_key not in self._cache:
                    path = self._data_dir / file
                    items, meta = _read_mapping_file(path)
                    ordered, digest = _compile_catalogue_for_set(items, meta)
                    # If the dataset self-describes a field, ensure consistency.
                    if meta and "field" in meta and meta["field"] != field:
                        logfire.warning(
                            "Mapping set field mismatch",
                            config_field=field,
                            file_field=meta["field"],
                            file=str(path),
                        )
                    self._cache[cache_key] = (ordered, digest)
                ordered, digest = self._cache[cache_key]
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

    def clear_cache(self) -> None:
        """Empty the internal mapping cache."""
        self._cache.clear()

    def discover(self) -> tuple[CatalogueMap, str]:
        """Return mapping data discovered from object-form datasets in the directory.

        Scans ``self._data_dir`` for ``*.json`` files in the top level. Files that
        provide an embedded ``field`` are included under that field. List-only
        datasets (without embedded metadata) are ignored in discovery mode.
        """
        with logfire.span(
            "mapping_loader.discover", attributes={"dir": str(self._data_dir)}
        ):
            data: CatalogueMap = {}
            digests: list[str] = []
            for path in sorted(self._data_dir.glob("*.json")):
                try:
                    items, meta = _read_mapping_file(path)
                except Exception:  # nosec B112 - skip unreadable/invalid files; logged by reader
                    continue
                field = (meta or {}).get("field") if meta else None
                if not field:
                    continue  # skip list-only datasets in discovery mode
                ordered, digest = _compile_catalogue_for_set(items, meta)
                data[field] = ordered
                digests.append(f"{field}:{digest}")
            combined = "|".join(sorted(digests))
            digest = hashlib.sha256(combined.encode("utf-8")).hexdigest()
            logfire.debug("Discovered mapping sets", count=len(data), digest=digest)
            return data, digest


T = TypeVar("T")


def _read_json_file(path: Path, schema: type[T]) -> T:
    with logfire.span("mapping_loader.read_json", attributes={"path": str(path)}):
        adapter = TypeAdapter(schema)
        with path.open("r", encoding="utf-8") as fh:
            data = adapter.validate_json(fh.read())
        logfire.debug("Read mapping file", path=str(path))
        return data


def _read_mapping_file(path: Path) -> tuple[ItemList, dict[str, str] | None]:
    """Return items and optional metadata from a mapping dataset file.

    Supports two formats:
    - Plain list[MappingItem]
    - MappingDatasetFile with optional ``field`` and ``label`` and an ``items`` list
    """
    with logfire.span("mapping_loader.read_dataset", attributes={"path": str(path)}):
        text = path.read_text(encoding="utf-8")
        # Try self-contained dataset first
        try:
            adapter = TypeAdapter(MappingDatasetFile)
            dataset: MappingDatasetFile = adapter.validate_json(text)
            meta: dict[str, str] | None = None
            if dataset.field or dataset.label:
                meta = {}
                if dataset.field:
                    meta["field"] = dataset.field
                if dataset.label:
                    meta["label"] = dataset.label
            logfire.debug(
                "Read mapping dataset (object)",
                path=str(path),
                field=dataset.field,
                label=dataset.label,
                count=len(dataset.items),
            )
            return dataset.items, meta
        except Exception:  # noqa: BLE001 - fall back to list format
            adapter = TypeAdapter(list[MappingItem])
            item_list = cast(ItemList, adapter.validate_json(text))
            logfire.debug(
                "Read mapping dataset (list)", path=str(path), count=len(item_list)
            )
            return item_list, None


def _sanitize(value: str) -> str:
    return value.replace("\n", " ").replace("\t", " ")


def _compile_catalogue_for_set(
    items: Sequence[MappingItem], meta: dict[str, str] | None = None
) -> tuple[ItemList, str]:
    ordered = sorted(items, key=lambda item: item.id)
    canonical = [
        {
            "id": _sanitize(i.id),
            "name": _sanitize(i.name),
            "description": _sanitize(i.description),
        }
        for i in ordered
    ]
    # Include file-provided metadata in the digest so changes to config
    # invalidate caches derived from this catalogue.
    payload = {"items": canonical, "meta": meta or {}}
    try:
        serialised = to_json(payload, sort_keys=True).decode("utf-8")  # type: ignore[call-arg]
    except TypeError:  # pragma: no cover - legacy pydantic-core
        serialised = to_json(payload).decode("utf-8")
    digest = hashlib.sha256(serialised.encode("utf-8")).hexdigest()
    return ordered, digest
