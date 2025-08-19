"""Utilities for enriching plateau features with mapping data.

Mapping information such as related applications or technologies is gathered by
querying the language model with a consolidated prompt. The helper functions
here prepare prompt content, validate responses and merge the returned
contributions back into :class:`PlateauFeature` objects.
"""

from __future__ import annotations

import asyncio
import json
import os
from functools import lru_cache
from typing import TYPE_CHECKING, Iterable, Mapping, Sequence

import logfire
import numpy as np
from openai import AsyncOpenAI
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)

from loader import load_mapping_items, load_mapping_type_config, load_prompt_text
from models import (
    Contribution,
    MappingItem,
    MappingResponse,
    MappingTypeConfig,
    PlateauFeature,
)

if TYPE_CHECKING:
    from conversation import ConversationSession


EMBED_MODEL = "text-embedding-3-small"
"""Embedding model used for catalogue pre-filtering."""

_EMBED_CLIENT: AsyncOpenAI | None = None
_EMBED_CACHE: dict[str, tuple[np.ndarray, list[MappingItem]]] = {}


def _chunked(
    seq: Sequence[PlateauFeature], size: int
) -> Iterable[Sequence[PlateauFeature]]:
    """Yield successive ``size``-sized chunks from ``seq``."""

    for i in range(0, len(seq), size):
        yield seq[i : i + size]


@lru_cache(maxsize=None)
def _catalogue_vectors(
    dataset: str,
) -> tuple[TfidfVectorizer, csr_matrix, list[MappingItem]]:
    """Return cached TF-IDF vectoriser and matrix for ``dataset`` catalogue."""

    items = load_mapping_items((dataset,))[dataset]
    item_texts = [f"{it.name} {it.description}" for it in items]
    vectorizer = TfidfVectorizer().fit(item_texts)
    item_vecs = vectorizer.transform(item_texts)
    return vectorizer, item_vecs, items


async def _get_embed_client() -> AsyncOpenAI:
    """Return a cached OpenAI client for embedding requests."""

    global _EMBED_CLIENT
    if _EMBED_CLIENT is None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY missing")
        _EMBED_CLIENT = AsyncOpenAI()
    return _EMBED_CLIENT


async def _catalogue_embeddings(
    dataset: str,
) -> tuple[np.ndarray, list[MappingItem]]:
    """Return embedding vectors and items for the ``dataset`` catalogue."""

    cache = _EMBED_CACHE.get(dataset)
    if cache is not None:
        return cache
    client = await _get_embed_client()
    items = load_mapping_items((dataset,))[dataset]
    texts = [f"{it.name} {it.description}" for it in items]
    resp = await client.embeddings.create(model=EMBED_MODEL, input=texts)
    vectors = np.array([d.embedding for d in resp.data])
    cache = (vectors, items)
    _EMBED_CACHE[dataset] = cache
    return cache


async def init_embeddings() -> None:
    """Pre-populate embedding vectors for all mapping datasets.

    Any errors during pre-computation are logged and ignored so that embedding
    vectors will be generated lazily when first requested.
    """

    try:
        cfg = load_mapping_type_config()
    except Exception as exc:
        # Loading configuration is a best-effort step; missing config merely
        # delays embedding generation until first use.
        logfire.warning(f"Failed to load mapping config: {exc}")
        return

    datasets = {c.dataset for c in cfg.values()}
    for name in datasets:
        try:
            await _catalogue_embeddings(name)
        except Exception as exc:
            # Log and continue so the cache can be populated lazily later.
            logfire.warning(f"Failed to warm embeddings for {name}: {exc}")


async def _embedding_top_k_items(
    features: Sequence[PlateauFeature], dataset: str, k: int = 40
) -> list[MappingItem]:
    """Return ``k`` catalogue items closest to ``features`` using embeddings.

    Results are ordered lexicographically by item identifier so selections
    remain stable across runs even when similarity scores tie.
    """

    item_vecs, items = await _catalogue_embeddings(dataset)
    client = await _get_embed_client()
    feature_texts = [f"{f.name} {f.description}" for f in features]
    resp = await client.embeddings.create(model=EMBED_MODEL, input=feature_texts)
    feat_vecs = np.array([d.embedding for d in resp.data])
    sims = feat_vecs @ item_vecs.T
    top_indices: set[int] = set()
    for row in sims:
        ranking = sorted(enumerate(row), key=lambda x: (-x[1], items[x[0]].id))
        idxs = [idx for idx, _ in ranking[:k]]
        top_indices.update(idxs)
    return sorted((items[i] for i in top_indices), key=lambda item: item.id)


async def _preselect_items(
    features: Sequence[PlateauFeature],
    cfg: MappingTypeConfig,
    *,
    k: int = 40,
) -> dict[str, list[MappingItem]] | None:
    """Return mapping catalogue overrides selected via embeddings."""

    try:
        slice_items = await _embedding_top_k_items(features, cfg.dataset, k)
    except Exception as exc:
        logfire.warning(f"Embedding pre-filter failed: {exc}")
        return None
    return {cfg.dataset: slice_items}


class MappingError(RuntimeError):
    """Raised when a mapping response is missing required data."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


def _render_items(items: list[MappingItem]) -> str:
    """Return a bullet list representation of mapping reference items.

    Each entry is formatted as ``<id>: <name> - <description>`` so the string
    can be embedded directly within agent prompts.
    """

    # Present each mapping reference item on a separate line so it can be
    # directly inserted into the agent prompt as a bullet list.
    return "\n".join(
        f"- {entry.id}: {entry.name} - {entry.description}" for entry in items
    )


def _render_features(features: Sequence[PlateauFeature]) -> str:
    """Return a bullet list of feature details for prompt construction.

    Features are presented using their ID, name and description to provide the
    agent with enough context for mapping decisions while keeping prompts
    compact.
    """

    # The agent prompt expects a concise summary of each feature, therefore we
    # format the feature ID, name and description on a single line.
    return "\n".join(
        f"- {feat.feature_id}: {feat.name} - {feat.description}" for feat in features
    )


def _top_k_items(
    features: Sequence[PlateauFeature],
    dataset: str,
    k: int = 5,
) -> list[MappingItem]:
    """Return ``k`` catalogue items most similar to ``features`` using TF-IDF.

    Ties are broken by lexicographic item identifier to keep results stable
    across runs. Returned items are also ordered by their identifier so the
    output sequence is deterministic.
    """

    vectorizer, item_vecs, items = _catalogue_vectors(dataset)
    feature_texts = [f"{f.name} {f.description}" for f in features]
    feat_vecs = vectorizer.transform(feature_texts)
    scores = feat_vecs @ item_vecs.T
    top_indices: set[int] = set()
    for row in scores.toarray():
        ranking = sorted(enumerate(row), key=lambda x: (-x[1], items[x[0]].id))
        idxs = [idx for idx, _ in ranking[:k]]
        top_indices.update(idxs)
    return sorted((items[i] for i in top_indices), key=lambda item: item.id)


async def map_feature_async(
    session: ConversationSession,
    feature: PlateauFeature,
    mapping_types: Mapping[str, MappingTypeConfig] | None = None,
) -> PlateauFeature:
    """Asynchronously return ``feature`` augmented with mapping data."""

    return (await map_features_async(session, [feature], mapping_types))[0]


def map_feature(
    session: ConversationSession,
    feature: PlateauFeature,
    mapping_types: Mapping[str, MappingTypeConfig] | None = None,
) -> PlateauFeature:
    """Return ``feature`` augmented with mapping information."""

    return asyncio.run(map_feature_async(session, feature, mapping_types))


def _build_mapping_prompt(
    features: Sequence[PlateauFeature],
    mapping_types: Mapping[str, MappingTypeConfig],
    *,
    item_overrides: Mapping[str, list[MappingItem]] | None = None,
    extra_instructions: str | None = None,
) -> str:
    """Return a prompt requesting mappings for ``features``."""

    template = load_prompt_text("mapping_prompt")
    schema = json.dumps(MappingResponse.model_json_schema(), indent=2)
    items = load_mapping_items(tuple(cfg.dataset for cfg in mapping_types.values()))
    sections = []
    for cfg in mapping_types.values():
        if item_overrides and cfg.dataset in item_overrides:
            dataset_items = item_overrides[cfg.dataset]
        else:
            dataset_items = items[cfg.dataset]
        sections.append(f"## Available {cfg.label}\n\n{_render_items(dataset_items)}\n")
    mapping_sections = "\n".join(sections)
    mapping_labels = ", ".join(cfg.label for cfg in mapping_types.values())
    mapping_fields = ", ".join(mapping_types.keys())
    prompt = template.format(
        mapping_labels=mapping_labels,
        mapping_sections=mapping_sections,
        mapping_fields=mapping_fields,
        features=_render_features(features),
        schema=str(schema),
    )
    if extra_instructions:
        prompt = f"{prompt}\n\n{extra_instructions}"
    return prompt


def _merge_mapping_results(
    features: Sequence[PlateauFeature],
    payload: MappingResponse,
    mapping_types: Mapping[str, MappingTypeConfig],
) -> list[PlateauFeature]:
    """Return ``features`` merged with mapping ``payload``.

    Any mappings referencing unknown item identifiers are dropped rather than
    causing an error. This allows the calling code to rerun generation without
    manual intervention when the agent invents IDs.
    """
    # Build a lookup of valid item identifiers for each mapping type to
    # prevent the agent from inventing IDs that do not exist in reference
    # datasets.
    catalogues = load_mapping_items(
        tuple(cfg.dataset for cfg in mapping_types.values())
    )
    valid_ids: dict[str, set[str]] = {
        key: {item.id for item in catalogues[cfg.dataset]}
        for key, cfg in mapping_types.items()
    }

    mapped_lookup = {item.feature_id: item.mappings for item in payload.features}
    results: list[PlateauFeature] = []
    for feature in features:
        mapped = mapped_lookup.get(feature.feature_id)
        if mapped is None:
            # Each feature must appear in the response; fail fast when absent.
            raise MappingError(f"Missing mappings for feature {feature.feature_id}")
        update_data: dict[str, list[Contribution]] = {}
        for key in mapping_types.keys():
            values = mapped.get(key, [])
            if not values:
                # Log missing or empty mappings rather than failing outright so
                # feature generation can continue when the agent omits a
                # category.  An empty list is stored for the mapping type.
                logfire.warning(
                    f"Missing mappings: feature={feature.feature_id} key={key}"
                )
            else:
                valid_values: list[Contribution] = []
                for item in values:
                    if item.item not in valid_ids[key]:
                        # Drop invalid mapping references so feature generation
                        # can proceed without manual intervention. These
                        # entries may be regenerated in a future run.
                        logfire.warning(
                            f"Dropping unknown {key} ID {item.item} for feature"
                            f" {feature.feature_id}"
                        )
                        continue
                    valid_values.append(item)
                values = valid_values
            update_data[key] = values
        merged = feature.model_copy(update={"mappings": feature.mappings | update_data})
        results.append(merged)
    return results


async def _request_mapping(
    session: ConversationSession,
    batch: Sequence[PlateauFeature],
    batch_index: int,
    key: str,
    cfg: MappingTypeConfig,
) -> tuple[int, str, MappingTypeConfig, ConversationSession, MappingResponse]:
    """Return mapping response for ``key`` type for ``batch`` features."""

    sub_session = session.derive()
    overrides = await _preselect_items(batch, cfg)
    prompt = _build_mapping_prompt(batch, {key: cfg}, item_overrides=overrides)
    logfire.debug(f"Requesting {key} mappings for {len(batch)} features")
    payload = await sub_session.ask_async(prompt, output_type=MappingResponse)
    return batch_index, key, cfg, sub_session, payload


async def map_features_async(
    session: ConversationSession,
    features: Sequence[PlateauFeature],
    mapping_types: Mapping[str, MappingTypeConfig] | None = None,
    *,
    batch_size: int = 30,
    parallel_types: bool = True,
) -> list[PlateauFeature]:
    """Asynchronously return ``features`` with mapping information.

    Args:
        session: Active conversation session used for mapping requests.
        features: Plateau features requiring mapping enrichment.
        mapping_types: Optional mapping configuration override keyed by type.
        batch_size: Number of features per mapping request batch.
        parallel_types: Dispatch mapping type requests concurrently across all
            batches when ``True``.
    """

    mapping_types = mapping_types or load_mapping_type_config()
    results: dict[str, PlateauFeature] = {f.feature_id: f for f in features}
    batches = list(_chunked(features, batch_size))

    if parallel_types:
        batch_map: dict[int, list[PlateauFeature]] = {
            i: list(batch) for i, batch in enumerate(batches)
        }
        tasks = [
            _request_mapping(session, batch, i, key, cfg)
            for i, batch in enumerate(batches)
            for key, cfg in mapping_types.items()
        ]
        responses = await asyncio.gather(*tasks)
        for idx, key, cfg, sub_session, payload in responses:
            batch = batches[idx]
            if all(not f.mappings.get(key) for f in payload.features):
                slice_items = _top_k_items(batch, cfg.dataset)
                reminder = (
                    "Previous response contained no mappings. Select relevant items "
                    "from the reduced catalogue below."
                )
                prompt = _build_mapping_prompt(
                    batch,
                    {key: cfg},
                    item_overrides={cfg.dataset: slice_items},
                    extra_instructions=reminder,
                )
                payload = await sub_session.ask_async(
                    prompt, output_type=MappingResponse
                )
            batch_map[idx] = _merge_mapping_results(batch_map[idx], payload, {key: cfg})
        for batch in batch_map.values():
            for updated in batch:
                results[updated.feature_id] = updated
    else:
        for i, batch in enumerate(batches):
            batch_list = list(batch)
            for key, cfg in mapping_types.items():
                _, _, _, sub_session, payload = await _request_mapping(
                    session, batch, i, key, cfg
                )
                if all(not f.mappings.get(key) for f in payload.features):
                    slice_items = _top_k_items(batch, cfg.dataset)
                    reminder = (
                        "Previous response contained no mappings. Select relevant items"
                        " from the reduced catalogue below."
                    )
                    prompt = _build_mapping_prompt(
                        batch,
                        {key: cfg},
                        item_overrides={cfg.dataset: slice_items},
                        extra_instructions=reminder,
                    )
                    payload = await sub_session.ask_async(
                        prompt, output_type=MappingResponse
                    )
                batch_list = _merge_mapping_results(batch_list, payload, {key: cfg})
            for updated in batch_list:
                results[updated.feature_id] = updated

    return [results[f.feature_id] for f in features]


def map_features(
    session: ConversationSession,
    features: Sequence[PlateauFeature],
    mapping_types: Mapping[str, MappingTypeConfig] | None = None,
    *,
    batch_size: int = 30,
    parallel_types: bool = True,
) -> list[PlateauFeature]:
    """Return ``features`` augmented with mapping information.

    Args:
        session: Active conversation session used for mapping requests.
        features: Plateau features requiring mapping enrichment.
        mapping_types: Optional mapping configuration override keyed by type.
        batch_size: Number of features per mapping request batch.
        parallel_types: Dispatch mapping type requests concurrently across all
            batches when ``True``.
    """

    return asyncio.run(
        map_features_async(
            session,
            features,
            mapping_types,
            batch_size=batch_size,
            parallel_types=parallel_types,
        )
    )


__all__ = [
    "map_feature",
    "map_feature_async",
    "map_features",
    "map_features_async",
    "MappingError",
    "init_embeddings",
]
