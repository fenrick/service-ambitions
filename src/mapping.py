"""Utilities for enriching plateau features with mapping data.

Mapping information such as related applications or technologies is gathered by
querying the language model with a consolidated prompt. The helper functions
here prepare prompt content, validate responses and merge the returned
contributions back into :class:`PlateauFeature` objects.

Embedding vectors for catalogue entries and individual features are cached at
module scope for the lifetime of the process. These caches avoid redundant
requests to the embedding service and are cleared only when the interpreter
restarts.
"""

from __future__ import annotations

import asyncio
import json
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Mapping, Sequence

import logfire
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import (  # type: ignore[import-untyped]
    TfidfVectorizer,
)

from generator import _with_retry
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

MIN_MAPPING_ITEMS = 2
"""Minimum number of mapping contributions required per feature."""

SECOND_PASS_TIMEOUT = 20.0
"""Timeout in seconds for second-pass mapping prompts."""

SECOND_PASS_ATTEMPTS = 2
"""Maximum retry attempts for second-pass mapping prompts."""

_QUARANTINED_MAPPINGS: list[Path] = []
"""Paths of mapping responses that failed to produce required items."""


async def init_embeddings() -> None:
    """Legacy no-op kept for backwards compatibility."""
    return None


def _quarantine_mapping(
    feature: PlateauFeature, key: str, payload: MappingResponse
) -> Path:
    """Persist raw mapping ``payload`` for ``feature`` and record its path."""

    qdir = Path("quarantine/mappings")
    qdir.mkdir(parents=True, exist_ok=True)
    file_path = qdir / f"{feature.feature_id}_{key}.json"
    file_path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")
    logfire.warning(
        f"Quarantined mapping payload for {feature.feature_id}",
        key=key,
        path=str(file_path),
    )
    _QUARANTINED_MAPPINGS.append(file_path)
    return file_path


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
    *,
    exhaustive: bool = True,
    max_items_per_mapping: int | None = None,
    second_pass_timeout: float = SECOND_PASS_TIMEOUT,
    second_pass_attempts: int = SECOND_PASS_ATTEMPTS,
) -> PlateauFeature:
    """Asynchronously return ``feature`` augmented with mapping data.

    Args:
        session: Active conversation session used for mapping requests.
        feature: Plateau feature requiring mapping enrichment.
        mapping_types: Optional mapping configuration override keyed by type.
        exhaustive: Retry to fill missing mappings when ``True``.
        max_items_per_mapping: Limit mapping items per type when provided.
        second_pass_timeout: Timeout in seconds for second-pass mapping prompts.
        second_pass_attempts: Maximum retry attempts for second-pass prompts.
    """

    return (
        await map_features_async(
            session,
            [feature],
            mapping_types,
            exhaustive=exhaustive,
            max_items_per_mapping=max_items_per_mapping,
            second_pass_timeout=second_pass_timeout,
            second_pass_attempts=second_pass_attempts,
        )
    )[0]


def map_feature(
    session: ConversationSession,
    feature: PlateauFeature,
    mapping_types: Mapping[str, MappingTypeConfig] | None = None,
    *,
    exhaustive: bool = True,
    max_items_per_mapping: int | None = None,
    second_pass_timeout: float = SECOND_PASS_TIMEOUT,
    second_pass_attempts: int = SECOND_PASS_ATTEMPTS,
) -> PlateauFeature:
    """Return ``feature`` augmented with mapping information.

    Args:
        session: Active conversation session used for mapping requests.
        feature: Plateau feature requiring mapping enrichment.
        mapping_types: Optional mapping configuration override keyed by type.
        exhaustive: Retry to fill missing mappings when ``True``.
        max_items_per_mapping: Limit mapping items per type when provided.
        second_pass_timeout: Timeout in seconds for second-pass mapping prompts.
        second_pass_attempts: Maximum retry attempts for second-pass prompts.
    """

    return asyncio.run(
        map_feature_async(
            session,
            feature,
            mapping_types,
            exhaustive=exhaustive,
            max_items_per_mapping=max_items_per_mapping,
            second_pass_timeout=second_pass_timeout,
            second_pass_attempts=second_pass_attempts,
        )
    )


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


def _clean_mapping_values(
    key: str,
    values: list[Contribution],
    valid_ids: dict[str, set[str]],
) -> tuple[list[Contribution], bool]:
    """Return filtered ``values`` and flag unknown IDs for ``key``."""

    valid: list[Contribution] = []
    unknown = False
    for item in values:
        if item.item not in valid_ids[key]:
            unknown = True
            continue
        valid.append(item)
    return valid, unknown


def _merge_mapping_results(
    features: Sequence[PlateauFeature],
    payload: MappingResponse,
    mapping_types: Mapping[str, MappingTypeConfig],
    *,
    max_items_per_mapping: int | None = None,
    catalogue_items: Mapping[str, list[MappingItem]] | None = None,
) -> list[PlateauFeature]:
    """Return ``features`` merged with mapping ``payload``.

    Any mappings referencing unknown item identifiers are dropped rather than
    causing an error. This allows the calling code to rerun generation without
    manual intervention when the agent invents IDs.
    """
    # Build a lookup of valid item identifiers for each mapping type to
    # prevent the agent from inventing IDs that do not exist in reference
    # datasets.
    catalogues = catalogue_items or load_mapping_items(
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
        update_data = {}
        for key in mapping_types.keys():
            original = mapped.get(key, [])
            cleaned, unknown = _clean_mapping_values(key, original, valid_ids)
            if unknown:
                _quarantine_mapping(feature, key, payload)
            if not cleaned:
                logfire.warning(
                    f"Missing mappings: feature={feature.feature_id} key={key}"
                )
            if max_items_per_mapping is not None:
                cleaned = cleaned[:max_items_per_mapping]
            update_data[key] = cleaned
        merged = feature.model_copy(update={"mappings": feature.mappings | update_data})
        results.append(merged)
    return results


async def map_set(
    session: "ConversationSession",
    set_name: str,
    items: Sequence[MappingItem],
    features: Sequence[PlateauFeature],
) -> list[PlateauFeature]:
    """Return ``features`` with ``set_name`` mappings populated."""

    cfg = MappingTypeConfig(dataset=set_name, label=set_name)
    prompt = _build_mapping_prompt(
        features,
        {set_name: cfg},
        item_overrides={set_name: list(items)},
    )
    payload = await session.ask_async(prompt, output_type=MappingResponse)
    return _merge_mapping_results(
        features,
        payload,
        {set_name: cfg},
        catalogue_items={set_name: list(items)},
    )


async def _reprompt_feature(
    session: "ConversationSession",
    feature: PlateauFeature,
    key: str,
    cfg: MappingTypeConfig,
    *,
    max_items_per_mapping: int | None = None,
) -> PlateauFeature:
    """Return ``feature`` with mappings filled using a reduced catalogue."""

    slice_items = _top_k_items([feature], cfg.dataset)
    reminder = (
        "Previous response contained insufficient mappings. Select relevant"
        " items from the reduced catalogue below."
    )
    prompt = _build_mapping_prompt(
        [feature],
        {key: cfg},
        item_overrides={cfg.dataset: slice_items},
        extra_instructions=reminder,
    )
    payload = await session.ask_async(prompt, output_type=MappingResponse)
    merged = _merge_mapping_results(
        [feature], payload, {key: cfg}, max_items_per_mapping=max_items_per_mapping
    )[0]
    if len(merged.mappings.get(key, [])) < MIN_MAPPING_ITEMS:
        _quarantine_mapping(feature, key, payload)
    return merged


async def _map_parallel(
    session: "ConversationSession",
    batches: list[Sequence[PlateauFeature]],
    mapping_types: Mapping[str, MappingTypeConfig],
    *,
    exhaustive: bool = True,
    max_items_per_mapping: int | None = None,
    second_pass_timeout: float = SECOND_PASS_TIMEOUT,
    second_pass_attempts: int = SECOND_PASS_ATTEMPTS,
) -> dict[str, PlateauFeature]:
    """Return mapping results when mapping types run in parallel."""

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
        if exhaustive and all(not f.mappings.get(key) for f in payload.features):
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

            # Second pass uses reduced timeout/attempts to limit tail latency.
            async def _second_pass(
                s: "ConversationSession" = sub_session, p: str = prompt
            ) -> MappingResponse:
                return await s.ask_async(p, output_type=MappingResponse)

            payload = await _with_retry(
                _second_pass,
                request_timeout=second_pass_timeout,
                attempts=second_pass_attempts,
            )
        merged = _merge_mapping_results(
            batch_map[idx],
            payload,
            {key: cfg},
            max_items_per_mapping=max_items_per_mapping,
        )
        for j, feat in enumerate(merged):
            if exhaustive and len(feat.mappings.get(key, [])) < MIN_MAPPING_ITEMS:
                # Retry only the under-filled feature to minimise extra cost.
                merged[j] = await _reprompt_feature(
                    sub_session,
                    feat,
                    key,
                    cfg,
                    max_items_per_mapping=max_items_per_mapping,
                )
        batch_map[idx] = merged
    results: dict[str, PlateauFeature] = {}
    for batch in batch_map.values():
        for updated in batch:
            results[updated.feature_id] = updated
    return results


async def _map_sequential(
    session: "ConversationSession",
    batches: list[Sequence[PlateauFeature]],
    mapping_types: Mapping[str, MappingTypeConfig],
    *,
    exhaustive: bool = True,
    max_items_per_mapping: int | None = None,
    second_pass_timeout: float = SECOND_PASS_TIMEOUT,
    second_pass_attempts: int = SECOND_PASS_ATTEMPTS,
) -> dict[str, PlateauFeature]:
    """Return mapping results when mapping types run sequentially."""

    results: dict[str, PlateauFeature] = {}
    for i, batch in enumerate(batches):
        batch_list = list(batch)
        for key, cfg in mapping_types.items():
            _, _, _, sub_session, payload = await _request_mapping(
                session, batch, i, key, cfg
            )
            if exhaustive and all(not f.mappings.get(key) for f in payload.features):
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

                # Second pass uses reduced timeout/attempts to limit tail latency.
                async def _second_pass(
                    s: "ConversationSession" = sub_session, p: str = prompt
                ) -> MappingResponse:
                    return await s.ask_async(p, output_type=MappingResponse)

                payload = await _with_retry(
                    _second_pass,
                    request_timeout=second_pass_timeout,
                    attempts=second_pass_attempts,
                )
            batch_list = _merge_mapping_results(
                batch_list,
                payload,
                {key: cfg},
                max_items_per_mapping=max_items_per_mapping,
            )
            for j, feat in enumerate(batch_list):
                if exhaustive and len(feat.mappings.get(key, [])) < MIN_MAPPING_ITEMS:
                    # Request additional mappings for under-filled features.
                    batch_list[j] = await _reprompt_feature(
                        sub_session,
                        feat,
                        key,
                        cfg,
                        max_items_per_mapping=max_items_per_mapping,
                    )
        for updated in batch_list:
            results[updated.feature_id] = updated
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
    prompt = _build_mapping_prompt(batch, {key: cfg})
    logfire.debug(f"Requesting {key} mappings for {len(batch)} features")

    # Configure retry parameters using session attributes when available to
    # align mapping requests with the generator's behaviour.
    request_timeout = getattr(session, "request_timeout", 60.0)
    attempts = getattr(session, "retries", 5)
    base_delay = getattr(session, "retry_base_delay", 0.5)
    limiter = getattr(session, "_limiter", None)
    on_retry_after = limiter.throttle if limiter else None
    metrics = getattr(session, "_metrics", None)

    async def _send_prompt() -> MappingResponse:
        return await sub_session.ask_async(prompt, output_type=MappingResponse)

    payload = await _with_retry(
        _send_prompt,
        request_timeout=request_timeout,
        attempts=attempts,
        base=base_delay,
        on_retry_after=on_retry_after,
        metrics=metrics,
    )
    return batch_index, key, cfg, sub_session, payload


async def map_features_async(
    session: ConversationSession,
    features: Sequence[PlateauFeature],
    mapping_types: Mapping[str, MappingTypeConfig] | None = None,
    *,
    strict: bool = False,
    batch_size: int = 30,
    parallel_types: bool = True,
    second_pass_timeout: float = SECOND_PASS_TIMEOUT,
    second_pass_attempts: int = SECOND_PASS_ATTEMPTS,
    exhaustive: bool = True,
    max_items_per_mapping: int | None = None,
) -> list[PlateauFeature]:
    """Asynchronously return ``features`` with mapping information.

    Args:
        session: Active conversation session used for mapping requests.
        features: Plateau features requiring mapping enrichment.
        mapping_types: Optional mapping configuration override keyed by type.
        strict: Enforce non-empty mappings for all requested types when ``True``.
        batch_size: Number of features per mapping request batch.
        parallel_types: Dispatch mapping type requests concurrently across all
            batches when ``True``.
        second_pass_timeout: Timeout in seconds for second-pass mapping prompts.
        second_pass_attempts: Maximum retry attempts for second-pass prompts.
        exhaustive: Retry to fill missing mappings when ``True``.
        max_items_per_mapping: Limit mapping items per type when provided.
    """

    _QUARANTINED_MAPPINGS.clear()
    mapping_types = mapping_types or load_mapping_type_config()
    batches = list(_chunked(features, batch_size))
    if parallel_types:
        results = await _map_parallel(
            session,
            batches,
            mapping_types,
            exhaustive=exhaustive,
            max_items_per_mapping=max_items_per_mapping,
            second_pass_timeout=second_pass_timeout,
            second_pass_attempts=second_pass_attempts,
        )
    else:
        results = await _map_sequential(
            session,
            batches,
            mapping_types,
            exhaustive=exhaustive,
            max_items_per_mapping=max_items_per_mapping,
            second_pass_timeout=second_pass_timeout,
            second_pass_attempts=second_pass_attempts,
        )
    if _QUARANTINED_MAPPINGS:
        logfire.warning(
            f"Quarantined {len(_QUARANTINED_MAPPINGS)} mapping payload(s)",
            paths=[str(p) for p in _QUARANTINED_MAPPINGS],
        )
    ordered = [results[f.feature_id] for f in features]
    if strict:
        for feature in ordered:
            for key in mapping_types.keys():
                values = feature.mappings.get(key)
                if not values:
                    raise MappingError(
                        f"Missing mappings for feature {feature.feature_id} type {key}"
                    )
    return ordered


def map_features(
    session: ConversationSession,
    features: Sequence[PlateauFeature],
    mapping_types: Mapping[str, MappingTypeConfig] | None = None,
    *,
    strict: bool = False,
    batch_size: int = 30,
    parallel_types: bool = True,
    second_pass_timeout: float = SECOND_PASS_TIMEOUT,
    second_pass_attempts: int = SECOND_PASS_ATTEMPTS,
    exhaustive: bool = True,
    max_items_per_mapping: int | None = None,
) -> list[PlateauFeature]:
    """Return ``features`` augmented with mapping information.

    Args:
        session: Active conversation session used for mapping requests.
        features: Plateau features requiring mapping enrichment.
        mapping_types: Optional mapping configuration override keyed by type.
        strict: Enforce non-empty mappings for all requested types when ``True``.
        batch_size: Number of features per mapping request batch.
        parallel_types: Dispatch mapping type requests concurrently across all
            batches when ``True``.
        second_pass_timeout: Timeout in seconds for second-pass mapping prompts.
        second_pass_attempts: Maximum retry attempts for second-pass prompts.
        exhaustive: Retry to fill missing mappings when ``True``.
        max_items_per_mapping: Limit mapping items per type when provided.
    """

    return asyncio.run(
        map_features_async(
            session,
            features,
            mapping_types,
            strict=strict,
            batch_size=batch_size,
            parallel_types=parallel_types,
            second_pass_timeout=second_pass_timeout,
            second_pass_attempts=second_pass_attempts,
            exhaustive=exhaustive,
            max_items_per_mapping=max_items_per_mapping,
        )
    )


__all__ = [
    "map_feature",
    "map_feature_async",
    "map_features",
    "map_features_async",
    "map_set",
    "MappingError",
    "init_embeddings",
]
