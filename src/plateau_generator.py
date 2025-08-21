"""Plateau feature generation and service evolution utilities.

The :class:`PlateauGenerator` coordinates prompt construction, model
interaction and mapping enrichment to evolve a ``ServiceInput`` across the
defined maturity plateaus. Each plateau is handled in an isolated
``ConversationSession`` so generation and mapping for one level do not leak
history into another while still reusing the same underlying agent.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import logfire

from conversation import ConversationSession
from loader import load_plateau_definitions, load_prompt_text, load_role_ids
from mapping import map_features_async
from mapping_utils import fit_batch_to_token_cap
from models import (
    DescriptionResponse,
    FeatureItem,
    FeaturesBlock,
    PlateauDescriptionsResponse,
    PlateauFeature,
    PlateauFeaturesResponse,
    PlateauResult,
    RoleFeaturesResponse,
    ServiceEvolution,
    ServiceInput,
)
from settings import load_settings
from token_scheduler import TokenScheduler
from token_utils import estimate_tokens

A_NON_EMPTY_STRING = "'description' must be a non-empty string"

# Snapshot of plateau definitions sourced from configuration.
_PLATEAU_DEFS = load_plateau_definitions()

# Mapping from plateau name to its numeric level derived from the order of
# ``service_feature_plateaus.json``. Callers may override these defaults when a
# different set of plateaus is required, but keeping module-level fallbacks
# allows CLI tools to operate without additional configuration.
DEFAULT_PLATEAU_MAP: dict[str, int] = {
    plateau.name: idx + 1 for idx, plateau in enumerate(_PLATEAU_DEFS)
}

# Ordered list of plateau names used to iterate in ascending maturity.
DEFAULT_PLATEAU_NAMES: list[str] = [plateau.name for plateau in _PLATEAU_DEFS]

# Core roles targeted during feature generation. These represent the default
# audience slices and should be updated if new roles are introduced.
DEFAULT_ROLE_IDS: list[str] = load_role_ids()


class PlateauGenerator:
    """Generate plateau features and service evolution summaries."""

    def __init__(
        self,
        session: ConversationSession,
        required_count: int = 5,
        roles: Sequence[str] | None = None,
        *,
        description_session: ConversationSession | None = None,
        mapping_session: ConversationSession | None = None,
        mapping_batch_size: int = 30,
        mapping_parallel_types: bool = True,
        mapping_token_cap: int = 8000,
    ) -> None:
        """Initialise the generator.

        Args:
            session: Active conversation session for feature generation.
            required_count: Minimum number of features per role.
            roles: Role identifiers to include during generation.
            description_session: Session used for plateau descriptions.
            mapping_session: Session used for feature mapping.
            mapping_batch_size: Number of features per mapping request batch.
            mapping_parallel_types: Dispatch mapping type requests concurrently
                across all batches when ``True``.
            mapping_token_cap: Maximum tokens permitted for a mapping request
                batch.
        """
        if required_count < 1:
            raise ValueError("required_count must be positive")
        self.session = session
        self.description_session = description_session or session
        self.mapping_session = mapping_session or session
        self.required_count = required_count
        self.roles = list(roles or DEFAULT_ROLE_IDS)
        self.mapping_batch_size = mapping_batch_size
        self.mapping_parallel_types = mapping_parallel_types
        self.mapping_token_cap = mapping_token_cap
        self._service: ServiceInput | None = None
        # Track quarantine file paths for invalid plateau descriptions.
        self.quarantined_descriptions: list[Path] = []

    def _quarantine_description(self, plateau_name: str, raw: str) -> Path:
        """Persist ``raw`` text for ``plateau_name`` and record its path."""

        # Create the quarantine directory if it does not yet exist.
        qdir = Path("quarantine/descriptions")
        qdir.mkdir(parents=True, exist_ok=True)
        file_path = qdir / f"{plateau_name}.txt"
        file_path.write_text(raw, encoding="utf-8")
        logfire.warning(f"Quarantined plateau description at {file_path}")
        self.quarantined_descriptions.append(file_path)
        return file_path

    def _compute_mapping_batch_size(
        self, description: str, features: Sequence[PlateauFeature]
    ) -> int:
        """Return an adjusted batch size limited by ``mapping_token_cap``."""
        return fit_batch_to_token_cap(
            features,
            min(self.mapping_batch_size, len(features)),
            self.mapping_token_cap,
            lambda fs: estimate_tokens(
                f"{description}\n"
                + "\n".join(
                    f"- {f.feature_id}: {f.name} - {f.description}" for f in fs
                ),
                0,
            ),
            label="mapping",
        )

    async def _map_features(
        self,
        session: ConversationSession,
        description: str,
        features: Sequence[PlateauFeature],
    ) -> list[PlateauFeature]:
        """Map ``features`` using a batch size constrained by ``mapping_token_cap``."""

        batch_size = self._compute_mapping_batch_size(description, features)
        return await map_features_async(
            session,
            features,
            batch_size=batch_size,
            parallel_types=self.mapping_parallel_types,
            token_cap=self.mapping_token_cap,
        )

    def _request_description(
        self, level: int, session: ConversationSession | None = None
    ) -> str:
        """Return the service description for ``level``.

        A description template is loaded from disk and rendered with the target
        ``level``. The prompt is sent to the agent via ``session`` and the JSON
        response parsed. A :class:`ValueError` is raised if the response cannot
        be validated or the description field is empty.
        """
        session = session or self.description_session
        template = load_prompt_text("description_prompt")
        schema = json.dumps(DescriptionResponse.model_json_schema(), indent=2)
        prompt = template.format(
            plateau=level,
            schema=str(schema),
        )

        # Query the model to obtain the raw response text.
        raw = session.ask(prompt)

        # Map the numeric level back to the plateau name for quarantine purposes.
        plateau_name = next(
            (n for n, lvl in DEFAULT_PLATEAU_MAP.items() if lvl == level),
            f"plateau_{level}",
        )

        try:
            response = DescriptionResponse.model_validate_json(raw)
            if not response.description:
                raise ValueError(A_NON_EMPTY_STRING)
            cleaned = self._sanitize_description(response.description)
            if not cleaned:
                raise ValueError(A_NON_EMPTY_STRING)
            return cleaned
        except Exception as exc:
            # Persist the invalid response and continue with a safe placeholder.
            self._quarantine_description(plateau_name, raw)
            logfire.error(f"Invalid plateau description for {plateau_name}: {exc}")
            return ""

    @staticmethod
    def _sanitize_description(text: str) -> str:
        """Remove any model-added preamble from ``text``.

        Some models prepend strings like "Prepared plateau-1 description for X:"
        before the actual description. This helper strips that prefix if present.
        """
        pattern = r"^Prepared plateau-\d+ description for [^:]+:\s*"
        return re.sub(pattern, "", text, flags=re.IGNORECASE)

    @staticmethod
    def _predict_token_load(text: str) -> int:
        """Return a token count prediction for ``text``."""

        return max(1, estimate_tokens(text, 0))

    def _request_descriptions(
        self,
        plateau_names: Sequence[str],
        session: ConversationSession | None = None,
    ) -> dict[str, str]:
        session = session or self.description_session
        lines: list[str] = []
        for name in plateau_names:
            try:
                level = DEFAULT_PLATEAU_MAP[name]
            except KeyError as exc:
                raise ValueError(f"Unknown plateau name: {name}") from exc
            lines.append(f"{level}. {name}")
        plateaus_str = "\n".join(lines)
        schema = json.dumps(PlateauDescriptionsResponse.model_json_schema(), indent=2)
        template = load_prompt_text("plateau_descriptions_prompt")
        prompt = template.format(plateaus=plateaus_str, schema=str(schema))
        raw = session.ask(prompt)

        try:
            data = json.loads(raw)
            items = data.get("descriptions", [])
        except Exception as exc:
            # If the overall payload is invalid JSON, quarantine each plateau.
            logfire.error(f"Invalid plateau descriptions: {exc}")
            items = []

        results: dict[str, str] = {}
        item_map = {i.get("plateau_name"): i for i in items if isinstance(i, dict)}
        for name in plateau_names:
            item = item_map.get(name)
            if item is None:
                # Missing entry for the plateau name.
                self._quarantine_description(name, raw)
                results[name] = ""
                continue
            try:
                resp = DescriptionResponse.model_validate_json(json.dumps(item))
                if not resp.description:
                    raise ValueError(A_NON_EMPTY_STRING)
                cleaned = self._sanitize_description(resp.description)
                if not cleaned:
                    raise ValueError(A_NON_EMPTY_STRING)
                results[name] = cleaned
            except Exception as exc:
                self._quarantine_description(name, raw)
                logfire.error(f"Invalid plateau description for {name}: {exc}")
                results[name] = ""
        return results

    async def _request_descriptions_async(
        self,
        plateau_names: Sequence[str],
        session: ConversationSession | None = None,
    ) -> dict[str, str]:
        """Asynchronously return descriptions for ``plateau_names``."""

        session = session or self.description_session
        lines: list[str] = []
        for name in plateau_names:
            try:
                level = DEFAULT_PLATEAU_MAP[name]
            except KeyError as exc:
                raise ValueError(f"Unknown plateau name: {name}") from exc
            lines.append(f"{level}. {name}")
        plateaus_str = "\n".join(lines)
        schema = json.dumps(PlateauDescriptionsResponse.model_json_schema(), indent=2)
        template = load_prompt_text("plateau_descriptions_prompt")
        prompt = template.format(plateaus=plateaus_str, schema=str(schema))
        raw = await session.ask_async(prompt)

        try:
            data = json.loads(raw)
            items = data.get("descriptions", [])
        except Exception as exc:
            # If the overall payload is invalid JSON, quarantine each plateau.
            logfire.error(f"Invalid plateau descriptions: {exc}")
            items = []

        results: dict[str, str] = {}
        item_map = {i.get("plateau_name"): i for i in items if isinstance(i, dict)}
        for name in plateau_names:
            item = item_map.get(name)
            if item is None:
                self._quarantine_description(name, raw)
                results[name] = ""
                continue
            try:
                resp = DescriptionResponse.model_validate_json(json.dumps(item))
                if not resp.description:
                    raise ValueError(A_NON_EMPTY_STRING)
                cleaned = self._sanitize_description(resp.description)
                if not cleaned:
                    raise ValueError(A_NON_EMPTY_STRING)
                results[name] = cleaned
            except Exception as exc:
                self._quarantine_description(name, raw)
                logfire.error(f"Invalid plateau description for {name}: {exc}")
                results[name] = ""
        return results

    def _to_feature(
        self, item: FeatureItem, role: str, plateau_name: str
    ) -> PlateauFeature:
        """Return a :class:`PlateauFeature` built from ``item``.

        Args:
            item: Raw feature details returned by the agent.
            role: Role the feature applies to.
            plateau_name: Plateau the feature belongs to.

        Returns:
            Plateau feature populated with the provided metadata.
        """

        raw = f"{item.name}|{role}|{plateau_name}".encode()
        feature_id = hashlib.sha1(raw, usedforsecurity=False).hexdigest()
        return PlateauFeature(
            feature_id=feature_id,
            name=item.name,
            description=item.description,
            score=item.score,
            customer_type=role,
        )

    def _build_plateau_prompt(self, level: int, description: str) -> str:
        """Return a prompt requesting features for ``level``."""

        schema = json.dumps(PlateauFeaturesResponse.model_json_schema(), indent=2)
        template = load_prompt_text("plateau_prompt")
        roles_str = ", ".join(f'"{r}"' for r in self.roles)
        return template.format(
            required_count=self.required_count,
            service_name=self._service.name if self._service else "",
            service_description=description,
            plateau=str(level),
            schema=str(schema),
            roles=str(roles_str),
        )

    def _collect_features(
        self, payload: PlateauFeaturesResponse, plateau_name: str
    ) -> list[PlateauFeature]:
        """Return PlateauFeature records extracted from ``payload``."""

        features: list[PlateauFeature] = []
        for role in self.roles:
            with logfire.span("collect_features", attributes={"role": role}):
                # ``PlateauFeaturesResponse.features`` is a ``FeaturesBlock``
                # model, not a dictionary. Retrieve role-specific items with
                # ``getattr`` to avoid attribute errors when roles are absent.
                raw_features = getattr(payload.features, role, [])
                for item in raw_features:
                    # Convert each raw item into a structured plateau feature.
                    features.append(self._to_feature(item, role, plateau_name))
        return features

    async def _request_role_features_async(
        self,
        level: int,
        role: str,
        description: str,
        count: int,
        session: ConversationSession,
        *,
        reason: str = "invalid features",
    ) -> list[FeatureItem]:
        """Return ``count`` features for ``role`` when initial parsing fails."""

        example = {
            "features": [
                {
                    "name": "Example feature",
                    "description": "Example description.",
                    "score": {
                        "level": 3,
                        "label": "Defined",
                        "justification": "Example justification.",
                    },
                }
            ]
        }
        schema = json.dumps(RoleFeaturesResponse.model_json_schema(), indent=2)
        prompt = (
            f"Previous output returned {reason} for role '{role}'.\nProvide exactly"
            f" {count} unique features for this role at plateau {level}.\n\nService"
            f" description:\n{description}\n\nExample"
            f" output:\n{json.dumps(example, indent=2)}\n\nJSON schema:\n{schema}"
        )
        payload = await session.ask_async(prompt, output_type=RoleFeaturesResponse)
        return payload.features

    def _validate_roles(
        self,
        role_data: dict[str, Any],
    ) -> tuple[dict[str, list[FeatureItem]], list[str], dict[str, int]]:
        """Return valid roles, invalid role names and missing counts."""

        valid: dict[str, list[FeatureItem]] = {}
        invalid: list[str] = []
        missing: dict[str, int] = {}
        for role in self.roles:
            items = role_data.get(role, [])
            try:
                role_block = RoleFeaturesResponse(features=items)
            except Exception:
                invalid.append(role)
                valid[role] = []
                continue
            valid[role] = list(role_block.features)
            if len(role_block.features) < self.required_count:
                missing[role] = self.required_count - len(role_block.features)
        return valid, invalid, missing

    async def _recover_invalid_roles(
        self,
        invalid: list[str],
        level: int,
        description: str,
        session: ConversationSession,
    ) -> dict[str, list[FeatureItem]]:
        """Return features for roles that failed validation."""

        fixes: dict[str, list[FeatureItem]] = {}
        for role in invalid:
            fixes[role] = await self._request_role_features_async(
                level, role, description, self.required_count, session
            )
        return fixes

    def _enforce_min_features(self, valid: dict[str, list[FeatureItem]]) -> None:
        """Ensure each role has at least ``required_count`` features."""

        for role in self.roles:
            items = valid.get(role, [])
            if len(items) < self.required_count:
                msg = (
                    f"Expected at least {self.required_count} features for '{role}',"
                    f" got {len(items)} after retry"
                )
                raise ValueError(msg)

    def _prepare_sessions(self, service_input: ServiceInput) -> None:
        """Attach ``service_input`` to all conversation sessions."""

        self._service = service_input
        self.session.add_parent_materials(service_input)
        self.description_session.add_parent_materials(service_input)
        self.mapping_session.add_parent_materials(service_input)

    async def _schedule_plateaus(
        self,
        plateau_names: Sequence[str],
        desc_map: Mapping[str, str],
        service_input: ServiceInput,
    ) -> list[PlateauResult]:
        """Return plateau results scheduled by token load."""
        settings = load_settings()
        scheduler = TokenScheduler(
            max_workers=min(4, len(plateau_names)),
            context_window=settings.context_window,
        )
        for name in plateau_names:
            description = desc_map[name]
            tokens = self._predict_token_load(description)

            async def task(n: str = name, d: str = description) -> PlateauResult:
                level = DEFAULT_PLATEAU_MAP.get(n)
                if level is None:
                    raise ValueError(f"Unknown plateau name: {n}")
                plateau_session = ConversationSession(
                    self.session.client,
                    stage=self.session.stage,
                    metrics=self.session.metrics,
                )
                plateau_session.add_parent_materials(service_input)
                return await self.generate_plateau_async(
                    level, n, session=plateau_session, description=d
                )

            scheduler.submit(task, tokens)
        return await scheduler.run()

    async def _assemble_evolution(
        self,
        service_input: ServiceInput,
        results: Sequence[PlateauResult],
        plateau_names: Sequence[str],
        role_ids: Sequence[str],
        transcripts_dir: Path | None,
    ) -> ServiceEvolution:
        """Return ``ServiceEvolution`` from plateau ``results``."""

        plateaus: list[PlateauResult] = []
        seen: set[str] = set()
        for result in results:
            if result.plateau_name not in plateau_names:
                raise ValueError(f"Unknown plateau name: {result.plateau_name}")
            valid: list[PlateauFeature] = []
            for feat in result.features:
                if feat.customer_type not in role_ids:
                    raise ValueError(f"Unknown customer_type: {feat.customer_type}")
                if feat.feature_id in seen:
                    logfire.warning(
                        "Duplicate feature removed",
                        feature=feat.name,
                        role=feat.customer_type,
                        plateau=result.plateau_name,
                    )
                    continue
                seen.add(feat.feature_id)
                valid.append(feat)
            plateaus.append(
                PlateauResult(
                    plateau=result.plateau,
                    plateau_name=result.plateau_name,
                    service_description=result.service_description,
                    features=valid,
                )
            )

        evolution = ServiceEvolution(service=service_input, plateaus=plateaus)
        if transcripts_dir is not None:
            payload = {
                "request": service_input.model_dump(),
                "response": evolution.model_dump(),
            }
            path = transcripts_dir / f"{service_input.service_id}.json"
            await asyncio.to_thread(
                path.write_text,
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
        return evolution

    @logfire.instrument()
    async def _request_missing_features_async(
        self,
        level: int,
        role: str,
        description: str,
        missing: int,
        session: ConversationSession,
    ) -> list[FeatureItem]:
        """Return additional features for ``role`` to meet the required count."""

        example = {
            "features": [
                {
                    "name": "Example feature",
                    "description": "Example description.",
                    "score": {
                        "level": 3,
                        "label": "Defined",
                        "justification": "Example justification.",
                    },
                }
            ]
        }
        schema = json.dumps(RoleFeaturesResponse.model_json_schema(), indent=2)
        prompt = (
            f"Previous output returned insufficient features for role '{role}'.\n"
            f"Provide exactly {missing} additional unique features for this role"
            f" at plateau {level}.\n\n"
            f"Service description:\n{description}\n\n"
            f"Example output:\n{json.dumps(example, indent=2)}\n\n"
            f"JSON schema:\n{schema}"
        )
        payload = await session.ask_async(prompt, output_type=RoleFeaturesResponse)
        return payload.features

    async def generate_plateau_async(
        self,
        level: int,
        plateau_name: str,
        *,
        session: ConversationSession | None = None,
        description: str,
    ) -> PlateauResult:
        """Asynchronously return mapped plateau features for ``level``.

        Args:
            level: Numeric plateau level.
            plateau_name: Human readable name of the plateau.
            session: Conversation session used for feature generation.
            description: Pre-fetched description for the target plateau.

        Returns:
            PlateauResult populated with mapped features.
        """

        if self._service is None:
            raise ValueError(
                "ServiceInput not set. Call generate_service_evolution first."
            )

        session = session or self.session

        with logfire.span("generate_plateau") as span:
            span.set_attribute("service.id", self._service.service_id)
            span.set_attribute("plateau", level)

            prompt = self._build_plateau_prompt(level, description)
            logfire.info(f"Requesting features for level={level}")

            try:
                raw = await session.ask_async(prompt)
                data = json.loads(raw)
            except Exception as exc:
                logfire.error(f"Invalid JSON from feature response: {exc}")
                raise ValueError("Agent returned invalid JSON") from exc

            if not isinstance(data, dict) or "features" not in data:
                raise ValueError("Agent returned invalid JSON")

            role_data = data.get("features", {})
            valid, invalid_roles, missing = self._validate_roles(role_data)

            fixes = await self._recover_invalid_roles(
                invalid_roles, level, description, session
            )
            valid.update(fixes)

            tasks = {
                role: asyncio.create_task(
                    self._request_missing_features_async(
                        level, role, description, need, session
                    )
                )
                for role, need in missing.items()
            }
            if tasks:
                results = await asyncio.gather(*tasks.values())
                for role, extras in zip(tasks.keys(), results, strict=False):
                    valid[role].extend(extras)

            self._enforce_min_features(valid)

            block = FeaturesBlock(
                learners=valid.get("learners", []),
                academics=valid.get("academics", []),
                professional_staff=valid.get("professional_staff", []),
            )
            payload = PlateauFeaturesResponse(features=block)

            features = self._collect_features(payload, plateau_name)
            map_session = ConversationSession(
                self.mapping_session.client,
                stage=self.mapping_session.stage,
                metrics=self.mapping_session.metrics,
            )
            if self._service is not None:
                map_session.add_parent_materials(self._service)
            mapped = await self._map_features(map_session, description, features)
            return PlateauResult(
                plateau=level,
                plateau_name=plateau_name,
                service_description=description,
                features=mapped,
            )

    def generate_plateau(
        self,
        level: int,
        plateau_name: str,
        *,
        session: ConversationSession | None = None,
        description: str,
    ) -> PlateauResult:
        """Return mapped plateau features for ``level``.

        Args:
            level: Numeric plateau level.
            plateau_name: Human readable name of the plateau.
            session: Conversation session used for feature generation.
            description: Pre-fetched description for the target plateau.
        """

        return asyncio.run(
            self.generate_plateau_async(
                level,
                plateau_name,
                session=session,
                description=description,
            )
        )

    async def generate_service_evolution_async(
        self,
        service_input: ServiceInput,
        plateau_names: Sequence[str] | None = None,
        role_ids: Sequence[str] | None = None,
        *,
        transcripts_dir: Path | None = None,
    ) -> ServiceEvolution:
        """Asynchronously return service evolution for selected plateaus.

        Args:
            service_input: Source service details to evolve.
            plateau_names: Optional subset of plateau names to include.
            role_ids: Optional subset of role identifiers to include.
            transcripts_dir: Directory to persist per-service transcripts. ``None``
                disables transcript persistence.
        """

        self.quarantined_descriptions.clear()
        self._prepare_sessions(service_input)

        with logfire.span("generate_service_evolution") as span:
            span.set_attribute("service.id", service_input.service_id)
            if service_input.customer_type:
                span.set_attribute("customer_type", service_input.customer_type)

            plateau_names = list(plateau_names or DEFAULT_PLATEAU_NAMES)
            role_ids = list(role_ids or self.roles)

            desc_map = await self._request_descriptions_async(
                plateau_names, session=self.description_session
            )

            results = await self._schedule_plateaus(
                plateau_names, desc_map, service_input
            )

            evolution = await self._assemble_evolution(
                service_input, results, plateau_names, role_ids, transcripts_dir
            )

            if self.quarantined_descriptions:
                logfire.warning(
                    f"Quarantined {len(self.quarantined_descriptions)} plateau"
                    " description(s)",
                    paths=[str(p) for p in self.quarantined_descriptions],
                )

            return evolution

    def generate_service_evolution(
        self,
        service_input: ServiceInput,
        plateau_names: Sequence[str] | None = None,
        role_ids: Sequence[str] | None = None,
        *,
        transcripts_dir: Path | None = None,
    ) -> ServiceEvolution:
        """Return service evolution for selected plateaus and roles."""

        return asyncio.run(
            self.generate_service_evolution_async(
                service_input,
                plateau_names,
                role_ids,
                transcripts_dir=transcripts_dir,
            )
        )
