"""Plateau feature generation and service evolution utilities.

The :class:`PlateauGenerator` coordinates prompt construction, model
interaction and mapping enrichment to evolve a ``ServiceInput`` across the
defined maturity plateaus. Each plateau is handled in an isolated
``ConversationSession`` so generation and mapping for one level do not leak
history into another while still reusing the same underlying agent.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Sequence

import logfire

from conversation import ConversationSession
from loader import load_plateau_definitions, load_prompt_text, load_roles
from mapping import map_features_async
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
from token_scheduler import TokenScheduler
from token_utils import estimate_tokens

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

# Snapshot of role definitions sourced from configuration.
_ROLE_DEFS = load_roles()

# Core roles targeted during feature generation. These represent the default
# audience slices and should be updated if new roles are introduced.
DEFAULT_ROLE_IDS: list[str] = [role.role_id for role in _ROLE_DEFS]


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
    ) -> None:
        """Initialise the generator.

        Args:
            session: Active conversation session for feature generation.
            required_count: Minimum number of features per role.
            roles: Role identifiers to include during generation.
            description_session: Session used for plateau descriptions.
            mapping_session: Session used for feature mapping.
        """
        if required_count < 1:
            raise ValueError("required_count must be positive")
        self.session = session
        self.description_session = description_session or session
        self.mapping_session = mapping_session or session
        self.required_count = required_count
        self.roles = list(roles or DEFAULT_ROLE_IDS)
        self._service: ServiceInput | None = None

    @logfire.instrument()
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

        # Query the model using the provided conversation session so the
        # description becomes part of the chat history for that plateau only.
        try:
            response = session.ask(prompt, output_type=DescriptionResponse)
        except Exception as exc:
            logfire.error(f"Invalid plateau description: {exc}")
            raise ValueError("Agent returned invalid plateau description") from exc
        if not response.description:
            raise ValueError("'description' must be a non-empty string")
        cleaned = self._sanitize_description(response.description)
        if not cleaned:
            raise ValueError("'description' must be a non-empty string")
        return cleaned

    @staticmethod
    @logfire.instrument()
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

    @logfire.instrument()
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
        try:
            payload = session.ask(prompt, output_type=PlateauDescriptionsResponse)
        except Exception as exc:
            logfire.error(f"Invalid plateau descriptions: {exc}")
            raise ValueError("Agent returned invalid plateau descriptions") from exc

        results: dict[str, str] = {}
        for item in payload.descriptions:
            if not item.description:
                raise ValueError("'description' must be a non-empty string")
            cleaned = self._sanitize_description(item.description)
            if not cleaned:
                raise ValueError("'description' must be a non-empty string")
            results[item.plateau_name] = cleaned
        return results

    @logfire.instrument()
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
        try:
            payload = await session.ask_async(
                prompt, output_type=PlateauDescriptionsResponse
            )
        except Exception as exc:
            logfire.error(f"Invalid plateau descriptions: {exc}")
            raise ValueError("Agent returned invalid plateau descriptions") from exc

        results: dict[str, str] = {}
        for item in payload.descriptions:
            if not item.description:
                raise ValueError("'description' must be a non-empty string")
            cleaned = self._sanitize_description(item.description)
            if not cleaned:
                raise ValueError("'description' must be a non-empty string")
            results[item.plateau_name] = cleaned
        return results

    @logfire.instrument()
    def _to_feature(self, item: FeatureItem, role: str) -> PlateauFeature:
        """Return a :class:`PlateauFeature` built from ``item``.

        Args:
            item: Raw feature details returned by the agent.
            role: Role the feature applies to.

        Returns:
            Plateau feature populated with the provided metadata.
        """

        return PlateauFeature(
            feature_id=item.feature_id,
            name=item.name,
            description=item.description,
            score=item.score,
            customer_type=role,
        )

    @logfire.instrument()
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

    @logfire.instrument()
    def _collect_features(
        self, payload: PlateauFeaturesResponse
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
                    features.append(self._to_feature(item, role))
        return features

    @logfire.instrument()
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
                    "feature_id": f"FEAT-{level}-{role}-example",
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
                    "feature_id": f"FEAT-{level}-{role}-example",
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

    @logfire.instrument()
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
            valid: dict[str, list[FeatureItem]] = {}
            invalid_roles: list[str] = []
            missing: dict[str, int] = {}

            for role in self.roles:
                items = role_data.get(role, [])
                try:
                    block = RoleFeaturesResponse(features=items)
                except Exception:
                    invalid_roles.append(role)
                    valid[role] = []
                    continue
                valid[role] = list(block.features)
                if len(block.features) < self.required_count:
                    missing[role] = self.required_count - len(block.features)

            for role in invalid_roles:
                valid[role] = await self._request_role_features_async(
                    level, role, description, self.required_count, session
                )

            tasks = {
                role: asyncio.create_task(
                    self._request_missing_features_async(
                        level, role, description, need, session
                    )
                )
                for role, need in missing.items()
            }
            if tasks:  # Only await when there are missing roles
                results = await asyncio.gather(*tasks.values())
                for role, extras in zip(tasks.keys(), results, strict=False):
                    valid[role].extend(extras)

            for role in self.roles:
                items = valid.get(role, [])
                if len(items) < self.required_count:
                    msg = (
                        f"Expected at least {self.required_count} features for"
                        f" '{role}', got {len(items)} after retry"
                    )
                    raise ValueError(msg)

            block = FeaturesBlock(
                learners=valid.get("learners", []),
                academics=valid.get("academics", []),
                professional_staff=valid.get("professional_staff", []),
            )
            payload = PlateauFeaturesResponse(features=block)

            features = self._collect_features(payload)
            map_session = ConversationSession(self.mapping_session.client)
            if self._service is not None:
                map_session.add_parent_materials(self._service)
            mapped = await map_features_async(map_session, features)
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

    @logfire.instrument()
    async def generate_service_evolution_async(
        self,
        service_input: ServiceInput,
        plateau_names: Sequence[str] | None = None,
        role_ids: Sequence[str] | None = None,
    ) -> ServiceEvolution:
        """Asynchronously return service evolution for selected plateaus."""

        self._service = service_input

        with logfire.span("generate_service_evolution") as span:
            span.set_attribute("service.id", service_input.service_id)
            if service_input.customer_type:
                span.set_attribute("customer_type", service_input.customer_type)

            self.session.add_parent_materials(service_input)
            self.description_session.add_parent_materials(service_input)
            self.mapping_session.add_parent_materials(service_input)

            plateau_names = list(plateau_names or DEFAULT_PLATEAU_NAMES)
            role_ids = list(role_ids or self.roles)

            desc_map = await self._request_descriptions_async(
                plateau_names, session=self.description_session
            )

            scheduler = TokenScheduler(max_workers=min(4, len(plateau_names)))

            for name in plateau_names:
                description = desc_map[name]
                tokens = self._predict_token_load(description)

                async def task(n: str = name, d: str = description) -> PlateauResult:
                    try:
                        level = DEFAULT_PLATEAU_MAP[n]
                    except KeyError as exc:
                        raise ValueError(f"Unknown plateau name: {n}") from exc
                    plateau_session = ConversationSession(self.session.client)
                    plateau_session.add_parent_materials(service_input)
                    return await self.generate_plateau_async(
                        level,
                        n,
                        session=plateau_session,
                        description=d,
                    )

                scheduler.submit(task, tokens)

            results = await scheduler.run()

            plateaus: list[PlateauResult] = []
            for result in results:
                filtered = [
                    feat for feat in result.features if feat.customer_type in role_ids
                ]
                plateaus.append(
                    PlateauResult(
                        plateau=result.plateau,
                        plateau_name=result.plateau_name,
                        service_description=result.service_description,
                        features=filtered,
                    )
                )

            return ServiceEvolution(service=service_input, plateaus=plateaus)

    def generate_service_evolution(
        self,
        service_input: ServiceInput,
        plateau_names: Sequence[str] | None = None,
        role_ids: Sequence[str] | None = None,
    ) -> ServiceEvolution:
        """Return service evolution for selected plateaus and roles."""

        return asyncio.run(
            self.generate_service_evolution_async(
                service_input, plateau_names, role_ids
            )
        )
