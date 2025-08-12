"""Plateau feature generation and service evolution utilities.

The :class:`PlateauGenerator` coordinates prompt construction, model
interaction and mapping enrichment to evolve a ``ServiceInput`` across the
defined maturity plateaus. Each plateau is handled in an isolated
``ConversationSession`` so generation and mapping for one level do not leak
history into another while still reusing the same underlying agent.
"""

from __future__ import annotations

import json
import logging
from typing import Sequence

import logfire

from conversation import ConversationSession
from loader import load_plateau_definitions, load_prompt_text, load_roles
from mapping import map_features
from models import (
    DescriptionResponse,
    FeatureItem,
    PlateauFeature,
    PlateauFeaturesResponse,
    PlateauResult,
    ServiceEvolution,
    ServiceInput,
)

logger = logging.getLogger(__name__)

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

    @logfire.instrument()
    def __init__(
        self,
        session: ConversationSession,
        required_count: int = 5,
        roles: Sequence[str] | None = None,
    ) -> None:
        """Initialise the generator.

        Args:
            session: Active conversation session for agent queries.
            required_count: Minimum number of features per role.
            roles: Role identifiers to include during generation.
        """
        if required_count < 1:
            raise ValueError("required_count must be positive")
        self.session = session
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
        session = session or self.session
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
        except Exception as exc:  # pragma: no cover - logging
            logger.error("Invalid plateau description: %s", exc)
            raise ValueError("Agent returned invalid plateau description") from exc
        if not response.description:
            raise ValueError("'description' must be a non-empty string")
        return response.description

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
                raw_features = payload.features.get(role, [])
                for item in raw_features:
                    # Convert each raw item into a structured plateau feature.
                    features.append(self._to_feature(item, role))
        return features

    @logfire.instrument()
    def generate_plateau(
        self,
        level: int,
        plateau_name: str,
        session: ConversationSession | None = None,
    ) -> PlateauResult:
        """Return mapped plateau features for ``level``.

        Args:
            level: Numeric identifier for the plateau.
            plateau_name: Human readable name of the plateau.
            session: Conversation session used for model interactions. Defaults
                to the instance's session.

        The function requests a plateau-specific service description and a list
        of features for each configured role. Responses must contain at least
        ``required_count`` features for every role. Raw features are converted to
        :class:`PlateauFeature` objects and enriched using :func:`map_features`
        before being returned as part of a :class:`PlateauResult`. A
        :class:`ValueError` is raised if the agent returns invalid JSON or an
        insufficient number of features.
        """
        if self._service is None:
            raise ValueError(
                "ServiceInput not set. Call generate_service_evolution first."
            )

        session = session or self.session

        # Attach useful context to the span so traces include the target service
        # and plateau level being generated.
        with logfire.span("generate_plateau") as span:
            span.set_attribute("service.id", self._service.service_id)
            span.set_attribute("plateau", level)

            # Ask the model to describe the service at the specified plateau level.
            description = self._request_description(level, session)
            prompt = self._build_plateau_prompt(level, description)
            logger.info("Requesting features for level=%s", level)

            # Generate features within the provided conversation session so
            # history is isolated per plateau.
            try:
                payload = session.ask(prompt, output_type=PlateauFeaturesResponse)
            except Exception as exc:  # pragma: no cover - logging
                logger.error("Invalid JSON from feature response: %s", exc)
                raise ValueError("Agent returned invalid JSON") from exc
            for role in self.roles:
                items = payload.features.get(role, [])
                # Fail fast if the model omitted any required features for a role.
                if len(items) < self.required_count:
                    # Construct a clear error message instead of returning a tuple
                    msg = (
                        f"Expected at least {self.required_count} features for"
                        f" '{role}', got {len(items)}"
                    )
                    raise ValueError(msg)
            features = self._collect_features(payload)
            # Enrich the raw features with mapping information before returning.
            mapped = map_features(session, features)
            return PlateauResult(
                plateau=level,
                plateau_name=plateau_name,
                service_description=description,
                features=mapped,
            )

    @logfire.instrument()
    def generate_service_evolution(
        self,
        service_input: ServiceInput,
        plateau_names: Sequence[str] | None = None,
        role_ids: Sequence[str] | None = None,
    ) -> ServiceEvolution:
        """Return service evolution for selected plateaus and roles.

        Args:
            service_input: Service under evaluation.
            plateau_names: Optional plateau names to evaluate. Defaults to
                :data:`DEFAULT_PLATEAU_NAMES`.
            role_ids: Optional role identifiers to include. Defaults to
                :data:`DEFAULT_ROLE_IDS`.

        Returns:
            Combined evolution limited to the default plateaus and roles.

        Side Effects:
            Stores ``service_input`` for subsequent plateau generation and seeds
            the conversation session with its details.
        """
        self._service = service_input

        # Record identifying attributes on the span for observability.
        with logfire.span("generate_service_evolution") as span:
            span.set_attribute("service.id", service_input.service_id)
            if service_input.customer_type:
                span.set_attribute("customer_type", service_input.customer_type)

            # Seed the conversation so later model queries have the service context.
            self.session.add_parent_materials(service_input)

            plateau_names = list(plateau_names or DEFAULT_PLATEAU_NAMES)
            role_ids = list(role_ids or self.roles)

            plateaus: list[PlateauResult] = []
            for name in plateau_names:
                try:
                    level = DEFAULT_PLATEAU_MAP[name]
                except KeyError as exc:  # pragma: no cover - checked by tests
                    raise ValueError(f"Unknown plateau name: {name}") from exc

                # Create an isolated session for this plateau using the same agent
                # to prevent cross-plateau chatter.
                plateau_session = ConversationSession(self.session.client)
                plateau_session.add_parent_materials(service_input)
                result = self.generate_plateau(level, name, session=plateau_session)

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
