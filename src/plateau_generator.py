"""Plateau feature generation and service evolution utilities.

The :class:`PlateauGenerator` coordinates prompt construction, model
interaction and mapping enrichment to evolve a ``ServiceInput`` across the
defined maturity plateaus. Each plateau is handled in an isolated
``ConversationSession`` so generation and mapping for one level do not leak
history into another while still reusing the same underlying agent.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
from typing import Sequence

import logfire
from pydantic import TypeAdapter

from conversation import ConversationSession
from loader import load_app_config, load_prompt_text, load_roles
from mapping import map_features
from models import (
    DescriptionResponse,
    FeatureItem,
    PlateauFeature,
    PlateauResult,
    ServiceEvolution,
    ServiceInput,
)

logger = logging.getLogger(__name__)

# Snapshot of plateau name-to-level mapping loaded from application configuration.
# Downstream callers may override these values when a different set of plateaus
# is required, but keeping a module-level default allows CLI tools to operate
# without additional configuration.
DEFAULT_PLATEAU_MAP: dict[str, int] = load_app_config().plateau_map

# Sorted list of plateau names used to iterate in ascending maturity order. The
# ordering comes from the numeric levels in ``DEFAULT_PLATEAU_MAP``.
DEFAULT_PLATEAU_NAMES: list[str] = [
    name for name, _ in sorted(DEFAULT_PLATEAU_MAP.items(), key=lambda item: item[1])
]


# Core customer segments targeted during feature generation. These represent the
# default audience slices and should be updated if new segments are introduced.
def default_customer_types() -> list[str]:
    """Return identifiers for all configured roles."""

    return [role.identifier for role in load_roles()]


def _strip_code_fences(payload: str) -> str:
    """Return ``payload`` with surrounding Markdown code fences removed.

    Models occasionally wrap JSON in triple backticks. Downstream parsing expects
    raw JSON, so this function extracts the inner content when fences are
    present.
    """

    match = re.search(r"```(?:json)?\s*(.*?)\s*```", payload.strip(), re.DOTALL)
    if match:  # Extract content between the first pair of fences
        return match.group(1)
    return payload


class PlateauGenerator:
    """Generate plateau features and service evolution summaries."""

    @logfire.instrument()
    def __init__(
        self,
        session: ConversationSession,
        required_count: int = 5,
    ) -> None:
        """Initialise the generator.

        Args:
            session: Active conversation session for agent queries.
            required_count: Minimum number of features per customer type.
        """
        if required_count < 1:
            raise ValueError("required_count must be positive")
        self.session = session
        self.required_count = required_count
        self._service: ServiceInput | None = None

    @logfire.instrument()
    async def _request_description(
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
        response = await session.ask(prompt)
        # Remove Markdown fences if the model wrapped the JSON payload.
        response = _strip_code_fences(response)
        try:
            description = DescriptionResponse.model_validate_json(response).description
        except Exception as exc:  # pragma: no cover - logging
            logger.error("Invalid plateau description: %s", exc)
            raise ValueError("Agent returned invalid plateau description") from exc
        if not description:
            raise ValueError("'description' must be a non-empty string")
        return description

    @logfire.instrument()
    def _to_feature(self, item: FeatureItem, customer: str) -> PlateauFeature:
        """Return a :class:`PlateauFeature` built from ``item``.

        Args:
            item: Raw feature details returned by the agent.
            customer: Customer segment the feature applies to.

        Returns:
            Plateau feature populated with the provided metadata.
        """

        return PlateauFeature(
            feature_id=item.feature_id,
            name=item.name,
            description=item.description,
            score=item.score,
            customer_type=customer,
        )

    @logfire.instrument()
    def _build_plateau_prompt(self, level: int, description: str) -> str:
        """Return a prompt requesting features for ``level``."""

        roles = load_roles()
        schema = json.dumps(
            TypeAdapter(dict[str, list[FeatureItem]]).json_schema(), indent=2
        )
        role_keys = ", ".join(f'"{r.identifier}"' for r in roles)
        role_list = "\n".join(
            f'- "{r.identifier}": {r.name} - {r.description}' for r in roles
        )
        template = load_prompt_text("plateau_prompt")
        return template.format(
            required_count=self.required_count,
            service_name=self._service.name if self._service else "",
            service_description=description,
            plateau=str(level),
            schema=str(schema),
            role_keys=role_keys,
            roles=role_list,
        )

    @staticmethod
    @logfire.instrument()
    def _parse_feature_payload(
        response: str,
    ) -> dict[str, list[FeatureItem]]:
        """Return validated plateau feature details."""

        clean = _strip_code_fences(response)
        adapter = TypeAdapter(dict[str, list[FeatureItem]])
        try:
            return adapter.validate_json(clean)
        except Exception as exc:  # pragma: no cover - logging
            logger.error("Invalid JSON from feature response: %s", exc)
            raise ValueError("Agent returned invalid JSON") from exc

    @logfire.instrument()
    def _collect_features(
        self, payload: dict[str, list[FeatureItem]]
    ) -> list[PlateauFeature]:
        """Return PlateauFeature records extracted from ``payload``."""

        features: list[PlateauFeature] = []
        for role in load_roles():
            customer = role.identifier
            with logfire.span(
                "collect_features", attributes={"customer_type": customer}
            ):
                raw_features = payload.get(customer, [])
                for item in raw_features:
                    # Convert each raw item into a structured plateau feature.
                    features.append(self._to_feature(item, customer))
        return features

    @logfire.instrument()
    async def generate_plateau(
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
        of features for learners, academics and professional staff. Responses
        must contain at least ``required_count`` features for each customer
        type. Raw features are converted to :class:`PlateauFeature` objects and
        enriched using :func:`map_features` before being returned as part of a
        :class:`PlateauResult`. A :class:`ValueError` is raised if the agent
        returns invalid JSON or an insufficient number of features.
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
            description = await self._request_description(level, session)
            prompt = self._build_plateau_prompt(level, description)
            logger.info("Requesting features for level=%s", level)

            # Generate features within the provided conversation session so
            # history is isolated per plateau.
            response = await session.ask(prompt)
            payload = self._parse_feature_payload(response)
            for role in load_roles():
                items = payload.get(role.identifier, [])
                # Fail fast if the model omitted any required features for a segment.
                if len(items) < self.required_count:
                    msg = (
                        (
                            f"Expected at least {self.required_count} features for"
                            f" '{role.identifier}',"
                        ),
                        f" got {len(items)}",
                    )
                    raise ValueError(msg)
            features = self._collect_features(payload)
            # Enrich the raw features with mapping information before returning.
            mapped_result = map_features(session, features)
            if inspect.isawaitable(mapped_result):
                mapped = await mapped_result
            else:
                mapped = mapped_result
            return PlateauResult(
                plateau=level,
                plateau_name=plateau_name,
                service_description=description,
                features=mapped,
            )

    @logfire.instrument()
    async def generate_service_evolution(
        self,
        service_input: ServiceInput,
        plateau_names: Sequence[str] | None = None,
        customer_types: Sequence[str] | None = None,
    ) -> ServiceEvolution:
        """Return service evolution for selected plateaus and customers.

        Args:
            service_input: Service under evaluation.
            plateau_names: Optional plateau names to evaluate. Defaults to
                :data:`DEFAULT_PLATEAU_NAMES`.
            customer_types: Optional customer segments to include. Defaults to
                :func:`default_customer_types`.

        Returns:
            Combined evolution limited to the default plateaus and customer
            types.

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
            customer_types = list(customer_types or default_customer_types())

            async def _generate(name: str) -> PlateauResult:
                try:
                    level = DEFAULT_PLATEAU_MAP[name]
                except KeyError as exc:  # pragma: no cover - checked by tests
                    raise ValueError(f"Unknown plateau name: {name}") from exc

                # Create an isolated session for this plateau using the same agent
                # to prevent cross-plateau chatter.
                plateau_session = ConversationSession(self.session.client)
                plateau_session.add_parent_materials(service_input)
                return await self.generate_plateau(level, name, session=plateau_session)

            results = await asyncio.gather(*[_generate(name) for name in plateau_names])

            plateaus: list[PlateauResult] = []
            for result in results:
                filtered = [
                    feat
                    for feat in result.features
                    if feat.customer_type in customer_types
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
