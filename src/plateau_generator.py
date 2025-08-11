"""Plateau feature generation and service evolution utilities.

The :class:`PlateauGenerator` coordinates prompt construction, model
interaction and mapping enrichment to evolve a ``ServiceInput`` across the
defined maturity plateaus. Each plateau is described and its features mapped in
the context of a shared conversation session so that responses build upon prior
interactions.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Sequence

import logfire

from conversation import ConversationSession
from loader import load_app_config, load_prompt_text
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
DEFAULT_CUSTOMER_TYPES: list[str] = ["learners", "academics", "professional_staff"]


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
    async def _request_description(self, level: int) -> str:
        """Return the service description for ``level``.

        A description template is loaded from disk and rendered with the target
        ``level``. The prompt is sent to the agent via the stored conversation
        session and the JSON response parsed. A :class:`ValueError` is raised if
        the response cannot be validated or the description field is empty.
        """
        template = load_prompt_text("description_prompt")
        schema = json.dumps(DescriptionResponse.model_json_schema(), indent=2)
        prompt = template.format(
            plateau=level,
            schema=str(schema),
        )

        # Query the model using the stored conversation session so the
        # description becomes part of the evolving chat history.
        response = await self.session.ask(prompt)
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

        schema = json.dumps(PlateauFeaturesResponse.model_json_schema(), indent=2)
        template = load_prompt_text("plateau_prompt")
        return template.format(
            required_count=self.required_count,
            service_name=self._service.name if self._service else "",
            service_description=description,
            plateau=str(level),
            schema=str(schema),
        )

    @staticmethod
    @logfire.instrument()
    def _parse_feature_payload(response: str) -> PlateauFeaturesResponse:
        """Return validated plateau feature details."""

        clean = _strip_code_fences(response)
        try:
            return PlateauFeaturesResponse.model_validate_json(clean)
        except Exception as exc:  # pragma: no cover - logging
            logger.error("Invalid JSON from feature response: %s", exc)
            raise ValueError("Agent returned invalid JSON") from exc

    @logfire.instrument()
    def _collect_features(
        self, payload: PlateauFeaturesResponse
    ) -> list[PlateauFeature]:
        """Return PlateauFeature records extracted from ``payload``."""

        features: list[PlateauFeature] = []
        for customer in ("learners", "academics", "professional_staff"):
            with logfire.span(
                "collect_features", attributes={"customer_type": customer}
            ):
                raw_features = getattr(payload, customer)
                for item in raw_features:
                    # Convert each raw item into a structured plateau feature.
                    features.append(self._to_feature(item, customer))
        return features

    @logfire.instrument()
    async def generate_plateau(self, level: int, plateau_name: str) -> PlateauResult:
        """Return mapped plateau features for ``level``.

        Args:
            level: Numeric identifier for the plateau.
            plateau_name: Human readable name of the plateau.

        The function requests a plateau-specific service description and a list
        of features for learners, academics and professional staff. Responses must contain at
        least ``required_count`` features for each customer type. Raw features
        are converted to :class:`PlateauFeature` objects and enriched using
        :func:`map_features` before being returned as part of a
        :class:`PlateauResult`. A :class:`ValueError` is raised if the agent
        returns invalid JSON or an insufficient number of features.
        """
        if self._service is None:
            raise ValueError(
                "ServiceInput not set. Call generate_service_evolution first."
            )

        # Attach useful context to the span so traces include the target service
        # and plateau level being generated.
        with logfire.span("generate_plateau") as span:
            span.set_attribute("service.id", self._service.service_id)
            span.set_attribute("plateau", level)

            # Ask the model to describe the service at the specified plateau level.
            description = await self._request_description(level)
            prompt = self._build_plateau_prompt(level, description)
            logger.info("Requesting features for level=%s", level)

            # Using the shared conversation session ensures features are generated
            # in the same context as previous interactions.
            response = await self.session.ask(prompt)
            payload = self._parse_feature_payload(response)
            for segment, items in {
                "learners": payload.learners,
                "academics": payload.academics,
                "professional_staff": payload.professional_staff,
            }.items():
                # Fail fast if the model omitted any required features for a segment.
                if len(items) < self.required_count:
                    msg = (
                        (
                            f"Expected at least {self.required_count} features for"
                            f" '{segment}',"
                        ),
                        f" got {len(items)}",
                    )
                    raise ValueError(msg)
            features = self._collect_features(payload)
            # Enrich the raw features with mapping information before returning.
            mapped = await map_features(self.session, features)
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
                :data:`DEFAULT_CUSTOMER_TYPES`.

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
            customer_types = list(customer_types or DEFAULT_CUSTOMER_TYPES)

            plateaus: list[PlateauResult] = []
            for name in plateau_names:
                try:
                    level = DEFAULT_PLATEAU_MAP[name]
                except KeyError as exc:  # pragma: no cover - checked by tests
                    # Fail fast when configuration references an unknown plateau.
                    raise ValueError(f"Unknown plateau name: {name}") from exc

                # Generate features for each plateau level in turn.
                result = await self.generate_plateau(level, name)
                # Restrict features to the requested customer segments only.
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
