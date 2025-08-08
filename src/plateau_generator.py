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
DEFAULT_CUSTOMER_TYPES: list[str] = ["learners", "staff", "community"]


class PlateauGenerator:
    """Generate plateau features and service evolution summaries."""

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

    def _request_description(self, level: int) -> str:
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
        response = self.session.ask(prompt)
        try:
            description = DescriptionResponse.model_validate_json(response).description
        except Exception as exc:  # pragma: no cover - logging
            logger.error("Invalid plateau description: %s", exc)
            raise ValueError("Agent returned invalid plateau description") from exc
        if not description:
            raise ValueError("'description' must be a non-empty string")
        return description

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

    def generate_plateau(self, level: int, plateau_name: str) -> PlateauResult:
        """Return mapped plateau features for ``level``.

        Args:
            level: Numeric identifier for the plateau.
            plateau_name: Human readable name of the plateau.

        The function requests a plateau-specific service description and a list
        of features for learners, staff and community. Responses must contain at
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

        # Ask the model to describe the service at the specified plateau level.
        description = self._request_description(level)
        schema = json.dumps(PlateauFeaturesResponse.model_json_schema(), indent=2)
        template = load_prompt_text("plateau_prompt")
        prompt = template.format(
            required_count=self.required_count,
            service_name=self._service.name,
            service_description=description,
            plateau=str(level),
            schema=str(schema),
        )
        logger.info("Requesting features for level=%s", level)

        # Using the shared conversation session ensures features are generated
        # in the same context as previous interactions.
        response = self.session.ask(prompt)
        try:
            payload = PlateauFeaturesResponse.model_validate_json(response)
        except Exception as exc:  # pragma: no cover - logging
            logger.error("Invalid JSON from feature response: %s", exc)
            raise ValueError("Agent returned invalid JSON") from exc

        features: list[PlateauFeature] = []
        for customer in ("learners", "staff", "community"):
            raw_features = getattr(payload, customer)
            if len(raw_features) < self.required_count:
                raise ValueError(
                    f"Insufficient number of features returned for {customer}"
                )
            for item in raw_features:
                features.append(self._to_feature(item, customer))

        # Enrich the raw features with mapping information before returning.
        mapped = map_features(self.session, features)
        return PlateauResult(
            plateau=level,
            plateau_name=plateau_name,
            service_description=description,
            features=mapped,
        )

    def generate_service_evolution(
        self,
        service_input: ServiceInput,
    ) -> ServiceEvolution:
        """Return service evolution for selected plateaus and customers.

        Args:
            service_input: Service under evaluation.

        Returns:
            Combined evolution limited to the default plateaus and customer
            types.

        Side Effects:
            Stores ``service_input`` for subsequent plateau generation and seeds
            the conversation session with its details.
        """
        self._service = service_input

        # Seed the conversation so later model queries have the service context.
        self.session.add_parent_materials(service_input)

        plateau_names = DEFAULT_PLATEAU_NAMES
        customer_types = DEFAULT_CUSTOMER_TYPES

        plateaus: list[PlateauResult] = []
        for name in plateau_names:
            try:
                level = DEFAULT_PLATEAU_MAP[name]
            except KeyError as exc:  # pragma: no cover - checked by tests
                raise ValueError(f"Unknown plateau name: {name}") from exc

            # Generate features for each plateau level in turn.
            result = self.generate_plateau(level, name)
            filtered = [
                feat for feat in result.features if feat.customer_type in customer_types
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


__all__ = [
    "PlateauGenerator",
    "DEFAULT_PLATEAU_MAP",
    "DEFAULT_PLATEAU_NAMES",
    "DEFAULT_CUSTOMER_TYPES",
]
