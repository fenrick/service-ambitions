"""Property-based tests ensuring schema round-trips preserve data."""

from __future__ import annotations

from typing import Dict, List

from hypothesis import given
from hypothesis import strategies as st

from models import (
    CMMI_LABELS,
    Contribution,
    MappingFeature,
    MappingResponse,
    MaturityScore,
    PlateauFeature,
)

# Basic string strategy limiting size and alphabet to keep examples small.
TEXT = st.text(min_size=1, max_size=20)

# Strategy generating valid ``Contribution`` instances.
contributions = st.builds(
    Contribution,
    item=TEXT,
    contribution=st.one_of(st.none(), st.floats(min_value=0.1, max_value=1.0)),
)


# Strategy producing mapping dictionaries with up to three types and unbounded items.
def mapping_dict() -> st.SearchStrategy[Dict[str, List[Contribution]]]:
    return st.dictionaries(
        TEXT,
        st.lists(contributions),
        max_size=3,
    )


# Strategy for ``MaturityScore`` ensuring label matches level.
def maturity_scores() -> st.SearchStrategy[MaturityScore]:
    def build(level: int) -> MaturityScore:
        return MaturityScore(
            level=level,
            label=CMMI_LABELS[level],
            justification="j",
        )

    return st.integers(min_value=1, max_value=5).map(build)


# Strategy for ``PlateauFeature`` round-trip testing.
@st.composite
def plateau_features(draw) -> PlateauFeature:
    return PlateauFeature(
        feature_id=draw(TEXT),
        name=draw(TEXT),
        description=draw(TEXT),
        score=draw(maturity_scores()),
        customer_type=draw(TEXT),
        mappings=draw(mapping_dict()),
    )


# Strategy for ``MappingFeature`` used in ``MappingResponse`` tests.
@st.composite
def mapping_features(draw) -> MappingFeature:
    return MappingFeature(
        feature_id=draw(TEXT),
        mappings=draw(mapping_dict()),
    )


@given(plateau_features())
def test_plateau_feature_round_trip(feature: PlateauFeature) -> None:
    """Dumping and reloading ``PlateauFeature`` should yield an identical model."""

    payload = feature.model_dump()
    result = PlateauFeature.model_validate(payload)
    assert result.model_dump() == payload


@given(st.lists(mapping_features(), min_size=1, max_size=4))
def test_mapping_response_round_trip(features: list[MappingFeature]) -> None:
    """``MappingResponse`` serialisation round-trips correctly."""

    response = MappingResponse(features=features)
    payload = response.model_dump()
    result = MappingResponse.model_validate(payload)
    assert result.model_dump() == payload
