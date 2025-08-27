# SPDX-License-Identifier: MIT
"""Property-based tests for ServiceInput model."""

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from models import ServiceFeature, ServiceInput

non_empty_text = st.text(min_size=1).filter(lambda s: s.strip())


@given(
    service_id=non_empty_text,
    name=non_empty_text,
    customer_type=st.one_of(st.none(), non_empty_text),
    description=non_empty_text,
    jobs=st.lists(st.builds(dict, name=non_empty_text), min_size=1, max_size=5),
    features=st.lists(
        st.builds(
            ServiceFeature,
            feature_id=non_empty_text,
            name=non_empty_text,
            description=non_empty_text,
        ),
        max_size=3,
    ),
)
def test_service_input_validates(
    service_id, name, customer_type, description, jobs, features
):
    """ServiceInput accepts a wide range of valid values."""
    model = ServiceInput(
        service_id=service_id,
        name=name,
        customer_type=customer_type,
        description=description,
        jobs_to_be_done=jobs,
        features=features,
    )
    assert model.service_id == service_id


def test_service_input_rejects_empty_id():
    """Empty identifiers should be rejected."""
    with pytest.raises(ValidationError):
        ServiceInput(
            service_id="", name="n", description="d", jobs_to_be_done=[{"name": "j"}]
        )
