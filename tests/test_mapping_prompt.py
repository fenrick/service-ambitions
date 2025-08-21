"""Tests for prompt rendering utilities."""

from __future__ import annotations

from typing import Sequence

import pytest

from mapping_prompt import render_set_prompt
from models import MappingItem, MaturityScore, PlateauFeature


@pytest.mark.parametrize("shuffle", [False, True])
def test_render_set_prompt_orders_content(shuffle: bool, monkeypatch) -> None:
    """Catalogue items and features are sorted deterministically."""

    template = "{mapping_sections}\n{features}\n{schema}"
    monkeypatch.setattr("mapping_prompt.load_prompt_text", lambda _n: template)

    items: Sequence[MappingItem] = [
        MappingItem(id="B", name="Item B", description="desc"),
        MappingItem(id="A", name="Item A", description="desc"),
    ]
    features: Sequence[PlateauFeature] = [
        PlateauFeature(
            feature_id="2",
            name="Second",
            description="d",
            score=MaturityScore(level=1, label="Initial", justification="j"),
            customer_type="learners",
        ),
        PlateauFeature(
            feature_id="1",
            name="First",
            description="d",
            score=MaturityScore(level=1, label="Initial", justification="j"),
            customer_type="learners",
        ),
    ]

    if shuffle:
        items = list(reversed(items))
        features = list(reversed(features))

    prompt = render_set_prompt("test", items, features)
    lines = prompt.splitlines()
    idx_a = lines.index("- A: Item A - desc")
    idx_b = lines.index("- B: Item B - desc")
    idx_1 = lines.index("- 1: First - d")
    idx_2 = lines.index("- 2: Second - d")
    assert idx_a < idx_b
    assert idx_1 < idx_2

    if shuffle:
        prompt_again = render_set_prompt(
            "test", list(reversed(items)), list(reversed(features))
        )
        assert prompt_again == prompt
