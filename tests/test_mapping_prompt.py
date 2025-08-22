"""Tests for prompt rendering utilities."""

from __future__ import annotations

import sys
import types
from typing import Sequence

import pytest

stub_loader = types.ModuleType("loader")
stub_loader.load_prompt_text = lambda name: ""  # type: ignore[attr-defined]
sys.modules["loader"] = stub_loader

import mapping_prompt  # noqa: E402
from mapping_prompt import render_set_prompt  # noqa: E402
from models import MappingItem, MaturityScore, PlateauFeature  # noqa: E402


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
    idx_a = lines.index("A\tItem A\tdesc")
    idx_b = lines.index("B\tItem B\tdesc")
    idx_1 = lines.index("1\tFirst\td")
    idx_2 = lines.index("2\tSecond\td")
    assert idx_a < idx_b
    assert idx_1 < idx_2

    if shuffle:
        prompt_again = render_set_prompt(
            "test", list(reversed(items)), list(reversed(features))
        )
        assert prompt_again == prompt


def test_render_items_normalizes_whitespace() -> None:
    """Newline and tab characters are replaced with spaces in items."""

    items = [
        MappingItem(
            id="A\nB",
            name="Item\tName",
            description="desc\nmore",
        )
    ]
    result = mapping_prompt._render_items(items)
    assert result == "A B\tItem Name\tdesc more"


def test_render_features_normalizes_whitespace() -> None:
    """Embedded whitespace in features is sanitized."""

    features = [
        PlateauFeature(
            feature_id="1\t2",
            name="First\nFeature",
            description="desc\tmore",
            score=MaturityScore(level=1, label="Initial", justification="j"),
            customer_type="learners",
        )
    ]
    result = mapping_prompt._render_features(features)
    assert result == "1 2\tFirst Feature\tdesc more"
