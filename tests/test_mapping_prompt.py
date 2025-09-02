# SPDX-License-Identifier: MIT
"""Tests for prompt rendering utilities."""

from __future__ import annotations

import importlib
import re
import sys
import types
from typing import Sequence

import pytest
from pydantic_core import from_json

# Replace ``io_utils.loader`` with a stub during ``mapping_prompt`` import to avoid
# reading actual prompt files. The real module is restored immediately so other
# tests see the genuine implementation during collection.
_real_loader = importlib.import_module("io_utils.loader")
stub_loader = types.ModuleType("io_utils.loader")
stub_loader.load_prompt_text = lambda name: ""  # type: ignore[attr-defined]
sys.modules["io_utils.loader"] = stub_loader
import core.mapping_prompt as mapping_prompt  # noqa: E402

sys.modules["io_utils.loader"] = _real_loader
from core.mapping_prompt import render_set_prompt  # noqa: E402
from models import MappingItem, MaturityScore, PlateauFeature  # noqa: E402


@pytest.mark.parametrize("shuffle", [False, True])
def test_render_set_prompt_orders_content(shuffle: bool, monkeypatch) -> None:
    """Catalogue items and features are sorted deterministically."""

    template = "{mapping_sections}\n{features}"
    monkeypatch.setattr("core.mapping_prompt.load_prompt_text", lambda _n: template)

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

    prompt = render_set_prompt(
        "test",
        items,
        features,
        service_name="svc",
        service_description="desc",
        plateau=1,
    )
    blocks = re.findall(r"```json\n(.*?)\n```", prompt, re.DOTALL)
    items_json = from_json(blocks[0])
    features_json = from_json(blocks[1])
    assert [i["id"] for i in items_json] == ["A", "B"]
    assert [f["id"] for f in features_json] == ["1", "2"]

    if shuffle:
        prompt_again = render_set_prompt(
            "test",
            list(reversed(items)),
            list(reversed(features)),
            service_name="svc",
            service_description="desc",
            plateau=1,
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
    data = from_json(result)
    assert data == [{"id": "A B", "name": "Item Name", "description": "desc more"}]


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
    data = from_json(result)
    assert data == [{"id": "1 2", "name": "First Feature", "description": "desc more"}]


def test_render_set_prompt_normalizes_whitespace(monkeypatch) -> None:
    """Whitespace is sanitised when rendering the full prompt."""

    template = "{mapping_sections}\n{features}"
    monkeypatch.setattr("core.mapping_prompt.load_prompt_text", lambda _n: template)

    items = [
        MappingItem(id="A\nB", name="Item\tName", description="desc"),
    ]
    features = [
        PlateauFeature(
            feature_id="1\t2",
            name="First\nFeature",
            description="desc\tmore",
            score=MaturityScore(level=1, label="Initial", justification="j"),
            customer_type="learners",
        )
    ]

    prompt = render_set_prompt(
        "test",
        items,
        features,
        service_name="svc",
        service_description="desc",
        plateau=1,
    )
    blocks = re.findall(r"```json\n(.*?)\n```", prompt, re.DOTALL)
    items_json = from_json(blocks[0])
    features_json = from_json(blocks[1])
    assert items_json == [{"id": "A B", "name": "Item Name", "description": "desc"}]
    assert features_json == [
        {"id": "1 2", "name": "First Feature", "description": "desc more"}
    ]


def test_render_set_prompt_uses_diagnostics_template(monkeypatch) -> None:
    """Diagnostics mode loads the alternate prompt template."""

    called: dict[str, str] = {}

    def fake_load(name: str) -> str:
        called["name"] = name
        return ""

    monkeypatch.setattr("core.mapping_prompt.load_prompt_text", fake_load)
    render_set_prompt(
        "test",
        [],
        [],
        service_name="svc",
        service_description="desc",
        plateau=1,
        diagnostics=True,
    )
    assert called["name"] == "mapping_prompt_diagnostics"


def test_render_set_prompt_handles_literal_braces(monkeypatch) -> None:
    """Unescaped braces in templates are preserved without KeyError."""

    template = (
        "{mapping_sections}\n"
        'Each array element must be an object with only one field: { "item": <ID> }\n'
    )
    monkeypatch.setattr("core.mapping_prompt.load_prompt_text", lambda _n: template)

    result = render_set_prompt(
        "test",
        [],
        [],
        service_name="svc",
        service_description="desc",
        plateau=1,
    )
    assert '{ "item": <ID> }' in result


def test_render_set_prompt_inserts_service_metadata(monkeypatch) -> None:
    """Service placeholders are replaced in the rendered prompt."""

    template = "{service_name}|{service_description}|{plateau}"
    monkeypatch.setattr("core.mapping_prompt.load_prompt_text", lambda _n: template)
    result = render_set_prompt(
        "test",
        [],
        [],
        service_name="svc",
        service_description="desc",
        plateau=2,
    )
    assert result == "svc|desc|2"
