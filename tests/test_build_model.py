# SPDX-License-Identifier: MIT
"""Tests for ``build_model`` utility."""

import os

from generation.generator import build_model


def test_build_model_disables_web_search_by_default(monkeypatch):
    """The model should omit web search tooling unless requested."""
    model = build_model("gpt-4o-mini", "key")
    assert model._settings.get("openai_builtin_tools") is None


def test_build_model_enables_web_search_when_requested(monkeypatch):
    """Enabling the flag should attach the web search tool."""
    model = build_model("gpt-4o-mini", "key", web_search=True)
    assert model._settings["openai_builtin_tools"] == [{"type": "web_search_preview"}]


def test_build_model_sets_openai_env_vars(monkeypatch):
    """The API key is passed directly to the provider without env vars."""
    monkeypatch.delenv("SA_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    model = build_model("gpt-4o-mini", "test-key")
    assert os.getenv("SA_OPENAI_API_KEY") is None
    assert os.getenv("OPENAI_API_KEY") is None
    assert model._provider._client.api_key == "test-key"
