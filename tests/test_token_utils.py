"""Tests for the :mod:`token_utils` module."""

from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import token_utils


def test_estimate_tokens_with_tiktoken(monkeypatch) -> None:
    """Ensure ``estimate_tokens`` uses ``tiktoken`` when available."""
    dummy_module = types.ModuleType("tiktoken")
    dummy_module.get_encoding = lambda name: SimpleNamespace(
        encode=lambda text: list(text)
    )
    monkeypatch.setitem(sys.modules, "tiktoken", dummy_module)
    importlib.reload(token_utils)

    prompt = "hello"
    assert token_utils.estimate_tokens(prompt, 2) == len(prompt) + 2


def test_estimate_tokens_without_tiktoken(monkeypatch) -> None:
    """Fallback path when ``tiktoken`` is not installed."""
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    importlib.reload(token_utils)

    prompt = "hello world"
    assert token_utils.estimate_tokens(prompt, 3) == len(prompt) // 4 + 3
