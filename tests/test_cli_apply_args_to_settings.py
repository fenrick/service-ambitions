# SPDX-License-Identifier: MIT
"""Unit tests for applying CLI arguments to settings."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

cli = importlib.import_module("cli.main")

os.environ.setdefault("SA_OPENAI_API_KEY", "test-key")


def _prepare_settings():
    """Return minimal settings namespace for CLI tests."""

    cache_dir = Path(os.environ.get("XDG_CACHE_HOME", "/tmp")) / "service-ambitions"
    return SimpleNamespace(
        log_level="INFO",
        logfire_token=None,
        diagnostics=False,
        strict_mapping=False,
        strict=False,
        model="openai:gpt-5",
        models=None,
        use_local_cache=True,
        cache_mode="read",
        cache_dir=cache_dir,
    )


def _build_args(**overrides):
    """Construct an ``argparse.Namespace`` like object for tests."""

    base = dict(
        model=None,
        descriptions_model=None,
        features_model=None,
        mapping_model=None,
        search_model=None,
        concurrency=None,
        strict_mapping=None,
        mapping_data_dir=None,
        web_search=None,
        use_local_cache=None,
        cache_mode=None,
        cache_dir=None,
        strict=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.mark.parametrize(
    ("arg_name", "attr_name", "value", "expected"),
    [
        ("model", "model", "m", "m"),
        ("concurrency", "concurrency", 5, 5),
        ("strict_mapping", "strict_mapping", True, True),
        ("mapping_data_dir", "mapping_data_dir", "/tmp/map", Path("/tmp/map")),
        ("web_search", "web_search", True, True),
        ("use_local_cache", "use_local_cache", False, False),
        ("cache_mode", "cache_mode", "off", "off"),
        ("cache_dir", "cache_dir", "/tmp/cache", Path("/tmp/cache")),
        ("strict", "strict", True, True),
    ],
)
def test_apply_args_to_settings_simple(arg_name, attr_name, value, expected):
    """Each simple flag should override the corresponding setting."""

    args = _build_args(**{arg_name: value})
    settings = _prepare_settings()
    cli._apply_args_to_settings(args, settings)
    assert getattr(settings, attr_name) == expected


@pytest.mark.parametrize(
    ("arg_name", "field"),
    [
        ("descriptions_model", "descriptions"),
        ("features_model", "features"),
        ("mapping_model", "mapping"),
        ("search_model", "search"),
    ],
)
def test_apply_args_to_settings_stage_models(arg_name, field):
    """Per-stage model flags should populate ``settings.models``."""

    args = _build_args(**{arg_name: "m"})
    settings = _prepare_settings()
    cli._apply_args_to_settings(args, settings)
    assert getattr(settings.models, field) == "m"
