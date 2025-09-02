# SPDX-License-Identifier: MIT
"""Integration tests for CLI subcommands."""

import sys
from pathlib import Path
from types import SimpleNamespace

import cli.main as cli
from runtime.environment import RuntimeEnv


def _prepare_settings():
    """Return minimal settings namespace for CLI tests."""

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
        cache_dir=Path(".cache"),
    )


def test_run_invokes_generator(monkeypatch):
    """The run subcommand should call the generator without enabling diagnostics."""

    called = {}

    async def fake_generate(args, transcripts_dir):
        called["args"] = args
        called["settings"] = RuntimeEnv.instance().settings

    monkeypatch.setattr(cli, "_cmd_generate_evolution", fake_generate)
    monkeypatch.setattr(cli, "load_settings", _prepare_settings)
    monkeypatch.setattr(cli, "_configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["main", "run", "--dry-run"])

    cli.main()

    assert called["args"].dry_run is True
    assert called["settings"].diagnostics is False


def test_validate_sets_dry_run(monkeypatch):
    """Validate subcommand performs a dry run by default."""

    called = {}

    async def fake_generate(args, transcripts_dir):
        called["args"] = args
        called["settings"] = RuntimeEnv.instance().settings

    monkeypatch.setattr(cli, "_cmd_generate_evolution", fake_generate)
    monkeypatch.setattr(cli, "load_settings", _prepare_settings)
    monkeypatch.setattr(cli, "_configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["main", "validate"])

    cli.main()

    assert called["args"].dry_run is True
    assert hasattr(called["args"], "transcripts_dir")
    assert called["args"].transcripts_dir is None


def test_cache_args_defaults(monkeypatch):
    """Cache CLI options default to read-only caching."""

    called = {}

    async def fake_generate(args, transcripts_dir):
        called["args"] = args
        called["settings"] = RuntimeEnv.instance().settings

    monkeypatch.setattr(cli, "_cmd_generate_evolution", fake_generate)
    monkeypatch.setattr(cli, "load_settings", _prepare_settings)
    monkeypatch.setattr(cli, "_configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["main", "run", "--dry-run"])

    cli.main()

    args = called["args"]
    settings = called["settings"]
    assert args.use_local_cache is None
    assert args.cache_mode is None
    assert args.cache_dir is None
    assert settings.use_local_cache is True
    assert settings.cache_mode == "read"
    assert settings.cache_dir == Path(".cache")


def test_cache_args_custom(monkeypatch):
    """Custom cache options propagate to runtime."""

    called = {}

    async def fake_generate(args, transcripts_dir):
        called["args"] = args
        called["settings"] = RuntimeEnv.instance().settings

    monkeypatch.setattr(cli, "_cmd_generate_evolution", fake_generate)
    monkeypatch.setattr(cli, "load_settings", _prepare_settings)
    monkeypatch.setattr(cli, "_configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main",
            "run",
            "--dry-run",
            "--use-local-cache",
            "--cache-mode",
            "write",
            "--cache-dir",
            "/tmp/cache",
        ],
    )

    cli.main()

    args = called["args"]
    settings = called["settings"]
    assert args.use_local_cache is True
    assert args.cache_mode == "write"
    assert args.cache_dir == "/tmp/cache"
    assert settings.use_local_cache is True
    assert settings.cache_mode == "write"
    assert settings.cache_dir == Path("/tmp/cache")


def test_apply_args_to_settings_updates_settings():
    args = SimpleNamespace(
        model="m",
        descriptions_model=None,
        features_model=None,
        mapping_model=None,
        search_model=None,
        concurrency=3,
        strict_mapping=True,
        mapping_data_dir="/tmp/map",
        web_search=True,
        use_local_cache=False,
        cache_mode="off",
        cache_dir="/tmp/cache",
        strict=True,
    )
    settings = _prepare_settings()
    cli._apply_args_to_settings(args, settings)
    assert settings.model == "m"
    assert settings.concurrency == 3
    assert settings.strict_mapping is True
    assert settings.mapping_data_dir == Path("/tmp/map")
    assert settings.web_search is True
    assert settings.use_local_cache is False
    assert settings.cache_mode == "off"
    assert settings.cache_dir == Path("/tmp/cache")
    assert settings.strict is True
