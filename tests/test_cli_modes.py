# SPDX-License-Identifier: MIT
"""Integration tests for CLI subcommands."""

import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import logfire

from core.conversation import ConversationSession
from engine.processing_engine import ProcessingEngine
from runtime.environment import RuntimeEnv

cli = importlib.import_module("cli.main")


def _prepare_settings(_config: str | None = None):
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
        prompt_dir=Path("prompts"),
        mapping_data_dir=Path("data"),
        roles_file=Path("data/roles.json"),
        trace_ids=False,
    )


def _make_engine(
    tmp_path: Path, *, json_logs: bool, progress: bool = True
) -> ProcessingEngine:
    """Return ``ProcessingEngine`` with minimal arguments for progress tests."""
    svc_file = tmp_path / "services.json"
    svc_file.write_text("[]", encoding="utf-8")
    roles_file = tmp_path / "roles.json"
    roles_file.write_text("[]", encoding="utf-8")
    args = SimpleNamespace(
        output_file=str(tmp_path / "out.jsonl"),
        resume=False,
        transcripts_dir=None,
        seed=0,
        roles_file=str(roles_file),
        input_file=str(svc_file),
        max_services=None,
        progress=progress,
        temp_output_dir=None,
        dry_run=False,
        allow_prompt_logging=False,
        json_logs=json_logs,
    )
    return ProcessingEngine(args, None)


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
    assert (
        settings.cache_dir
        == Path(os.environ.get("XDG_CACHE_HOME", "/tmp")) / "service-ambitions"
    )


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


def test_version_flag_prints_version(monkeypatch, capsys):
    """`--version` should output the package version and exit."""
    monkeypatch.setattr(sys, "argv", ["main", "--version"])

    cli.main()

    out = capsys.readouterr().out
    assert "service-ambitions" in out


def test_diagnostics_flag_prints_environment(monkeypatch, capsys):
    """`--diagnostics` should output environment information."""
    monkeypatch.setattr(sys, "argv", ["main", "--diagnostics"])

    cli.main()

    out = capsys.readouterr().out
    assert "Python" in out


def test_run_passes_config_path(monkeypatch, tmp_path):
    """Providing --config forwards the path to load_settings."""
    called: dict[str, str | None] = {}

    def _fake_settings(path=None):
        called["config"] = path
        return _prepare_settings()

    async def _fake_generate(*a, **k):
        return None

    monkeypatch.setattr(cli, "load_settings", _fake_settings)
    monkeypatch.setattr(cli, "_cmd_generate_evolution", _fake_generate)
    monkeypatch.setattr(cli, "_configure_logging", lambda *a, **k: None)
    cfg = tmp_path / "alt.yaml"
    monkeypatch.setattr(sys, "argv", ["main", "run", "--dry-run", "--config", str(cfg)])

    cli.main()

    assert called["config"] == str(cfg)


def test_progress_suppressed_in_non_tty(monkeypatch, tmp_path):
    """Progress bar should not render when stdout is not a TTY."""
    engine = _make_engine(tmp_path, json_logs=False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
    assert engine._create_progress(1) is None


def test_progress_suppressed_with_json_logs(monkeypatch, tmp_path):
    """Progress bar should be disabled when structured logs are requested."""
    engine = _make_engine(tmp_path, json_logs=True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    assert engine._create_progress(1) is None


def test_trace_ids_reports_request_ids(monkeypatch, capsys):
    """`--trace-ids` prints provider request IDs for failures."""

    class _Agent:
        def run(self, _prompt, message_history=None):  # pragma: no cover - dummy
            raise RuntimeError("boom")

        def run_sync(self, _prompt, message_history=None):  # pragma: no cover
            raise RuntimeError("boom")

    async def fake_generate(_args, _dir):
        session = ConversationSession(_Agent(), diagnostics=False, log_prompts=False)
        session._handle_failure(RuntimeError("boom"), "stage", "model", 0, "req-id-1")

    monkeypatch.setattr(cli, "_cmd_generate_evolution", fake_generate)
    monkeypatch.setattr(cli, "load_settings", _prepare_settings)
    monkeypatch.setattr(
        cli,
        "_configure_logging",
        lambda *a, **k: logfire.configure(
            console=logfire.ConsoleOptions(
                min_log_level="error", show_project_link=False, verbose=True
            )
        ),
    )
    monkeypatch.setattr(sys, "argv", ["main", "run", "--trace-ids"])

    cli.main()

    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "request_id" in combined
