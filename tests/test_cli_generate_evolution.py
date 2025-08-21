"""Tests for the generate-evolution CLI subcommand."""

import argparse
import asyncio
import json
import sys
from types import SimpleNamespace

import pytest

import cli
from cli import _cmd_generate_evolution
from models import SCHEMA_VERSION, ServiceEvolution, ServiceInput, ServiceMeta


class DummyFactory:
    def __init__(self, *a, **k):
        pass

    def model_name(self, stage, override=None):
        return "m"

    def get(self, stage, override=None):
        return object()


cli.ModelFactory = DummyFactory  # type: ignore[assignment,misc]


async def _noop_init_embeddings() -> None:
    """Test stub for ``init_embeddings``."""


cli.init_embeddings = _noop_init_embeddings


def test_generate_evolution_writes_results(tmp_path, monkeypatch) -> None:
    """_cmd_generate_evolution should write evolution results to disk."""
    input_path = tmp_path / "services.jsonl"
    output_path = tmp_path / "out.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "service_id": "svc-1",
                "name": "svc",
                "description": "desc",
                "customer_type": "retail",
                "jobs_to_be_done": [{"name": "job"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class DummyAgent:
        def __init__(self, model, instructions):
            self.model = model
            self.instructions = instructions

    async def fake_generate(
        self,
        service: ServiceInput,
        plateau_names=None,
        role_ids=None,
        transcripts_dir=None,
        meta: ServiceMeta | None = None,
    ) -> ServiceEvolution:
        assert meta is not None
        return ServiceEvolution(meta=meta, service=service, plateaus=[])

    monkeypatch.setattr("cli.Agent", DummyAgent)
    monkeypatch.setattr(
        "cli.PlateauGenerator.generate_service_evolution_async", fake_generate
    )
    monkeypatch.setattr("cli.configure_prompt_dir", lambda _path: None)
    monkeypatch.setattr("cli.load_evolution_prompt", lambda _ctx, _insp: "prompt")
    monkeypatch.setattr("cli.logfire.force_flush", lambda: None)

    settings = SimpleNamespace(
        model="test-model",
        log_level="INFO",
        openai_api_key="key",
        logfire_token=None,
        concurrency=2,
        prompt_dir="prompts",
        context_id="university",
        inspiration="general",
        reasoning=None,
        features_per_role=5,
        mapping_batch_size=30,
        mapping_parallel_types=True,
        exhaustive_mapping=True,
        max_items_per_mapping=None,
        mapping_feature_batch_cap_tokens=95000,
        web_search=False,
        models=None,
    )
    args = argparse.Namespace(
        input_file=str(input_path),
        output_file=str(output_path),
        model=None,
        logfire_service=None,
        log_level=None,
        verbose=0,
        max_services=None,
        dry_run=False,
        progress=False,
        concurrency=None,
        resume=False,
        seed=None,
        roles_file="data/roles.json",
        mapping_batch_size=None,
        mapping_parallel_types=None,
        exhaustive_mapping=True,
        transcripts_dir=None,
        web_search=None,
        no_logs=False,
        search_model=None,
        strict=False,
    )

    monkeypatch.setattr(cli, "_RUN_META", None)
    asyncio.run(_cmd_generate_evolution(args, settings, None))

    payload = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert payload["service"]["name"] == "svc"
    assert payload["service"]["service_id"] == "svc-1"
    assert payload["meta"]["schema_version"] == SCHEMA_VERSION


def test_generate_evolution_dry_run(tmp_path, monkeypatch) -> None:
    """Dry run should skip processing and not write output."""
    input_path = tmp_path / "services.jsonl"
    output_path = tmp_path / "out.jsonl"
    input_path.write_text("{}\n", encoding="utf-8")

    class DummyAgent:
        def __init__(self, model, instructions):
            pass

    called = {"ran": False, "embed": False}

    async def fake_init_embeddings() -> None:
        called["embed"] = True

    async def fake_generate(
        self,
        service: ServiceInput,
        plateau_names=None,
        role_ids=None,
        transcripts_dir=None,
        meta: ServiceMeta | None = None,
    ) -> ServiceEvolution:
        called["ran"] = True
        assert meta is not None
        return ServiceEvolution(meta=meta, service=service, plateaus=[])

    monkeypatch.setattr("cli.Agent", DummyAgent)
    monkeypatch.setattr(
        "cli.PlateauGenerator.generate_service_evolution_async", fake_generate
    )
    monkeypatch.setattr("cli.init_embeddings", fake_init_embeddings)
    monkeypatch.setattr("cli.configure_prompt_dir", lambda _path: None)
    monkeypatch.setattr("cli.load_evolution_prompt", lambda _ctx, _insp: "prompt")
    monkeypatch.setattr("cli.logfire.force_flush", lambda: None)

    settings = SimpleNamespace(
        model="m",
        log_level="INFO",
        openai_api_key="k",
        logfire_token=None,
        concurrency=1,
        prompt_dir="prompts",
        context_id="ctx",
        inspiration="insp",
        reasoning=None,
        features_per_role=5,
        mapping_batch_size=30,
        mapping_parallel_types=True,
        exhaustive_mapping=True,
        max_items_per_mapping=None,
        mapping_feature_batch_cap_tokens=95000,
        web_search=False,
        models=None,
    )
    args = argparse.Namespace(
        input_file=str(input_path),
        output_file=str(output_path),
        model=None,
        logfire_service=None,
        log_level=None,
        verbose=0,
        max_services=None,
        dry_run=True,
        progress=False,
        concurrency=None,
        resume=False,
        seed=None,
        roles_file="data/roles.json",
        mapping_batch_size=None,
        mapping_parallel_types=None,
        exhaustive_mapping=True,
        transcripts_dir=None,
        web_search=None,
        no_logs=False,
        search_model=None,
        strict=False,
    )

    monkeypatch.setattr(cli, "_RUN_META", None)
    asyncio.run(_cmd_generate_evolution(args, settings, None))

    assert not output_path.exists()
    assert not called["ran"]
    assert not called["embed"]


def test_generate_evolution_resume(tmp_path, monkeypatch) -> None:
    """Resume should append new results and track processed IDs."""
    input_path = tmp_path / "services.jsonl"
    output_path = tmp_path / "out.jsonl"
    processed_path = tmp_path / "processed_ids.txt"
    input_path.write_text(
        json.dumps({"service_id": "s1", "name": "svc1", "jobs_to_be_done": []})
        + "\n"
        + json.dumps({"service_id": "s2", "name": "svc2", "jobs_to_be_done": []})
        + "\n",
        encoding="utf-8",
    )
    output_path.write_text(json.dumps({"service_id": "s1"}) + "\n", encoding="utf-8")
    processed_path.write_text("s1\n", encoding="utf-8")

    class DummyAgent:
        def __init__(self, model, instructions):
            pass

    processed: list[str] = []

    async def fake_generate(
        self,
        service: ServiceInput,
        plateau_names=None,
        role_ids=None,
        transcripts_dir=None,
        meta: ServiceMeta | None = None,
    ) -> ServiceEvolution:
        processed.append(service.service_id)
        assert meta is not None
        return ServiceEvolution(meta=meta, service=service, plateaus=[])

    monkeypatch.setattr("cli.Agent", DummyAgent)
    monkeypatch.setattr(
        "cli.PlateauGenerator.generate_service_evolution_async", fake_generate
    )
    monkeypatch.setattr("cli.configure_prompt_dir", lambda _path: None)
    monkeypatch.setattr("cli.load_evolution_prompt", lambda _ctx, _insp: "prompt")
    monkeypatch.setattr("cli.logfire.force_flush", lambda: None)

    settings = SimpleNamespace(
        model="m",
        log_level="INFO",
        openai_api_key="k",
        logfire_token=None,
        concurrency=1,
        prompt_dir="prompts",
        context_id="ctx",
        inspiration="insp",
        reasoning=None,
        features_per_role=5,
        mapping_batch_size=30,
        mapping_parallel_types=True,
        exhaustive_mapping=True,
        max_items_per_mapping=None,
        mapping_feature_batch_cap_tokens=95000,
        web_search=False,
        models=None,
    )
    args = argparse.Namespace(
        input_file=str(input_path),
        output_file=str(output_path),
        model=None,
        logfire_service=None,
        log_level=None,
        verbose=0,
        max_services=None,
        dry_run=False,
        progress=False,
        concurrency=None,
        resume=True,
        seed=None,
        roles_file="data/roles.json",
        mapping_batch_size=None,
        mapping_parallel_types=None,
        exhaustive_mapping=True,
        transcripts_dir=None,
        web_search=None,
        no_logs=False,
        search_model=None,
        strict=False,
    )

    monkeypatch.setattr(cli, "_RUN_META", None)
    asyncio.run(_cmd_generate_evolution(args, settings, None))

    assert processed == ["s2"]
    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert processed_path.read_text(encoding="utf-8").splitlines() == ["s1", "s2"]


def test_generate_evolution_rejects_invalid_concurrency(tmp_path, monkeypatch) -> None:
    """Invalid concurrency should raise an error rather than deadlock."""

    input_path = tmp_path / "services.jsonl"
    output_path = tmp_path / "out.jsonl"
    input_path.write_text("{}\n", encoding="utf-8")

    class DummyAgent:
        def __init__(self, model, instructions):
            pass

    async def fake_generate(
        self,
        service: ServiceInput,
        plateau_names=None,
        role_ids=None,
        transcripts_dir=None,
        meta: ServiceMeta | None = None,
    ) -> ServiceEvolution:
        assert meta is not None
        return ServiceEvolution(meta=meta, service=service, plateaus=[])

    monkeypatch.setattr("cli.Agent", DummyAgent)
    monkeypatch.setattr(
        "cli.PlateauGenerator.generate_service_evolution_async", fake_generate
    )
    monkeypatch.setattr("cli.configure_prompt_dir", lambda _path: None)
    monkeypatch.setattr("cli.load_evolution_prompt", lambda _ctx, _insp: "prompt")
    monkeypatch.setattr("cli.logfire.force_flush", lambda: None)

    settings = SimpleNamespace(
        model="m",
        log_level="INFO",
        openai_api_key="k",
        logfire_token=None,
        concurrency=1,
        prompt_dir="prompts",
        context_id="ctx",
        inspiration="insp",
        reasoning=None,
        features_per_role=5,
        mapping_batch_size=30,
        mapping_parallel_types=True,
        web_search=False,
        models=None,
    )
    args = argparse.Namespace(
        input_file=str(input_path),
        output_file=str(output_path),
        model=None,
        logfire_service=None,
        log_level=None,
        verbose=0,
        max_services=None,
        dry_run=False,
        progress=False,
        concurrency=0,
        resume=False,
        seed=None,
        roles_file="data/roles.json",
        mapping_batch_size=None,
        mapping_parallel_types=None,
        transcripts_dir=None,
        web_search=None,
        no_logs=False,
        search_model=None,
        strict=False,
    )

    monkeypatch.setattr(cli, "_RUN_META", None)
    with pytest.raises(ValueError, match="concurrency must be a positive integer"):
        asyncio.run(_cmd_generate_evolution(args, settings, None))


def test_cli_parses_mapping_options(tmp_path, monkeypatch) -> None:
    """generate-evolution should parse mapping options via the common parser."""

    called: dict[str, argparse.Namespace] = {}

    def fake_cmd(args: argparse.Namespace, _settings, transcripts_dir) -> None:
        called["args"] = args
        called["transcripts_dir"] = transcripts_dir

    monkeypatch.setattr(cli, "_cmd_generate_evolution", fake_cmd)
    monkeypatch.setattr(
        cli,
        "load_settings",
        lambda: SimpleNamespace(log_level="INFO", logfire_token=None),
    )
    monkeypatch.setattr("cli.logfire.force_flush", lambda: None)

    argv = [
        "prog",
        "generate-evolution",
        "--input-file",
        str(tmp_path / "services.jsonl"),
        "--output-file",
        str(tmp_path / "out.jsonl"),
        "--mapping-batch-size",
        "12",
        "--no-mapping-parallel-types",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    cli.main()

    assert called["args"].mapping_batch_size == 12
    assert called["args"].mapping_parallel_types is False


def test_generate_evolution_writes_transcripts(tmp_path, monkeypatch) -> None:
    """_cmd_generate_evolution writes per-service transcripts to disk."""

    input_path = tmp_path / "services.jsonl"
    output_path = tmp_path / "out.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "service_id": "svc-1",
                "name": "svc",
                "description": "desc",
                "jobs_to_be_done": [{"name": "job"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class DummyAgent:
        def __init__(self, model, instructions):
            self.model = model
            self.instructions = instructions

    async def fake_generate(
        self,
        service: ServiceInput,
        plateau_names=None,
        role_ids=None,
        transcripts_dir=None,
        meta: ServiceMeta | None = None,
    ) -> ServiceEvolution:
        assert transcripts_dir is not None
        assert meta is not None
        path = transcripts_dir / f"{service.service_id}.json"
        path.write_text("{}", encoding="utf-8")
        return ServiceEvolution(meta=meta, service=service, plateaus=[])

    monkeypatch.setattr("cli.Agent", DummyAgent)
    monkeypatch.setattr(
        "cli.PlateauGenerator.generate_service_evolution_async", fake_generate
    )
    monkeypatch.setattr("cli.configure_prompt_dir", lambda _path: None)
    monkeypatch.setattr("cli.load_evolution_prompt", lambda _ctx, _insp: "prompt")
    monkeypatch.setattr("cli.logfire.force_flush", lambda: None)

    settings = SimpleNamespace(
        model="test-model",
        log_level="INFO",
        openai_api_key="key",
        logfire_token=None,
        concurrency=2,
        prompt_dir="prompts",
        context_id="university",
        inspiration="general",
        reasoning=None,
        features_per_role=5,
        mapping_batch_size=30,
        mapping_parallel_types=True,
        exhaustive_mapping=True,
        max_items_per_mapping=None,
        mapping_feature_batch_cap_tokens=95000,
        models=None,
        web_search=False,
    )
    args = argparse.Namespace(
        input_file=str(input_path),
        output_file=str(output_path),
        model=None,
        logfire_service=None,
        log_level=None,
        verbose=0,
        max_services=None,
        dry_run=False,
        progress=False,
        concurrency=None,
        resume=False,
        seed=None,
        roles_file="data/roles.json",
        mapping_batch_size=None,
        mapping_parallel_types=None,
        exhaustive_mapping=True,
        transcripts_dir=None,
        web_search=None,
        no_logs=False,
        search_model=None,
        strict=False,
    )

    monkeypatch.setattr(cli, "_RUN_META", None)
    asyncio.run(_cmd_generate_evolution(args, settings, None))

    transcript = output_path.parent / "_transcripts" / "svc-1.json"
    assert transcript.exists()
