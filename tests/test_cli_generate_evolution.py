"""Tests for the generate-evolution CLI subcommand."""

import argparse
import asyncio
import json
from types import SimpleNamespace

from cli import _cmd_generate_evolution
from models import SCHEMA_VERSION, ServiceEvolution, ServiceInput


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

    def fake_build_model(
        model_name, api_key, *, seed=None, reasoning=None, web_search=False
    ):
        return object()

    class DummyAgent:
        def __init__(self, model, instructions):
            self.model = model
            self.instructions = instructions

    async def fake_generate(self, service: ServiceInput) -> ServiceEvolution:
        return ServiceEvolution(service=service, plateaus=[])

    monkeypatch.setattr("cli.build_model", fake_build_model)
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
    )

    asyncio.run(_cmd_generate_evolution(args, settings))

    payload = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert payload["service"]["name"] == "svc"
    assert payload["service"]["service_id"] == "svc-1"
    assert payload["schema_version"] == SCHEMA_VERSION


def test_generate_evolution_dry_run(tmp_path, monkeypatch) -> None:
    """Dry run should skip processing and not write output."""
    input_path = tmp_path / "services.jsonl"
    output_path = tmp_path / "out.jsonl"
    input_path.write_text("{}\n", encoding="utf-8")

    def fake_build_model(*args, **kwargs):
        return object()

    class DummyAgent:
        def __init__(self, model, instructions):
            pass

    called = {"ran": False}

    async def fake_generate(self, service: ServiceInput) -> ServiceEvolution:
        called["ran"] = True
        return ServiceEvolution(service=service, plateaus=[])

    monkeypatch.setattr("cli.build_model", fake_build_model)
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
    )

    asyncio.run(_cmd_generate_evolution(args, settings))

    assert not output_path.exists()
    assert not called["ran"]


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

    def fake_build_model(*args, **kwargs):
        return object()

    class DummyAgent:
        def __init__(self, model, instructions):
            pass

    processed: list[str] = []

    async def fake_generate(self, service: ServiceInput) -> ServiceEvolution:
        processed.append(service.service_id)
        return ServiceEvolution(service=service, plateaus=[])

    monkeypatch.setattr("cli.build_model", fake_build_model)
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
    )

    asyncio.run(_cmd_generate_evolution(args, settings))

    assert processed == ["s2"]
    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert processed_path.read_text(encoding="utf-8").splitlines() == ["s1", "s2"]
