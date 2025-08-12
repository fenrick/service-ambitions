"""Tests for the generate-evolution CLI subcommand."""

import argparse
import json
from types import SimpleNamespace

import pytest

from cli import _cmd_generate_evolution
from models import SCHEMA_VERSION, ServiceEvolution, ServiceInput


@pytest.mark.asyncio
async def test_generate_evolution_writes_results(tmp_path, monkeypatch) -> None:
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
        model_name: str,
        api_key: str,
        *,
        seed: int | None = None,
        reasoning=None,
        web_search=False,
    ) -> object:  # pragma: no cover - stub
        return object()

    class DummyAgent:  # pragma: no cover - simple stub
        def __init__(self, model: object, instructions: str) -> None:
            self.model = model
            self.instructions = instructions

    async def fake_generate(self, service: ServiceInput) -> ServiceEvolution:
        return ServiceEvolution(service=service, plateaus=[])

    monkeypatch.setattr("cli.build_model", fake_build_model)
    monkeypatch.setattr("cli.Agent", DummyAgent)
    monkeypatch.setattr(
        "cli.PlateauGenerator.generate_service_evolution", fake_generate
    )
    monkeypatch.setattr("cli.configure_prompt_dir", lambda _path: None)
    monkeypatch.setattr("cli.load_prompt", lambda _ctx, _insp: "prompt")
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
    )

    await _cmd_generate_evolution(args, settings)

    payload = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert payload["service"]["name"] == "svc"
    assert payload["service"]["service_id"] == "svc-1"
    assert payload["schema_version"] == SCHEMA_VERSION


@pytest.mark.asyncio
async def test_generate_evolution_uses_agent_model(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "services.jsonl"
    output_path = tmp_path / "out.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "service_id": "s1",
                "name": "svc",
                "description": "d",
                "customer_type": "retail",
                "jobs_to_be_done": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_build_model(
        model_name: str,
        api_key: str,
        *,
        seed: int | None = None,
        reasoning=None,
        web_search=False,
    ) -> object:
        captured["model_name"] = model_name
        captured["api_key"] = api_key
        captured["seed"] = seed
        captured["web_search"] = web_search
        return "model"

    class DummyAgent:
        def __init__(
            self, model: object, instructions: str
        ) -> None:  # pragma: no cover - simple init
            captured["agent_model"] = model
            captured["instructions"] = instructions

    async def fake_generate(self, service: ServiceInput) -> ServiceEvolution:
        return ServiceEvolution(service=service, plateaus=[])

    monkeypatch.setattr("cli.build_model", fake_build_model)
    monkeypatch.setattr("cli.Agent", DummyAgent)
    monkeypatch.setattr(
        "cli.PlateauGenerator.generate_service_evolution", fake_generate
    )
    monkeypatch.setattr("cli.configure_prompt_dir", lambda _path: None)
    monkeypatch.setattr("cli.load_prompt", lambda _ctx, _insp: "prompt")
    monkeypatch.setattr("cli.logfire.force_flush", lambda: None)

    settings = SimpleNamespace(
        model=None,
        log_level="INFO",
        openai_api_key="key",
        logfire_token=None,
        concurrency=1,
        prompt_dir="prompts",
        context_id="ctx",
        inspiration="insp",
        reasoning=None,
    )
    args = argparse.Namespace(
        input_file=str(input_path),
        output_file=str(output_path),
        model="special",  # override default
        logfire_service=None,
        log_level=None,
        verbose=0,
        max_services=None,
        dry_run=False,
        progress=False,
        concurrency=None,
        resume=False,
        seed=None,
    )

    await _cmd_generate_evolution(args, settings)

    assert captured["model_name"] == "special"
    assert captured["agent_model"] == "model"
    assert captured["web_search"] is False


@pytest.mark.asyncio
async def test_generate_evolution_respects_concurrency(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "services.jsonl"
    output_path = tmp_path / "out.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "service_id": "s1",
                "name": "svc",
                "description": "d",
                "customer_type": "retail",
                "jobs_to_be_done": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class DummyAgent:
        def __init__(
            self, model: object, instructions: str
        ) -> None:  # pragma: no cover - simple init
            self.model = model
            self.instructions = instructions

    def fake_build_model(
        model_name: str,
        api_key: str,
        *,
        seed: int | None = None,
        reasoning=None,
        web_search=False,
    ) -> object:  # pragma: no cover - stub
        return object()

    async def fake_generate(
        self, service: ServiceInput
    ) -> ServiceEvolution:  # pragma: no cover - stub
        return ServiceEvolution(service=service, plateaus=[])

    captured: dict[str, int] = {}

    class DummySemaphore:
        def __init__(self, value: int) -> None:
            captured["workers"] = value

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):  # pragma: no cover - no cleanup
            return False

    monkeypatch.setattr("cli.asyncio.Semaphore", lambda value: DummySemaphore(value))
    monkeypatch.setattr("cli.build_model", fake_build_model)
    monkeypatch.setattr("cli.Agent", DummyAgent)
    monkeypatch.setattr(
        "cli.PlateauGenerator.generate_service_evolution", fake_generate
    )
    monkeypatch.setattr("cli.configure_prompt_dir", lambda _path: None)
    monkeypatch.setattr("cli.load_prompt", lambda _ctx, _insp: "prompt")
    monkeypatch.setattr("cli.logfire.force_flush", lambda: None)

    settings = SimpleNamespace(
        model="m",
        log_level="INFO",
        openai_api_key="k",
        logfire_token=None,
        concurrency=3,
        prompt_dir="prompts",
        context_id="ctx",
        inspiration="insp",
        reasoning=None,
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
    )

    await _cmd_generate_evolution(args, settings)

    assert captured["workers"] == 3


@pytest.mark.asyncio
async def test_generate_evolution_dry_run(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "services.jsonl"
    output_path = tmp_path / "out.jsonl"
    input_path.write_text('{"service_id": "s1", "name": "svc"}\n', encoding="utf-8")

    def fake_build_model(
        model_name: str,
        api_key: str,
        *,
        seed: int | None = None,
        reasoning=None,
        web_search=False,
    ) -> object:  # pragma: no cover - stub
        return object()

    class DummyAgent:  # pragma: no cover - simple stub
        def __init__(self, model: object, instructions: str) -> None:
            self.model = model
            self.instructions = instructions

    called = {"ran": False}

    async def fake_generate(
        self, service: ServiceInput
    ) -> ServiceEvolution:  # pragma: no cover - stub
        called["ran"] = True
        return ServiceEvolution(service=service, plateaus=[])

    monkeypatch.setattr("cli.build_model", fake_build_model)
    monkeypatch.setattr("cli.Agent", DummyAgent)
    monkeypatch.setattr(
        "cli.PlateauGenerator.generate_service_evolution", fake_generate
    )
    monkeypatch.setattr("cli.configure_prompt_dir", lambda _path: None)
    monkeypatch.setattr("cli.load_prompt", lambda _ctx, _insp: "prompt")
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
    )

    await _cmd_generate_evolution(args, settings)

    assert not output_path.exists()
    assert not called["ran"]


@pytest.mark.asyncio
async def test_generate_evolution_resume(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "services.jsonl"
    output_path = tmp_path / "out.jsonl"
    input_path.write_text(
        '{"service_id": "s1", "name": "svc1", "description": "d", "jobs_to_be_done":'
        ' [{"name": "j"}]}\n{"service_id": "s2", "name": "svc2", "description": "d",'
        ' "jobs_to_be_done": [{"name": "j"}]}\n',
        encoding="utf-8",
    )
    output_path.write_text('{"service_id": "s1"}\n', encoding="utf-8")
    (tmp_path / "processed_ids.txt").write_text("s1\n", encoding="utf-8")

    def fake_build_model(
        model_name: str,
        api_key: str,
        *,
        seed: int | None = None,
        reasoning=None,
        web_search=False,
    ) -> object:
        return object()

    class DummyAgent:
        def __init__(
            self, model: object, instructions: str
        ) -> None:  # pragma: no cover
            self.model = model
            self.instructions = instructions

    processed: list[str] = []

    async def fake_generate(self, service: ServiceInput) -> ServiceEvolution:
        processed.append(service.service_id)
        return ServiceEvolution(service=service, plateaus=[])

    monkeypatch.setattr("cli.build_model", fake_build_model)
    monkeypatch.setattr("cli.Agent", DummyAgent)
    monkeypatch.setattr(
        "cli.PlateauGenerator.generate_service_evolution", fake_generate
    )
    monkeypatch.setattr("cli.configure_prompt_dir", lambda _path: None)
    monkeypatch.setattr("cli.load_prompt", lambda _ctx, _insp: "prompt")
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
    )

    await _cmd_generate_evolution(args, settings)

    assert processed == ["s2"]
    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert (tmp_path / "processed_ids.txt").read_text(
        encoding="utf-8"
    ).splitlines() == ["s1", "s2"]
