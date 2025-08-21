"""Tests for the generate-mapping CLI subcommand."""

import argparse
import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, cast

import pytest

import cli
import mapping
from backpressure import RollingMetrics
from cli import _cmd_generate_mapping
from conversation import ConversationSession
from models import (
    Contribution,
    MappingFeature,
    MappingResponse,
    MappingTypeConfig,
    MaturityScore,
    PlateauFeature,
    PlateauResult,
    ServiceEvolution,
    ServiceInput,
    ServiceMeta,
)
from stage_metrics import record_stage_metrics


class DummyFactory:
    def __init__(self, *a, **k):
        pass

    def model_name(self, stage, override=None):
        return "m"

    def get(self, stage, override=None):
        return object()


class DummyAgent:
    def __init__(self, model, instructions):
        self.model = model
        self.instructions = instructions


cli.ModelFactory = DummyFactory  # type: ignore[assignment, misc]
cli.Agent = DummyAgent  # type: ignore[assignment, misc]


def test_generate_mapping_updates_features(tmp_path, monkeypatch) -> None:
    """_cmd_generate_mapping should write mapped results to disk."""

    input_path = tmp_path / "evo.jsonl"
    output_path = tmp_path / "out.jsonl"

    meta = ServiceMeta(
        run_id="run",
        seed=None,
        models={},
        web_search=False,
        mapping_types=[],
        created=datetime.now(timezone.utc),
    )
    evo = ServiceEvolution(
        meta=meta,
        service=ServiceInput(
            service_id="svc-1",
            name="svc",
            description="d",
            customer_type="retail",
            jobs_to_be_done=[{"name": "job"}],
        ),
        plateaus=[
            PlateauResult(
                plateau=1,
                plateau_name="p1",
                service_description="desc",
                features=[
                    PlateauFeature(
                        feature_id="FEAT-1-learners-test",
                        name="feat",
                        description="fd",
                        score=MaturityScore(
                            level=1, label="Initial", justification="j"
                        ),
                        customer_type="learners",
                        mappings={"cat": []},
                    )
                ],
            )
        ],
    )
    input_path.write_text(evo.model_dump_json() + "\n", encoding="utf-8")

    called: dict[str, object] = {}

    async def fake_map_features_async(
        session,
        features,
        mapping_types=None,
        *,
        strict,
        batch_size,
        parallel_types,
    ):
        called["batch_size"] = batch_size
        called["parallel_types"] = parallel_types
        for feat in features:
            feat.mappings = {"cat": [Contribution(item="X", contribution=1.0)]}
        return list(features)

    monkeypatch.setattr(cli, "map_features_async", fake_map_features_async)

    init_called = {"ran": False}

    async def fake_init_embeddings() -> None:
        init_called["ran"] = True

    monkeypatch.setattr(cli, "init_embeddings", fake_init_embeddings)

    settings = SimpleNamespace(
        model="cfg",
        openai_api_key="key",
        mapping_batch_size=30,
        mapping_parallel_types=True,
        reasoning=None,
        models=None,
        web_search=False,
    )
    args = argparse.Namespace(
        input=str(input_path),
        output=str(output_path),
        model=None,
        mapping_model=None,
        mapping_batch_size=5,
        mapping_parallel_types=False,
        seed=None,
        web_search=None,
        strict=False,
    )

    asyncio.run(_cmd_generate_mapping(args, settings))

    payload = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert payload["plateaus"][0]["features"][0]["mappings"] == {
        "cat": [{"item": "X", "contribution": 1.0}]
    }
    assert called["batch_size"] == 5
    assert called["parallel_types"] is False
    assert init_called["ran"] is True


def test_generate_mapping_logs_stage_totals(tmp_path, monkeypatch) -> None:
    """_cmd_generate_mapping should log stage totals after processing."""

    input_path = tmp_path / "evo.jsonl"
    output_path = tmp_path / "out.jsonl"

    meta = ServiceMeta(
        run_id="run",
        seed=None,
        models={},
        web_search=False,
        mapping_types=[],
        created=datetime.now(timezone.utc),
    )
    evo = ServiceEvolution(
        meta=meta,
        service=ServiceInput(
            service_id="svc-1",
            name="svc",
            description="d",
            customer_type="retail",
            jobs_to_be_done=[{"name": "job"}],
        ),
        plateaus=[
            PlateauResult(
                plateau=1,
                plateau_name="p1",
                service_description="desc",
                features=[
                    PlateauFeature(
                        feature_id="FEAT-1-learners-test",
                        name="feat",
                        description="fd",
                        score=MaturityScore(
                            level=1, label="Initial", justification="j"
                        ),
                        customer_type="learners",
                        mappings={"cat": []},
                    )
                ],
            )
        ],
    )
    input_path.write_text(evo.model_dump_json() + "\n", encoding="utf-8")

    async def fake_map_features_async(
        session,
        features,
        mapping_types=None,
        *,
        strict,
        batch_size,
        parallel_types,
    ):
        record_stage_metrics(
            "mapping",
            tokens=20,
            cost=0.0,
            duration=4.0,
            is_429=True,
            prompt_tokens=10,
        )
        return list(features)

    monkeypatch.setattr(cli, "map_features_async", fake_map_features_async)

    async def fake_init_embeddings() -> None:
        return None

    monkeypatch.setattr(cli, "init_embeddings", fake_init_embeddings)

    logged: list[dict[str, float]] = []

    def fake_logfire_info(msg: str, **kwargs) -> None:
        if msg == "Stage totals":
            logged.append(kwargs)

    monkeypatch.setattr(cli.logfire, "info", fake_logfire_info)

    settings = SimpleNamespace(
        model="cfg",
        openai_api_key="key",
        mapping_batch_size=30,
        mapping_parallel_types=True,
        reasoning=None,
        models=None,
        web_search=False,
    )
    args = argparse.Namespace(
        input=str(input_path),
        output=str(output_path),
        model=None,
        mapping_model=None,
        mapping_batch_size=5,
        mapping_parallel_types=False,
        seed=None,
        web_search=None,
        strict=False,
    )

    asyncio.run(_cmd_generate_mapping(args, settings))

    assert logged and logged[0]["tokens_per_sec"] == pytest.approx(5.0)
    assert logged[0]["avg_latency"] == pytest.approx(4.0)
    assert logged[0]["rate_429"] == pytest.approx(1.0)


def test_request_mapping_retries(monkeypatch) -> None:
    """_request_mapping should retry transient errors and pass hooks."""

    async def fake_preselect_items(*_a, **_k):
        return None

    monkeypatch.setattr(mapping, "_preselect_items", fake_preselect_items)

    def fake_build_mapping_prompt(*_a, **_k) -> str:
        return "prompt"

    monkeypatch.setattr(mapping, "_build_mapping_prompt", fake_build_mapping_prompt)

    recorded: dict[str, object] = {}

    async def spy_with_retry(
        coro_factory,
        *,
        request_timeout,
        attempts,
        base=0.5,
        cap=8.0,
        on_retry_after=None,
        metrics=None,
    ):
        recorded["request_timeout"] = request_timeout
        recorded["attempts"] = attempts
        recorded["on_retry_after"] = on_retry_after
        recorded["metrics"] = metrics
        try:
            return await coro_factory()
        except Exception:
            return await coro_factory()

    monkeypatch.setattr(mapping, "_with_retry", spy_with_retry)

    class DummyLimiter:
        def __init__(self) -> None:
            self.calls: list[float] = []

        def throttle(self, delay: float) -> None:
            self.calls.append(delay)

    limiter = DummyLimiter()
    metrics = RollingMetrics(window=1)

    class DummySession:
        def __init__(self) -> None:
            self.calls = 0
            self._limiter = limiter
            self._metrics = metrics
            self.request_timeout = 0.1
            self.retries = 2
            self.retry_base_delay = 0.01

        def derive(self):
            return self

        async def ask_async(self, prompt: str, output_type):
            self.calls += 1
            if self.calls == 1:
                raise asyncio.TimeoutError()
            return MappingResponse(
                features=[
                    MappingFeature(
                        feature_id="F",
                        mappings={"cat": [Contribution(item="A", contribution=1.0)]},
                    )
                ]
            )

    feature = PlateauFeature(
        feature_id="F",
        name="n",
        description="d",
        score=MaturityScore(level=1, label="Initial", justification="j"),
        customer_type="c",
        mappings={"cat": []},
    )
    cfg = MappingTypeConfig(dataset="ds", label="Dataset")

    session = DummySession()
    result = asyncio.run(
        mapping._request_mapping(
            cast(ConversationSession, session), [feature], 0, "cat", cfg
        )
    )

    assert session.calls == 2
    hook = cast(Any, recorded["on_retry_after"])
    assert hook is not None and hook.__self__ is limiter
    assert recorded["metrics"] is metrics
    assert result[0] == 0
    assert isinstance(result[4], MappingResponse)
