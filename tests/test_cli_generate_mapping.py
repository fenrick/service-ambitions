"""Tests for the generate-mapping CLI subcommand."""

import argparse
import asyncio
import json
from types import SimpleNamespace

import cli
from cli import _cmd_generate_mapping
from models import (
    Contribution,
    MaturityScore,
    PlateauFeature,
    PlateauResult,
    ServiceEvolution,
    ServiceInput,
)


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


cli.ModelFactory = DummyFactory
cli.Agent = DummyAgent


def test_generate_mapping_updates_features(tmp_path, monkeypatch) -> None:
    """_cmd_generate_mapping should write mapped results to disk."""

    input_path = tmp_path / "evo.jsonl"
    output_path = tmp_path / "out.jsonl"

    evo = ServiceEvolution(
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
        batch_size,
        parallel_types,
    ):
        called["batch_size"] = batch_size
        called["parallel_types"] = parallel_types
        for feat in features:
            feat.mappings = {"cat": [Contribution(item="X", contribution=1.0)]}
        return list(features)

    monkeypatch.setattr(cli, "map_features_async", fake_map_features_async)

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
        input_file=str(input_path),
        output_file=str(output_path),
        model=None,
        mapping_model=None,
        mapping_batch_size=5,
        mapping_parallel_types=False,
        seed=None,
        web_search=None,
    )

    asyncio.run(_cmd_generate_mapping(args, settings))

    payload = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert payload["plateaus"][0]["features"][0]["mappings"] == {
        "cat": [{"item": "X", "contribution": 1.0}]
    }
    assert called["batch_size"] == 5
    assert called["parallel_types"] is False
