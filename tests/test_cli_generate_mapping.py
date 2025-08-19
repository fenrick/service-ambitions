import argparse
import asyncio
import json
from types import SimpleNamespace

import cli
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


cli.ModelFactory = DummyFactory  # type: ignore[assignment]


class DummyAgent:
    def __init__(self, model, instructions):
        self.model = model
        self.instructions = instructions


cli.Agent = DummyAgent  # type: ignore[assignment]


async def _noop_init_embeddings() -> None:
    pass


cli.init_embeddings = _noop_init_embeddings


def test_generate_mapping_writes_remapped_features(tmp_path, monkeypatch) -> None:
    feature = PlateauFeature(
        feature_id="f1",
        name="feat",
        description="d",
        score=MaturityScore(level=3, label="Defined", justification="j"),
        customer_type="retail",
        mappings={"old": [Contribution(item="x", contribution=0.2)]},
    )
    plateau = PlateauResult(
        plateau=1,
        plateau_name="start",
        service_description="desc",
        features=[feature],
    )
    evo = ServiceEvolution(
        service=ServiceInput(
            service_id="s1",
            name="svc",
            description="desc",
            jobs_to_be_done=[{"name": "job"}],
        ),
        plateaus=[plateau],
    )
    input_path = tmp_path / "in.jsonl"
    input_path.write_text(evo.model_dump_json() + "\n", encoding="utf-8")
    output_path = tmp_path / "out.jsonl"

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
        for f in features:
            f.mappings["new"] = [Contribution(item="y", contribution=1.0)]
        return features

    monkeypatch.setattr(cli, "map_features_async", fake_map_features_async)
    monkeypatch.setattr(cli, "configure_prompt_dir", lambda _path: None)
    monkeypatch.setattr(cli, "load_evolution_prompt", lambda _c, _i: "p")

    settings = SimpleNamespace(
        model="test",
        log_level="INFO",
        openai_api_key="k",
        prompt_dir="p",
        context_id="ctx",
        inspiration="insp",
        mapping_batch_size=30,
        mapping_parallel_types=True,
        reasoning=None,
        models=None,
        web_search=False,
        logfire_token=None,
    )
    args = argparse.Namespace(
        input=str(input_path),
        output=str(output_path),
        mapping_batch_size=12,
        mapping_parallel_types=False,
        mapping_model=None,
        model=None,
        seed=None,
        web_search=None,
    )

    asyncio.run(cli._cmd_generate_mapping(args, settings))

    payload = json.loads(output_path.read_text(encoding="utf-8").strip())
    feat = payload["plateaus"][0]["features"][0]
    assert "old" not in feat["mappings"]
    assert feat["mappings"]["new"][0]["item"] == "y"
    assert called["batch_size"] == 12
    assert called["parallel_types"] is False
