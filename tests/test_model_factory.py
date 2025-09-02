# SPDX-License-Identifier: MIT
from models import StageModels
from models.factory import ModelFactory


def test_model_factory_resolves_models(monkeypatch) -> None:
    def fake_build_model(name, api_key, *, seed=None, reasoning=None, web_search=False):
        return f"built:{name}"

    monkeypatch.setattr("models.factory.build_model", fake_build_model)

    stage = StageModels(
        descriptions=None, features="cfg-feat", mapping=None, search=None
    )
    factory = ModelFactory("default", "key", stage_models=stage)

    assert factory.model_name("features") == "cfg-feat"
    assert factory.model_name("descriptions") == "default"
    assert factory.model_name("features", "cli") == "cli"
    assert factory.get("features") == "built:cfg-feat"
