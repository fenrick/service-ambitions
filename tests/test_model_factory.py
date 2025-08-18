from model_factory import ModelFactory
from models import StageModels


def test_model_factory_resolves_models(monkeypatch) -> None:
    def fake_build_model(name, api_key, *, seed=None, reasoning=None, web_search=False):
        return f"built:{name}"

    monkeypatch.setattr("model_factory.build_model", fake_build_model)

    stage = StageModels(features="cfg-feat")
    factory = ModelFactory("default", "key", stage_models=stage)

    assert factory.model_name("features") == "cfg-feat"
    assert factory.model_name("descriptions") == "default"
    assert factory.model_name("features", "cli") == "cli"
    assert factory.get("features") == "built:cfg-feat"
