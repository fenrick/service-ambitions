import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import service_ambitions.generator as generator


class DummyPromptTemplate:
    """Minimal prompt template that forwards to the next runnable."""

    def __init__(self, *args, **kwargs):
        pass

    def __or__(self, other):  # pragma: no cover - trivial forwarding
        return other


def test_process_service_async(monkeypatch):
    monkeypatch.setattr(generator, "ChatPromptTemplate", DummyPromptTemplate)

    class DummyModel:
        def with_structured_output(self, _):  # pragma: no cover - simple stub
            class DummyChain:
                def invoke(self, inputs):
                    return SimpleNamespace(
                        model_dump=lambda: {"service": inputs["user_prompt"]}
                    )

            return DummyChain()

    model = DummyModel()
    service = {"name": "alpha"}

    gen = generator.ServiceAmbitionGenerator(model)
    result = asyncio.run(gen.process_service(service, "prompt"))

    assert result == {"service": json.dumps(service)}
