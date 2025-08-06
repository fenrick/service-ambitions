import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main


class DummyPromptTemplate:
    """Minimal prompt template that forwards to the next runnable."""

    def __init__(self, *args, **kwargs):
        pass

    def __or__(self, other):  # pragma: no cover - trivial forwarding
        return other


@pytest.mark.asyncio
async def test_process_service_async(monkeypatch):
    monkeypatch.setattr(main, "ChatPromptTemplate", DummyPromptTemplate)

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

    result = await main.process_service(service, model, "prompt")

    assert result == {"service": json.dumps(service)}
