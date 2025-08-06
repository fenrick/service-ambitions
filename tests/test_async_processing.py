import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main


class DummyPromptTemplate:
    """Minimal prompt template returning user input."""

    def __init__(self, *args, **kwargs):
        pass

    def invoke(self, inputs):
        return inputs["user_prompt"]


@pytest.mark.asyncio
async def test_process_service_async(monkeypatch):
    monkeypatch.setattr(main, "ChatPromptTemplate", DummyPromptTemplate)

    async def fake_ainvoke(prompt_message):
        return SimpleNamespace(content={"service": prompt_message})

    model = SimpleNamespace(ainvoke=fake_ainvoke)
    service = {"name": "alpha"}

    result = await main.process_service(service, model, "prompt")

    assert result == {"service": json.dumps(service)}
