import asyncio
from types import SimpleNamespace

import pytest

from runtime.environment import RuntimeEnv
from utils import MappingLoader, PromptLoader


class DummyPromptLoader(PromptLoader):
    """Prompt loader stub tracking cache clears."""

    def __init__(self) -> None:
        self.cleared = False

    def load(self, name: str) -> str:  # pragma: no cover - simple stub
        return ""

    def clear_cache(self) -> None:
        self.cleared = True


class DummyMappingLoader(MappingLoader):
    """Mapping loader stub tracking cache clears."""

    def __init__(self) -> None:
        self.cleared = False

    def load(self, sets):  # pragma: no cover - simple stub
        return {}, ""

    def clear_cache(self) -> None:
        self.cleared = True


def test_reset_clears_run_meta_and_caches(tmp_path):
    RuntimeEnv.reset()
    RuntimeEnv.initialize(
        SimpleNamespace(prompt_dir=tmp_path, mapping_data_dir=tmp_path)
    )
    env = RuntimeEnv.instance()
    prompt = DummyPromptLoader()
    mapping = DummyMappingLoader()
    env.prompt_loader = prompt
    env.mapping_loader = mapping
    env.run_meta = SimpleNamespace()
    RuntimeEnv.reset()
    assert prompt.cleared
    assert mapping.cleared
    assert env.run_meta is None


@pytest.mark.asyncio()
async def test_loader_concurrency(tmp_path):
    RuntimeEnv.reset()
    RuntimeEnv.initialize(
        SimpleNamespace(prompt_dir=tmp_path, mapping_data_dir=tmp_path)
    )
    env = RuntimeEnv.instance()
    prompt = env.prompt_loader
    mapping = env.mapping_loader

    async def hit_prompt() -> None:
        await asyncio.sleep(0)
        assert env.prompt_loader is prompt

    async def hit_mapping() -> None:
        await asyncio.sleep(0)
        assert env.mapping_loader is mapping

    async with asyncio.TaskGroup() as tg:
        for _ in range(50):
            tg.create_task(hit_prompt())
            tg.create_task(hit_mapping())
