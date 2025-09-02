from types import SimpleNamespace

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


def test_reset_clears_run_meta_and_caches():
    RuntimeEnv.reset()
    RuntimeEnv.initialize(SimpleNamespace())
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
