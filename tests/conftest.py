import sys
import types

# Stub external dependencies so tests can import `main` without installing
# heavy packages.
dotenv_stub = types.ModuleType("dotenv")
dotenv_stub.load_dotenv = lambda *args, **kwargs: None  # type: ignore[attr-defined]
sys.modules.setdefault("dotenv", dotenv_stub)

langchain_core_prompts = types.ModuleType("langchain_core.prompts")
langchain_core_prompts.ChatPromptTemplate = object  # type: ignore[attr-defined]
langchain_core_utils_json = types.ModuleType("langchain_core.utils.json")
langchain_core_utils_json.parse_json_markdown = lambda x: x  # type: ignore[attr-defined]

langchain_openai = types.ModuleType("langchain_openai")
langchain_openai.ChatOpenAI = lambda **kwargs: None  # type: ignore[attr-defined]

sys.modules.setdefault("langchain", types.ModuleType("langchain"))
sys.modules.setdefault("langchain_core", types.ModuleType("langchain_core"))
sys.modules.setdefault("langchain_core.prompts", langchain_core_prompts)
sys.modules.setdefault("langchain_core.utils", types.ModuleType("langchain_core.utils"))
sys.modules.setdefault("langchain_core.utils.json", langchain_core_utils_json)
sys.modules.setdefault("langchain_openai", langchain_openai)
