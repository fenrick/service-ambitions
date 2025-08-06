import sys
import types

# Stub external dependencies so tests can import `main` without installing
# heavy packages.
dotenv_stub = types.ModuleType("dotenv")
dotenv_stub.load_dotenv = lambda *args, **kwargs: None  # type: ignore[attr-defined]
sys.modules.setdefault("dotenv", dotenv_stub)

pydantic_stub = types.ModuleType("pydantic")
pydantic_stub.BaseModel = object  # type: ignore[attr-defined]
sys.modules.setdefault("pydantic", pydantic_stub)

langchain_chat_models = types.ModuleType("langchain.chat_models")
langchain_chat_models.init_chat_model = (  # type: ignore[attr-defined]
    lambda **kwargs: None
)
langchain_core_prompts = types.ModuleType("langchain_core.prompts")
langchain_core_prompts.ChatPromptTemplate = object  # type: ignore[attr-defined]
langchain_core_utils_json = types.ModuleType("langchain_core.utils.json")
langchain_core_utils_json.parse_json_markdown = (  # type: ignore[attr-defined]
    lambda x: x
)
langchain_core_language_models = types.ModuleType("langchain_core.language_models")
langchain_core_language_models_chat_models = types.ModuleType(
    "langchain_core.language_models.chat_models"
)
langchain_core_language_models_chat_models.BaseChatModel = object  # type: ignore[attr-defined]

sys.modules.setdefault("langchain", types.ModuleType("langchain"))
sys.modules.setdefault("langchain.chat_models", langchain_chat_models)
sys.modules.setdefault("langchain_core", types.ModuleType("langchain_core"))
sys.modules.setdefault("langchain_core.prompts", langchain_core_prompts)
sys.modules.setdefault(
    "langchain_core.language_models",
    langchain_core_language_models,
)
sys.modules.setdefault(
    "langchain_core.language_models.chat_models",
    langchain_core_language_models_chat_models,
)
sys.modules.setdefault("langchain_core.utils", types.ModuleType("langchain_core.utils"))
sys.modules.setdefault("langchain_core.utils.json", langchain_core_utils_json)
