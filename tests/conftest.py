import sys
import types

# Stub external dependencies so tests can import `main` without installing
# heavy packages.
dotenv_stub = types.ModuleType("dotenv")
dotenv_stub.load_dotenv = lambda *args, **kwargs: None
sys.modules.setdefault("dotenv", dotenv_stub)

langchain_chat_models = types.ModuleType("langchain.chat_models")
langchain_chat_models.init_chat_model = lambda **kwargs: None
langchain_core_prompts = types.ModuleType("langchain_core.prompts")
langchain_core_prompts.ChatPromptTemplate = object
langchain_core_utils_json = types.ModuleType("langchain_core.utils.json")
langchain_core_utils_json.parse_json_markdown = lambda x: x

sys.modules.setdefault("langchain", types.ModuleType("langchain"))
sys.modules.setdefault("langchain.chat_models", langchain_chat_models)
sys.modules.setdefault("langchain_core", types.ModuleType("langchain_core"))
sys.modules.setdefault("langchain_core.prompts", langchain_core_prompts)
sys.modules.setdefault("langchain_core.utils", types.ModuleType("langchain_core.utils"))
sys.modules.setdefault("langchain_core.utils.json", langchain_core_utils_json)
