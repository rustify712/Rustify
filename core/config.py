from core.utils.prompt_loader import PromptLoader


class Config:

    LIB_CLANG = "/usr/lib/llvm-14/lib/libclang.so"
    """clang库路径"""

    RUSTC_BIN = "rustc"
    """rustc可执行文件路径"""
    CARGO_BIN = "cargo"
    """cargo 可执行文件路径"""

    PROMPT_PATHS = ["core/prompts"]
    """Prompt模板路径"""

    RUST_PROJECTS_PATH = "../../Output"

    LOG_LEVEL = "DEBUG"
    LOG_TYPE = "file"
    # LOG_TYPE = "console"
    LOG_DIR = "../../Output/logs"

    LLM_CONFIG = {
        "provider": "openai",
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
        "api_key": "sk-71f422b50f524917a61fbc6f5243f174",
        "timeout": 30000,
        "temperature": 0.0,
        "max_tokens": 8192
    }
    LLM_CONFIGS = [
        {
            "provider": "openai",
            "model": "deepseek-chat",
            "base_url": "https://api.deepseek.com/beta",
            "api_key": "sk-71f422b50f524917a61fbc6f5243f174",
            "timeout": 30000,
            "temperature": 0.0,
            "max_tokens": 8192
        },
    ]
    """LLM配置"""
    RAG_CONFIG = {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": "sk-a77dc4830e3d441db099f72dfaf7c484",
        "model": "text-embedding-v3",
    }
    """RAG配置, 目前仅支持通过 API 调用生成检索向量，不支持本地模型"""
    RAG_KNOWLEDGE_DIR = "../chromadb"

    DB_URL = "sqlite+aiosqlite:///transfactor.db"


PromptLoader.from_paths(Config.PROMPT_PATHS)