from core.utils.prompt_loader import PromptLoader


class Config:
    CLANG_LIB_FILE = "/usr/lib/llvm-14/lib/libclang.so"
    """clang库路径"""

    RUSTC_BIN = "rustc"
    """rustc可执行文件路径"""
    CARGO_BIN = "cargo"
    """cargo 可执行文件路径"""

    PROMPT_PATHS = ["core/prompts"]
    """Prompt模板路径"""

    LOG_LEVEL = "DEBUG"
    LOG_TYPE = "file"
    # LOG_TYPE = "console"
    LOG_DIR = "../../Output/logs"

    LLM_CONFIGS = [
        # General LLM
        {
            "provider": "openai",
            "model": "<your-model>",
            "base_url": "<your-provider-base-url>",
            "api_key": "<your-api-key>",
            "timeout": 30000,
            "temperature": 0.0,
            "max_tokens": 8192
        },
        # Reasoner
        {
            "provider": "openai",
            "model": "<your-model>",
            "base_url": "<your-provider-base-url>",
            "api_key": "<your-api-key>",
            "timeout": 30000,
            "temperature": 0.0,
            "max_tokens": 8192
        }
    ]

    """LLM配置"""
    RAG_CONFIG = {
        "base_url": "<your-embedding-model-base-url>",
        "api_key": "<your-embedding-model-api-key>",
        "model": "<your-embedding-model>",
        "knowledge_dir": "../chromadb"
    }
    """RAG配置, 目前仅支持通过 API 调用生成检索向量，不支持本地模型"""
    RAG_KNOWLEDGE_DIR = "../chromadb"

    DB_URL = "sqlite+aiosqlite:///rustify.db"


PromptLoader.from_paths(Config.PROMPT_PATHS)
