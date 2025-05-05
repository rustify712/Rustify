from core.utils.prompt_loader import PromptLoader


class Config:
    CLANG_LIB_FILE = "/usr/lib/llvm-14/lib/libclang.so."
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
        # 火山引擎
        {
            "provider": "openai",
            "model": "ep-20250220154917-m5tv5",  # deepseek-v3, ep-20250213154706-429dr, ep-20250218120542-fdhwp
            "base_url": "https://ark.cn-beijing.volces.com/api/v3",
            "api_key": "cb7cb751-3de9-4e2b-89bb-bbd7a303f193",
            "timeout": 30000,
            "temperature": 0.0,
            "max_tokens": 8192
        },
        # 火山引擎
        {
            "provider": "openai",
            "model": "ep-20250220155009-8ckjl",  # deepseek-reasoner, ep-20250214201319-7gv7f, ep-20250218120505-tlgs8
            "base_url": "https://ark.cn-beijing.volces.com/api/v3",
            "api_key": "cb7cb751-3de9-4e2b-89bb-bbd7a303f193",
            "timeout": 30000,
            "temperature": 0.6,
            "max_tokens": 8192
        },
        # 腾讯云
        {
            "provider": "openai",
            "model": "deepseek-v3",
            "base_url": "https://api.lkeap.cloud.tencent.com/v1",
            "api_key": "sk-K9lh3Ed3izLmUNzOMW7puJDH6JgoJ8QW9VbCLvSMkiNZINwL",
            "timeout": 30000,
            "temperature": 0.0,
            "max_tokens": 8192
        },
        # DeepSeek
        {
            "provider": "openai",
            "model": "deepseek-chat",
            "base_url": "https://api.deepseek.com/beta",
            "api_key": "sk-71f422b50f524917a61fbc6f5243f174",
            "timeout": 30000,
            "temperature": 0.0,
            "max_tokens": 8192
        },
        # DeepSeek R1
        {
            "provider": "openai",
            "model": "deepseek-reasoner",
            "base_url": "https://api.deepseek.com",
            "api_key": "sk-71f422b50f524917a61fbc6f5243f174",
            "timeout": 30000,
            "temperature": 0.6,
            "max_tokens": 8192
        }
    ]

    """LLM配置"""
    RAG_CONFIG = {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": "sk-a77dc4830e3d441db099f72dfaf7c484",
        "model": "text-embedding-v3",
        "knowledge_dir": "../chromadb"
    }
    """RAG配置, 目前仅支持通过 API 调用生成检索向量，不支持本地模型"""
    RAG_KNOWLEDGE_DIR = "../chromadb"

    DB_URL = "sqlite+aiosqlite:///transfactor.db"


PromptLoader.from_paths(Config.PROMPT_PATHS)
