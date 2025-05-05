from typing import TypedDict

from core.state.state_manager import FileLockManager, StateManager


class LLMConfig(TypedDict):
    """LLM 配置

    Notes:
        provider 目前仅支持 OpenAI

    Attributes:
        provider (str): 模型提供商
        base_url (str): 模型 API 地址
        api_key (str): 模型 API 密钥
        model (str): 模型名称
        timeout (int): 请求超时时间
        temperature (float): 生成温度
        max_tokens (int): 最大生成长度
    """
    provider: str
    base_url: str
    api_key: str
    model: str
    timeout: int
    temperature: float
    max_tokens: int

class RAGConfig(TypedDict):
    """RAG 配置

    Notes:
        目前仅支持通过 API 调用生成检索向量，不支持本地模型

    Attributes:
        base_url (str): 模型 API 地址
        api_key (str): 模型 API 密钥
        model (str): 模型名称
        knowledge_dir (str): 知识库路径
    """
    base_url: str
    api_key: str
    model: str
    knowledge_dir: str

class DBConfig(TypedDict):
    """数据库配置

    Attributes:
        url (str): 数据库连接地址
        debug_sql (bool): 是否开启 SQL 调试模式
    """
    url: str
    debug_sql: bool

class Translator:
    def __init__(
        self,
        *,
        prompt_folders: list[str],
        rustc_bin: str,
        cargo_bin: str,
        llm_config: LLMConfig,
        rag_config: RAGConfig,
        reasoner_config: LLMConfig = None,
        state_file: str = None,
    ):
        """
        Args:
            prompt_folders (list[str]): Prompt 模板路径
            rustc_bin (str): rustc 可执行文件路径
            cargo_bin (str): cargo 可执行文件
            llm_config (LLMConfig): LLM 配置
            rag_config (RAGConfig): RAG 配置
            reasoner_config (LLMConfig): 推理器配置
        """
        # 配置项
        self.prompt_folders = prompt_folders
        self.rustc_bin = rustc_bin
        self.cargo_bin = cargo_bin
        self.llm_config = llm_config
        self.rag_config = rag_config
        self.reasoner_config = reasoner_config
        # 知识库
        self.kb_client, self.embedding_function = self.setup_knowledge(rag_config)
        # 文件锁
        self.file_lock_manager = FileLockManager()

        # 全局状态
        self.state_manager = StateManager(
            state_file
        )

    def setup_prompts(self, prompt_folders: list[str]):
        """设置 Prompt 模板"""
        from core.utils.prompt_loader import PromptLoader

        PromptLoader.from_paths(prompt_folders)

    def setup_knowledge(self, rag_config: RAGConfig):
        """初始化知识库
        """
        import chromadb
        from chromadb.utils.embedding_functions.openai_embedding_function import OpenAIEmbeddingFunction

        chroma_client = chromadb.PersistentClient(
            path=rag_config["knowledge_dir"],
        )
        embedding_function = OpenAIEmbeddingFunction(
            model_name=rag_config["model"],
            api_key=rag_config["api_key"],
            api_base=rag_config["base_url"],
        )
        return chroma_client, embedding_function
