import threading
import uuid
import math
import itertools
from concurrent.futures import ThreadPoolExecutor
from typing import Type

import chromadb
from chromadb.utils.embedding_functions.openai_embedding_function import OpenAIEmbeddingFunction
from pydantic import BaseModel, Field
from sqlalchemy import inspect, select

from core.config import Config
from core.agents.base import BaseAgent
from core.db.models.base import BaseTable
from core.db.models.translation import TranslationTable
from core.schema.translation import TranslationTask
from core.state.state_manager import ModuleTranslation, StateManager
from core.db.session import SessionManager
from core.utils.decorator import singleton
from core.utils.prompt_loader import PromptLoader


def insert_data(table: Type[BaseTable], obj):
    with SessionManager() as session:
        inspector = inspect(session.bind)
        if table.__tablename__ not in inspector.get_table_names():
            table.metadata.create_all(session.bind)
        session.add(obj)
        session.commit()
        return obj


def batch_insert_data(table: Type[BaseTable], objs) -> list:
    with SessionManager() as session:
        inspector = inspect(session.bind)
        if table.__tablename__ not in inspector.get_table_names():
            table.metadata.create_all(session.bind)
        session.add_all(objs)
        session.commit()
        return objs


class TranslationDescription(BaseModel):
    source_description: str = Field(description="转译前的原始代码描述")
    target_description: str = Field(description="转译后的代码描述")
    experience: str = Field(description="转译经验")
    queries: list[str] = Field(description="查询问题列表")


def l2_normalize(distance, lambda_=1.0):
    return math.exp(-lambda_ * distance)


@singleton
class TranspileMemory(BaseAgent):
    ROLE = "transpile_memory"
    DESCRIPTION = "Transpile memory"

    def __init__(
            self,
            llm_config: dict,
            state_manager: "StateManager",
            **kwargs
    ):
        super().__init__(llm_config, **kwargs, name=self.ROLE)
        self.state_manager = state_manager
        chroma_client = chromadb.PersistentClient(
            path=Config.RAG_KNOWLEDGE_DIR
        )
        embedding_function = OpenAIEmbeddingFunction(
            model_name=Config.RAG_CONFIG["model"],
            api_key=Config.RAG_CONFIG["api_key"],
            api_base=Config.RAG_CONFIG["base_url"],
        )
        # 转译记录集合
        self.translation_collection = chroma_client.get_or_create_collection(
            "translation",
            embedding_function=embedding_function,
            metadata={"hnsw:space": "l2"}
        )
        # 修复记录集合
        self.fixing_collection = chroma_client.get_or_create_collection(
            "fixing",
            embedding_function=embedding_function,
            metadata={"hnsw:space": "l2"}
        )

        self.executor = ThreadPoolExecutor(max_workers=8)
        self.futures = []
        self.futures_lock = threading.Lock()

    def _on_future_done(self, future):
        # 如果任务在执行中抛出异常，这里可以获取到
        exc = future.exception()
        if exc is not None:
            # 处理异常情况
            self.logger.error(f"Future failed with exception: {exc}")
            # 在这里执行你希望的失败处理逻辑
        else:
            # 任务成功完成时的逻辑
            with self.futures_lock:
                if future in self.futures:
                    self.futures.remove(future)
            self.logger.debug("Future completed successfully.")

    def _store_translation_memory(
            self,
            module_translation: ModuleTranslation,
            translation_task: TranslationTask,
            messages: list[dict]
    ):
        """存储转译记忆

        Args:
            module_translation: 转译模块
            translation_task: 转译任务
            messages: 消息列表
        """
        # TODO：查询数据表，检查相同项目中，是否已经存在相同的转译记录，若存在则不再存储（后期考虑全部存储标记，人工筛查）
        translation_source_text = "\n".join([
            node.text
            for node in translation_task.source.nodes
        ])
        with SessionManager() as session:
            inspector = inspect(session.bind)
            if TranslationTable.__tablename__ not in inspector.get_table_names():
                TranslationTable.metadata.create_all(session.bind)
            results = session.execute(
                select(TranslationTable)
                .where(TranslationTable.source_name == translation_task.source.name)
                .where(TranslationTable.source_text == translation_source_text)
                .where(TranslationTable.source_module == module_translation.module_name)
                .where(TranslationTable.source_project == self.state_manager.source_project.path)
            )
            if results.scalars().all():
                self.logger.info(
                    f"Translation record [{translation_task.source.name} of {self.state_manager.source_project.path}] already exists, skip storing")
                return
        describe_translation_prompt = PromptLoader.get_prompt(
            f"{self.ROLE}/describe_translation.prompt",
            source_lang="C",
            target_lang="Rust",
            translation_task=translation_task
        )
        self.logger.debug(f"Describe translation prompt: {describe_translation_prompt}")
        describe_translation_messages = [
            {"role": "user", "content": describe_translation_prompt}
        ]
        try_count = 0
        translation_description_obj = None
        while try_count < 3:
            describe_translation_response = self.call_llm(messages=describe_translation_messages,
                                                          json_format=TranslationDescription)
            describe_translation_messages.append(
                {"role": "assistant", "content": describe_translation_response.choices[0].message.content}
            )
            translation_description_obj = describe_translation_response.format_object
            if translation_description_obj is None:
                describe_translation_messages.append(
                    {"role": "user", "content": "JSON 解析错误，请确保遵循 JSON 格式"}
                )
                try_count += 1
                continue
            else:
                self.logger.debug(f"Translation description: {translation_description_obj}")
                break
        if translation_description_obj is None:
            self.logger.error("Failed to parse translation description")
            return
        translation_record = TranslationTable(
            source_lang="C",
            source_name=translation_task.source.name,
            source_text=translation_source_text,
            source_description=translation_description_obj.source_description,
            source_module=module_translation.module_name,
            source_project=self.state_manager.source_project.path,
            target_lang="Rust",
            target_name=translation_task.target.name,
            target_text=translation_task.target.text,
            target_description=translation_description_obj.target_description,
            description=translation_description_obj.experience,
            chat_messages=messages
        )
        translation_record = insert_data(TranslationTable, translation_record)
        queries = translation_description_obj.queries
        ids = [
            str(uuid.uuid4())
            for _ in range(len(queries))
        ]
        self.translation_collection.add(
            ids=ids,
            documents=queries,
            metadatas=[
                {"table_name": TranslationTable.__tablename__, "record_id": translation_record.id}
                for _ in range(len(queries))
            ]
        )
        self.logger.info(
            f"Translation record [{translation_task.source.name} of {self.state_manager.source_project.path}] stored successfully")

    def store_translation_memory(
            self,
            module_translation: ModuleTranslation,
            translation_task: TranslationTask,
            messages: list[dict]
    ):
        """存储转译记忆

        Args:
            module_translation: 转译模块
            translation_task: 转译任务
            messages: 消息列表
        """
        ...
        # future = self.executor.submit(
        #     self._store_translation_memory,
        #     module_translation,
        #     translation_task,
        #     messages
        # )
        # future.add_done_callback(self._on_future_done)
        # with self.futures_lock:
        #     self.futures.append(future)

    def query_translation_memory(self, queries: list[str], threshold=0.7):
        """查询转译记忆"""
        if len(queries) == 0:
            return []
        results = self.translation_collection.query(
            query_texts=queries,
            n_results=10
        )
        ids = list(itertools.chain.from_iterable(results["ids"]))
        documents = list(itertools.chain.from_iterable(results["documents"]))
        metadatas = list(itertools.chain.from_iterable(results["metadatas"]))
        distances = list(map(l2_normalize, itertools.chain.from_iterable(results["distances"])))
        if len(ids) == 0:
            return []

        query_results = []
        for id_, doc, meta, dist in zip(ids, documents, metadatas, distances):
            if dist < threshold:
                continue
            query_results.append((id_, doc, meta, dist))
        record_ids = []



    def wait_all_tasks(self):
        ...
        # wait(self.futures)
        # self.futures.clear()

    def retrieve_docs(self, query: str, ):
        """检索文档"""
        ...
