import asyncio
import os
import time
import traceback
from collections import defaultdict
from typing import Optional

from pydantic import BaseModel, Field

from core.agents.base import AgentResponse, AgentResponseStatus, BaseAgent
from core.agents.code_monkey import CodeMonkey
from core.schema.response import TechLeaderResponseType
from core.schema.translation import TranslationTask, TranslationTaskStatus
from core.state.state_manager import ModuleTranslation, ModuleTranslationStatus
from core.translator import Translator
from core.utils.prompt_loader import PromptLoader

class CreateModuleFile(BaseModel):
    filepath: str = Field(description="文件路径，相对于模块目录")
    content: str = Field(description="文件内容")

class CreateModuleResponse(BaseModel):
    files: list[CreateModuleFile] = Field(description="模块文件列表")

class TechLeader(BaseAgent):
    """团队领袖智能体
    负责管理某个模块的转译，测试，基准测试等流程

    ```mermaid
    graph TD
        A[创建模块转译任务] --> B[分配转译任务]
        B --> C[转译任务完成]
    ```

    """

    ROLE = "tech_leader"
    DESCRIPTION = "A powerful and efficient AI assistant responsible for managing the translation process."

    def __init__(
            self,
            translator: Translator,
            **kwargs
    ):
        super().__init__(translator.llm_config, **kwargs)
        self.translator = translator
        # 当前模块转译任务
        self.module_translation_id: Optional[str] = None
        # 任务依赖：task_id -> int
        self.task_dependency_counter = defaultdict(int)
        # 任务依赖：task_id -> children_task_ids
        self.task_dependency_map = defaultdict(set)
        # 转译任务队列
        self.translation_task_queue = asyncio.Queue(maxsize=5)
        # 并发限制
        self.concurrent_semaphore = asyncio.Semaphore(3)

    @property
    def module_translation(self):
        module_translation = self.translator.state_manager.get_module_translation_by_id(self.module_translation_id)
        if module_translation is None:
            raise ValueError(f"module translation not found: {self.module_translation_id}")
        return module_translation

    def run(self, agent_response: AgentResponse) -> AgentResponse:
        if agent_response.status == AgentResponseStatus.DONE:
            if agent_response.type == TechLeaderResponseType.PREPARE_MODULE_TRANSLATION_TASKS:
                # 分配转译任务
                return asyncio.run(self.assign_translation_tasks(agent_response))
            elif agent_response.type == TechLeaderResponseType.MODULE_TRANSLATION_COMPLETION_DONE:
                # TODO test and benchmark
                return AgentResponse.done(self, TechLeaderResponseType.MODULE_DONE)
            else:
                self.logger.error(f"unknown response: {agent_response}")
        else:
            # TODO: exception handling
            self.logger.error(f"error occurred in run: {agent_response}")
            return agent_response

    def start(self, module_translation: ModuleTranslation) -> AgentResponse:
        self.logger.info(f"start Tech Leader [{self.name}] for module translation: [{module_translation.name}]")
        agent_response = asyncio.run(self.prepare_module_translation_tasks(module_translation))
        last_response = agent_response
        while True:
            agent_response = self.run(agent_response)
            if agent_response is None:
                self.logger.error(f"agent response is None, break. last_response: {last_response}")
                return AgentResponse.error(self, TechLeaderResponseType.MODULE_DONE, error={
                    "message": f"agent response is None, last response: {last_response}"
                })
            if agent_response.type == TechLeaderResponseType.MODULE_DONE:
                return agent_response
            last_response = agent_response

    async def prepare_module_translation_tasks(self, module_translation: ModuleTranslation) -> AgentResponse:
        """预处理转译任务
        """
        self.logger.info(f"preparing module translation: [{module_translation.name}]")
        self.module_translation_id = module_translation.id
        if not os.path.exists(module_translation.path):
            # 创建 Rust 模块
            project_type = "bin" if "main" in [
                os.path.splitext(os.path.basename(file))[0]
                for file in module_translation.related_files
            ] else "lib"
            try:
                self.logger.info(f"creating rust project: {self.module_translation.path}")
                project_files = self.translator.state_manager.state.source_project.list_files(
                    show_summary=True,
                    ignore_func=lambda filepath: filepath not in self.module_translation.related_files
                )
                create_module_prompt = PromptLoader.get_prompt(
                    f"{self.ROLE}/create_module.prompt",
                    module_name=module_translation.name,
                    module_description=module_translation.description,
                    project_files=project_files,
                    show_file_summary=True,
                    crate_type=project_type
                )
                self.logger.debug(f"create module prompt: {create_module_prompt}")
                create_module_messages = [
                    {"role": "user", "content": create_module_prompt}
                ]
                max_try_count = 3
                create_module = None
                while max_try_count > 0:
                    create_module_response = self.call_llm(messages=[
                        {"role": "user", "content": create_module_prompt}
                    ], json_format=CreateModuleResponse)
                    create_module_response_content = create_module_response.choices[0].message.content
                    create_module_messages.append({"role": "assistant", "content": create_module_response_content})
                    create_module = create_module_response.format_object
                    if create_module:
                        self.logger.debug(f"create module response: {create_module_response_content}")
                        break
                    else:
                        create_module_messages.append(
                            {"role": "user", "content": "解析转译结果失败，请严格遵守转译结果的 JSON 格式。"}
                        )
                        max_try_count -= 1
                        continue
                if create_module:
                    self.logger.info(f"creating module for module translation {self.module_translation.name}")
                    for file in create_module.files:
                        filepath = os.path.join(module_translation.path, file.filepath)
                        os.makedirs(os.path.dirname(filepath), exist_ok=True)
                        with open(filepath, "w", encoding="utf-8") as f:
                            f.write(file.content)
                    await self.translator.state_manager.add_module_translation_related_rust_files(
                        module_translation.id, [
                            file.filepath
                            for file in create_module.files
                        ]
                    )
                else:
                    raise ValueError("create module failed")
                # update_cargo_toml(
                #     project_path=self.module_translation.path,
                #     section="package",
                #     key="description",
                #     value=self.module_translation.description
                # )
            except Exception as e:
                self.logger.error(f"cargo new project failed: {e}")
                raise e
            if self.translator.state_manager.file_system:
                self.translator.state_manager.file_system.load(module_translation.path)
        # 构建任务依赖关系
        for translation_task in self.module_translation.translation_tasks:
            for prerequisite_task_id in translation_task.prerequisites:
                self.task_dependency_map[prerequisite_task_id].add(translation_task.id)
            self.task_dependency_counter[translation_task.id] = len(translation_task.prerequisites)

        # 根据当前模块的状态采取接下来的操作
        if self.module_translation.status == ModuleTranslationStatus.CREATED:
            return AgentResponse.done(self, TechLeaderResponseType.PREPARE_MODULE_TRANSLATION_TASKS)
        elif self.module_translation.status == ModuleTranslationStatus.TRANSPILE:
            return AgentResponse.done(self, TechLeaderResponseType.PREPARE_MODULE_TRANSLATION_TASKS)
        elif self.module_translation.status == ModuleTranslationStatus.TEST:
            return AgentResponse.done(self, TechLeaderResponseType.MODULE_TRANSLATION_DONE)
        elif self.module_translation.status == ModuleTranslationStatus.BENCHMARK:
            return AgentResponse.done(self, TechLeaderResponseType.MODULE_TEST_DONE)
        else:
            return AgentResponse.error(self, TechLeaderResponseType.PREPARE_MODULE_TRANSLATION_TASKS, error={
                "message": f"Unknown module translation status: {self.module_translation.status}"
            })

    async def assign_translation_tasks(self, pre_response: AgentResponse) -> AgentResponse:
        await self.translator.state_manager.update_module_translation_status(
            self.module_translation.id,
            ModuleTranslationStatus.TRANSPILE
        )
        scheduler = asyncio.create_task(
            self.assign_translation_tasks_scheduler()
        )
        # 初始转译任务
        init_tasks = []
        for translation_task in self.module_translation.translation_tasks:
            if self.task_dependency_counter[translation_task.id] > 0:
                continue
            init_tasks.append(translation_task)
        for translation_task in init_tasks:
            await self.translation_task_queue.put(translation_task)

        await scheduler
        # 备份转译结果
        for file in self.module_translation.related_rust_files:
            filepath = os.path.join(self.module_translation.path, file)
            back_filepath = os.path.join(self.module_translation.path, "backup", "translation_result", file)
            os.makedirs(os.path.dirname(back_filepath), exist_ok=True)
            with open(filepath, "r", encoding="utf-8") as f:
                file_content = f.read()
            with open(back_filepath, "w", encoding="utf-8") as f:
                f.write(file_content)
        return AgentResponse.done(self, TechLeaderResponseType.MODULE_TRANSLATION_COMPLETION_DONE)

    async def assign_translation_tasks_scheduler(self):
        """定时检查任务队列，并将其分配给 Code Monkey"""
        translation_task_coros_map = {}
        last_print_timestamp = time.time()
        # 直到所有的任务都完成或失败
        while not all([
            translation_task.status in (TranslationTaskStatus.DONE, TranslationTaskStatus.FAILED)
            for translation_task in self.module_translation.translation_tasks
        ]):
            # 定期检查任务队列
            try:
                translation_task = await asyncio.wait_for(self.translation_task_queue.get(), timeout=5)
                translation_task_future = asyncio.create_task(
                    self.assign_translation_task_to_code_monkey(translation_task)
                )
                translation_task_coros_map[translation_task_future] = translation_task
            except asyncio.TimeoutError:
                ...
            except Exception as e:
                self.logger.error(f"assign translation task failed: {e}")

            # 检查任务是否完成，将完成的任务的子任务加入队列
            done_coros = [
                translation_task
                for translation_task in translation_task_coros_map.keys()
                if translation_task.done()
            ]
            for done_coro in done_coros:
                if done_coro.exception():
                    self.logger.error(f"assign translation task failed: {done_coro.exception()}")
                    raise done_coro.exception()
                translation_task_finished = translation_task_coros_map.pop(done_coro)

            # 每隔 1 分钟打印正在进行的任务
            if time.time() - last_print_timestamp > 60:
                last_print_timestamp = time.time()
                # 打印正在进行的任务
                for translation_task in translation_task_coros_map.values():
                    self.logger.info(
                        f"translation task {translation_task.source.name}({translation_task.id}) is {translation_task.status} ...")


    async def assign_translation_task_to_code_monkey(self, translation_task: TranslationTask):
        """取出转译任务，将其分配给 Code Monkey"""
        current_translation_task = translation_task
        if translation_task.status == TranslationTaskStatus.DONE:
            self.logger.info(
                f"translation task {translation_task.source.name}({translation_task.id}) already done, skipping ...")
        elif translation_task.status == TranslationTaskStatus.FAILED:
            self.logger.info(
                f"translation task {translation_task.source.name}({translation_task.id}) failed, skipping ...")
        else:
            # 并发限制
            async with self.concurrent_semaphore:
                code_monkey = CodeMonkey(
                    self.translator,
                    name=f"{CodeMonkey.ROLE}_{self.module_translation.name}_{translation_task.source.name}"
                )
                self.logger.info(
                    f"assigning translation task {translation_task.source.name}({translation_task.id}) to {code_monkey.name}")
                try:
                    agent_response = await asyncio.to_thread(
                        code_monkey.start,
                        self.module_translation,
                        translation_task
                    )
                    if agent_response and agent_response.status == AgentResponseStatus.DONE:
                        # 转译任务正常完成
                        await self.translator.state_manager.update_translation_task_status(
                            self.module_translation.id,
                            translation_task.id,
                            TranslationTaskStatus.DONE
                        )
                        self.logger.info(
                            f"translation task {translation_task.source.name}({translation_task.id}) completed by {code_monkey.name}")
                    else:
                        self.logger.error(
                            f"translation task {translation_task.source.name}({translation_task.id}) failed by {code_monkey.name}: {agent_response}"
                        )
                        await self.translator.state_manager.update_translation_task_status(
                            self.module_translation.id,
                            translation_task.id,
                            TranslationTaskStatus.FAILED
                        )
                except Exception as e:
                    error_details = traceback.format_exc()
                    self.logger.error(
                        f"translation task {translation_task.source.name}({translation_task.id}) failed by {code_monkey.name}: {e}\n{error_details}")
                    await self.translator.state_manager.update_translation_task_status(
                        self.module_translation.id,
                        translation_task.id,
                        TranslationTaskStatus.FAILED
                    )
        # 更新子任务状态
        for child_task_id in self.task_dependency_map[translation_task.id]:
            self.task_dependency_counter[child_task_id] -= 1
            if self.task_dependency_counter[child_task_id] <= 0:
                child_translation_task = self.module_translation.get_translation_task_by_id(child_task_id)
                if child_translation_task is None:
                    self.logger.error(
                        f"child translation task not found: {child_task_id}, parent translation task id: {translation_task.id}")
                    continue
                await self.translation_task_queue.put(child_translation_task)


    async def generate_translation_report(self, pre_response: AgentResponse) -> AgentResponse:
        """生成转译报告"""
        self.logger.info(f"generate translation report for {self.module_translation.name} ...")
        related_project_files = self.translator.state_manager.state.target_project.list_files(
            show_content=True,
            # 忽略不在 self.module_translation.related_rust_files 中的文件
            ignore_func=lambda filepath: filepath not in self.module_translation.related_rust_files,
            relpath=self.module_translation.path,
            show_line_numbers=True
        )
        translation_report_prompt = PromptLoader.get_prompt(
            f"{self.ROLE}/translation_report.prompt",
            module_name=self.module_translation.name,
            module_description=self.module_translation.description,
            translation_tasks=self.module_translation.translation_tasks,
            project_files=related_project_files,
            show_file_content=True
        )
        self.logger.debug(f"translation report prompt: {translation_report_prompt}")
        translation_report_response = self.call_llm(messages=[
            {"role": "user", "content": translation_report_prompt}
        ])
        translation_report_response_content = translation_report_response.choices[0].message.content
        self.logger.debug(f"translation report response content: {translation_report_response_content}")
        report_name = self.module_translation.name + "_translation_report.md"
        report_filepath = os.path.join(self.module_translation.path, report_name)
        with self.translator.file_lock_manager.file_lock(report_filepath):
            with open(report_filepath, "w", encoding="utf-8") as f:
                f.write(translation_report_response_content)
        return AgentResponse.done(self, TechLeaderResponseType.MODULE_TRANSLATION_DONE)