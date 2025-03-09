import asyncio
import os
import re
import traceback
from collections import defaultdict
from typing import Optional

from core.agents.base import AgentResponse, AgentResponseStatus, BaseAgent
from core.agents.bench_engineer import BenchEngineer
from core.agents.code_monkey import CodeMonkey
from core.agents.test_engineer import TestEngineer
from core.schema.response import TechLeaderResponseType, BenchEngineerResponseType, TestEngineerResponseType
from core.schema.translation import TranslationTask, TranslationTaskStatus
from core.state.state_manager import ModuleTranslation, ModuleTranslationStatus, StateManager
from core.utils.prompt_loader import PromptLoader


class TechLeader(BaseAgent):

    ROLE = "tech_leader"
    DESCRIPTION = "Tech Leader"

    def __init__(
            self,
            llm_config: dict,
            state_manager: "StateManager",
            **kwargs
    ):
        super().__init__(llm_config, **kwargs)
        self.state_manager = state_manager
        # 当前转译的 C 模块及转译任务
        self.module_translation: Optional[ModuleTranslation] = None
        # 任务依赖：task_id -> int
        self.task_dependency_counter = defaultdict(int)
        # 任务依赖：task_id -> children_task_ids
        self.task_dependency_map = defaultdict(set)
        # 转译任务队列
        self.module_translation_tasks_queue = asyncio.Queue()

    def run(self, agent_response: AgentResponse) -> AgentResponse:
        if agent_response.status == AgentResponseStatus.DONE:
            if agent_response.type == TechLeaderResponseType.PREPARE_MODULE_TRANSLATION_TASKS:
                # 分配转译任务
                return asyncio.run(self.assign_translation_tasks(agent_response))
            elif agent_response.type == TechLeaderResponseType.MODULE_TRANSLATION_DONE:
                # 风格统一
                # return self.beautify_code(agent_response)
                return AgentResponse.done(self, TechLeaderResponseType.MODULE_DONE)
                # return AgentResponse.done(self, TechLeaderResponseType.MODULE_BEAUTIFY_DONE)
            elif agent_response.type == TechLeaderResponseType.MODULE_BEAUTIFY_DONE:
                # 生成转译报告
                return self.generate_translation_report(agent_response)
            elif agent_response.type == TechLeaderResponseType.MODULE_TRANSLATION_REPORT_DONE:
                # 进行功能测试
                return self.assign_test_task(agent_response)
            elif agent_response.type == TechLeaderResponseType.MODULE_TEST_DONE:
                # 进行 benchmark 测试
                return self.assign_benchmark_task(agent_response)
            elif agent_response.type == TechLeaderResponseType.MODULE_BENCH_DONE:
                self.state_manager.mark_module_translation_as(self.module_translation.module_name, ModuleTranslationStatus.DONE)
                return AgentResponse.done(self, TechLeaderResponseType.MODULE_DONE)
            else:
                self.logger.error(f"Unknown response: {agent_response}")
        else:
            # TODO: 错误处理
            self.state_manager.mark_module_translation_as(self.module_translation.module_name, ModuleTranslationStatus.DONE)
            return AgentResponse.error(self, TechLeaderResponseType.MODULE_DONE, error=agent_response.data)

    def start(self, module_name: str) -> AgentResponse:
        self.logger.info(f"Tech Leader [{self.name}] started.")
        agent_response = asyncio.run(self.prepare_module_translation_tasks(module_name))
        while True:
            agent_response = self.run(agent_response)
            if agent_response is None:
                return AgentResponse.error(self, TechLeaderResponseType.MODULE_TRANSLATION_DONE, error={
                    "message": "No response from the agent."
                })
            if agent_response.type == TechLeaderResponseType.MODULE_DONE:
                return agent_response

    async def summary_translation_task(self, translation_task: TranslationTask):
        """总结转译任务"""
        file_summary = self.state_manager.source_project.file_summaries.get(translation_task.source.nodes[0].filepath, "")
        summary_translation_task_prompt = PromptLoader.get_prompt(
            f"{self.ROLE}/summary_translation_task.prompt",
            translation_task=translation_task,
            file_summary=file_summary
        )
        summary_response = self.call_llm(messages=[
            {"role": "user", "content": summary_translation_task_prompt}
        ])
        summary_response_content = summary_response.choices[0].message.content
        return summary_response_content

    async def process_summary_translation_task(
            self,
            translation_task: TranslationTask,
            semaphore: asyncio.Semaphore,
            completed_tasks_counter: list,
            update_threshold: int = 10
    ):
        async with semaphore:
            summary = await self.summary_translation_task(translation_task)
            translation_task.source.description = summary

            completed_tasks_counter[0] += 1  # 使用列表来保持引用传递

            if completed_tasks_counter[0] >= update_threshold:
                self.state_manager.update()  # 执行更新操作
                completed_tasks_counter[0] = 0  # 重置计数器

    async def prepare_module_translation_tasks(self, module_name: str) -> AgentResponse:
        self.logger.info(f"Preparing module translation: [{module_name}]")
        self.module_translation = self.state_manager.translator.module_translations.get(module_name)
        self.logger.info(f"Summary translation tasks for module: [{module_name}]")
        # 为每个任务添加一段描述
        summary_coros = []
        summary_translation_tasks = []

        # 设置并发限制
        semaphore = asyncio.Semaphore(20)
        # 计数器用于跟踪已完成的任务数量
        completed_tasks_counter = [0]  # 使用列表来保持引用传递
        # TODO: 任务数过多时会导致内存占用过高，阻塞卡死
        for translation_task in self.module_translation.translation_tasks:
            if translation_task.source.description:
                continue
            # 每个任务都会调用 process_translation_task
            summary_coros.append(self.process_summary_translation_task(translation_task, semaphore, completed_tasks_counter))

        # for translation_task in self.module_translation.translation_tasks:
        #     if translation_task.source.description:
        #         continue
        #     summary_coros.append(
        #         asyncio.create_task(self.summary_translation_task(translation_task))
        #     )
        #     summary_translation_tasks.append(translation_task)
        await asyncio.gather(*summary_coros)
        # for summary, translation_task in zip(summaries, summary_translation_tasks):
        #     translation_task.source.description = summary
        self.state_manager.update()

        # 将没有前置任务的任务加入队列
        for translation_task in self.module_translation.translation_tasks:
            if len(translation_task.prerequisites) == 0:
                self.module_translation_tasks_queue.put_nowait(translation_task)
            for prerequisite_task_id in translation_task.prerequisites:
                self.task_dependency_map[prerequisite_task_id].add(translation_task.source.id)
            self.task_dependency_counter[translation_task.source.id] = len(translation_task.prerequisites)
        self.logger.info(f"Preparing module translation: [{module_name}]")
        if self.module_translation.status == ModuleTranslationStatus.INIT:
            return AgentResponse.done(self, TechLeaderResponseType.PREPARE_MODULE_TRANSLATION_TASKS)
        elif self.module_translation.status == ModuleTranslationStatus.TRANSPILE:
            return AgentResponse.done(self, TechLeaderResponseType.MODULE_TRANSLATION_REPORT_DONE)
        elif self.module_translation.status == ModuleTranslationStatus.TEST:
            return AgentResponse.done(self, TechLeaderResponseType.MODULE_TEST_DONE)
        elif self.module_translation.status == ModuleTranslationStatus.BENCHMARK:
            return AgentResponse.done(self, TechLeaderResponseType.MODULE_BENCH_DONE)

    async def assign_translation_tasks(self, pre_response: AgentResponse) -> AgentResponse:
        await asyncio.create_task(
            self.assign_translation_tasks_scheduler()
        )
        return AgentResponse.done(self, TechLeaderResponseType.MODULE_TRANSLATION_DONE)

    async def assign_translation_tasks_scheduler(self):
        """定时检查任务队列，并将其分配给 Code Monkey"""
        translation_task_future_map = {}
        while not all([translation_task.status in (TranslationTaskStatus.DONE, TranslationTaskStatus.FAILED) for translation_task in self.module_translation.translation_tasks]):
            # 定期检查任务队列
            try:
                translation_task = await asyncio.wait_for(self.module_translation_tasks_queue.get(), timeout=1)
                translation_task_future = asyncio.create_task(
                    self.assign_translation_task_to_code_monkey(translation_task)
                )
                translation_task_future_map[translation_task_future] = translation_task.source.id
            except asyncio.TimeoutError:
                ...

            # 检查任务是否完成，将完成的任务的子任务加入队列
            done_futures = [future for future in translation_task_future_map.keys() if future.done()]
            for done_future in done_futures:
                translation_task_id = translation_task_future_map.pop(done_future)
                for child_task_id in self.task_dependency_map[translation_task_id]:
                    self.task_dependency_counter[child_task_id] -= 1
                    if self.task_dependency_counter[child_task_id] == 0:
                        self.module_translation_tasks_queue.put_nowait(
                            self.module_translation.get_translation_task_by_task_id(child_task_id)
                        )

    async def assign_translation_task_to_code_monkey(self, translation_task: TranslationTask):
        """取出转译任务，将其分配给 Code Monkey"""
        if translation_task.status == TranslationTaskStatus.DONE:
            self.logger.info(
                f"Translation task {translation_task.source.name}({translation_task.source.id}) already done, skipping ...")
            return
        code_monkey = CodeMonkey(self.llm_config, self.state_manager,
                                 name=f"{CodeMonkey.ROLE}_{self.module_translation.module_name.replace('/', '_')}_{translation_task.source.name}")
        self.logger.info(
            f"Assign translation task {translation_task.source.name}({translation_task.source.id}) to {code_monkey.name}")
        try:
            agent_response = await asyncio.to_thread(
                code_monkey.start,
                self.module_translation.module_name,
                translation_task
            )
            if agent_response.status == AgentResponseStatus.DONE:
                self.state_manager.mark_translation_task_as(
                    self.module_translation.module_name,
                    translation_task.source.id,
                    TranslationTaskStatus.DONE
                )
                self.logger.info(
                    f"Translation task {translation_task.source.name}({translation_task.source.id}) completed by {code_monkey.name}")
            else:
                self.state_manager.mark_translation_task_as(
                    self.module_translation.module_name,
                    translation_task.source.id,
                    TranslationTaskStatus.FAILED
                )
                self.logger.error(
                    f"Translation task {translation_task.source.name}({translation_task.source.id}) failed by {code_monkey.name}.\n{agent_response.error}")
        except Exception as e:
            error_details = traceback.format_exc()
            self.logger.error(
                f"Translation task {translation_task.source.name}({translation_task.source.id}) failed by {code_monkey.name}.\nexception: {error_details}")
            self.state_manager.mark_translation_task_as(
                self.module_translation.module_name,
                translation_task.source.id,
                TranslationTaskStatus.FAILED
            )

    def beautify_code(self, pre_response: AgentResponse) -> AgentResponse:
        """由于 Tech Leader 是将所有转译任务派发给多个 Code Monkey，因此可能导致代码编写混乱
        因此 Tech Leader 需要对转译后的代码进行风格统一
        """
        self.logger.info(f"Beautify {self.module_translation.module_name} code ...")
        related_project_files = self.state_manager.target_project.list_files(
            show_content=True,
            # 忽略不在 self.module_translation.related_rust_files 中的文件
            ignore_func=lambda filepath: os.path.isfile(os.path.join(
                self.state_manager.target_project.path,
                filepath
            )) and filepath not in self.module_translation.related_rust_files
        )
        beautify_prompt = PromptLoader.get_prompt(
            f"{self.ROLE}/beautify.prompt",
            project_files=related_project_files,
            show_file_content=True
        )
        self.logger.debug(f"Beautify prompt: {beautify_prompt}")
        beautify_messages = [
            {"role": "user", "content": beautify_prompt}
        ]
        beautify_response = self.call_llm(beautify_messages)
        beautify_response_content = beautify_response.choices[0].message.content
        self.logger.debug(f"Beautify response content: {beautify_response_content}")
        # 提取代码
        rust_code_blocks = re.findall(r"(?<!\/\/\/)(?<=```rust\n)([\s\S]*?)(?=\n```)", beautify_response_content,
                                      re.DOTALL)
        for rust_code_block in rust_code_blocks:
            # 第一行为文件路径
            filepath_line, *code_lines = rust_code_block.split("\n")
            if "filepath" not in filepath_line:
                continue
            rust_file = filepath_line.split(":")[1].strip()
            rust_filepath = os.path.join(self.state_manager.target_project.path, rust_file)
            with self.state_manager.file_lock_manager.file_lock(rust_filepath):
                with open(rust_filepath, "w", encoding="utf-8") as f:
                    f.write("\n".join(code_lines))
                self.state_manager.target_vcs.add([rust_filepath])
        self.state_manager.target_vcs.commit(f"Beautify code: {self.module_translation.module_name}")
        return AgentResponse.done(self, TechLeaderResponseType.MODULE_BEAUTIFY_DONE)

    def generate_translation_report(self, pre_response: AgentResponse) -> AgentResponse:
        """生成转译报告"""
        self.logger.info(f"Generate translation report for {self.module_translation.module_name} ...")
        related_project_files = self.state_manager.target_project.list_files(
            show_content=True,
            # 忽略不在 self.module_translation.related_rust_files 中的文件
            ignore_func=lambda filepath: os.path.isfile(os.path.join(
                self.state_manager.target_project.path,
                filepath
            )) and filepath not in self.module_translation.related_rust_files
        )
        translation_report_prompt = PromptLoader.get_prompt(
            f"{self.ROLE}/translation_report.prompt",
            translation_tasks=self.module_translation.translation_tasks,
            project_files=related_project_files,
            show_file_content=True
        )
        self.logger.debug(f"Translation report prompt: {translation_report_prompt}")
        translation_report_response = self.call_llm(messages=[
            {"role": "user", "content": translation_report_prompt}
        ])
        translation_report_response_content = translation_report_response.choices[0].message.content
        self.logger.debug(f"Translation report response content: {translation_report_response_content}")
        # FIX: 目前仅支持一个文件
        rust_file = self.module_translation.related_rust_files[0]
        report_name = rust_file.split("/")[-1].replace(".rs", "_report.md")
        report_file = os.path.join("reports", "translations", report_name)
        report_filepath = os.path.join(self.state_manager.target_project.path, report_file)
        os.makedirs(os.path.dirname(report_filepath), exist_ok=True)
        with self.state_manager.file_lock_manager.file_lock(report_filepath):
            with open(report_filepath, "w", encoding="utf-8") as f:
                f.write(translation_report_response_content)
            self.state_manager.target_vcs.add([report_filepath])
        self.state_manager.target_vcs.commit(f"Generate translation report: {self.module_translation.module_name}")
        self.state_manager.mark_module_translation_as(self.module_translation.module_name, ModuleTranslationStatus.TRANSPILE)
        return AgentResponse.done(self, TechLeaderResponseType.MODULE_TRANSLATION_REPORT_DONE)

    def assign_test_task(self, agent_response: AgentResponse) -> AgentResponse:
        # 找到失败的任务
        failed_tasks = [translation_task for translation_task in self.module_translation.translation_tasks if
                        translation_task.status == TranslationTaskStatus.FAILED]
        # FIX: 当转译任务存在失败时，不进行测试
        if len(failed_tasks) > 0:
            self.logger.error(f"Translation task failed, skip test ...")
            return AgentResponse.done(self, TechLeaderResponseType.MODULE_TEST_DONE)
        test_engineer = TestEngineer(
            self.llm_config,
            self.state_manager,
            name=f"{TestEngineer.ROLE}_{self.module_translation.module_name.replace('/', '_')}"
        )
        self.logger.info(f"Assign test task to {test_engineer.name} ...")
        try:
            agent_response = test_engineer.start(
                module_name=self.module_translation.module_name,
                pre_response=agent_response
            )
            self.state_manager.mark_module_translation_as(self.module_translation.module_name,
                                                          ModuleTranslationStatus.TEST)
            if agent_response.status == AgentResponseStatus.DONE and agent_response.type == TestEngineerResponseType.TEST_PASSED:
                return AgentResponse.done(self, TechLeaderResponseType.MODULE_TEST_DONE)
            else:
                return AgentResponse.error(self, TechLeaderResponseType.MODULE_TEST_DONE, error=agent_response.data)
        except Exception as e:
            # FIX: 当测试任务报错时，返回错误信息
            error_details = traceback.format_exc()
            self.logger.error(f"Test task failed: {error_details}")
            return AgentResponse.error(self, TechLeaderResponseType.MODULE_TEST_DONE, error={
                "message": f"Test task failed: {error_details}",
            })

    def assign_benchmark_task(self, pre_response: AgentResponse) -> AgentResponse:
        # 找到失败的任务
        failed_tasks = [translation_task for translation_task in self.module_translation.translation_tasks if
                        translation_task.status == TranslationTaskStatus.FAILED]
        if len(failed_tasks) > 0:
            self.logger.error(f"Translation task failed, skip benchmark ...")
            return AgentResponse.done(self, TechLeaderResponseType.MODULE_BENCH_DONE)
        bench_engineer = BenchEngineer(
            llm_config=self.llm_config,
            state_manager=self.state_manager,
            name=f"{BenchEngineer.ROLE}_{self.module_translation.module_name.replace('/', '_')}"
        )
        self.logger.info(f"Assign benchmark task to {bench_engineer.name} ...")
        try:
            agent_response = bench_engineer.start(self.module_translation.module_name)
            self.state_manager.mark_module_translation_as(self.module_translation.module_name,
                                                          ModuleTranslationStatus.BENCHMARK)
            if agent_response.status == AgentResponseStatus.DONE and agent_response.type == BenchEngineerResponseType.BENCH_DONE:
                return AgentResponse.done(self, TechLeaderResponseType.MODULE_BENCH_DONE)
            else:
                return AgentResponse.error(self, TechLeaderResponseType.MODULE_BENCH_DONE, error=agent_response.data)
        except Exception as e:
            error_details = traceback.format_exc()
            self.logger.error(f"Benchmark task failed: {error_details}")
            return AgentResponse.error(self, TechLeaderResponseType.MODULE_BENCH_DONE, error={
                "message": f"Benchmark task failed: {error_details}",
            })


