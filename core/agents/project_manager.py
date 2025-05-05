import asyncio
import os
import time
import traceback
from typing import  Optional

from pydantic import BaseModel, Field

from core.agents.base import AgentResponse, AgentResponseStatus, BaseAgent
from core.schema.translation import ModuleTranslation, ModuleTranslationStatus
from core.utils.prompt_loader import PromptLoader
from core.agents.tech_leader import TechLeader
from core.graph.dep_graph_visitor import DepGraphClangNodeVisitor
from core.schema.response import ProjectManagerResponseType
from core.translator import Translator


class SummarizeModuleResponse(BaseModel):
    name: str = Field(description="模块名称")
    description: str = Field(description="模块描述")

class ProjectManager(BaseAgent):
    """项目经理智能体
    负责管理转译项目的整个生命周期，包括项目的创建及初始化、项目代码追踪、团队的管理、任务的分配等。

    ```mermaid
    graph TD
        A[加载 C 项目, 执行初始化操作] --> B[为项目中每个源代码文件生成简短摘要]
        B --> C[分析 C/C++ 项目的依赖关系]
        C --> D[创建目标项目，并初始化模块信息]
        D --> E[分配模块转译任务给 Tech Leader]
    ```
    """
    ROLE = "project_manager"
    DESCRIPTION = "A powerful and efficient AI assistant responsible for managing the project."

    def __init__(
        self,
        translator: Translator,
        *,
        name: str = None,
    ):
        super().__init__(translator.llm_config, name=name)
        self.translator = translator

        self.target_project_dir: Optional[str] = None
        # 就绪队列，存放待转译的模块任务
        self.ready_module_translation_tasks = asyncio.Queue(maxsize=2)
        # 并发限制
        self.concurrent_semaphore = asyncio.Semaphore(1)

    def run(self, agent_response: AgentResponse) -> AgentResponse:
        """运行智能体"""
        if agent_response.status == AgentResponseStatus.DONE:
            # 前一个任务完成，根据前一个任务的类型执行下一个任务
            if agent_response.type == ProjectManagerResponseType.LOAD_SOURCE_PROJECT:
                # 已经成功加载 C/C++ 项目，接下来应该总结 C/C++ 项目的内容
                return asyncio.run(self.summarize_source_project(agent_response))
            elif agent_response.type == ProjectManagerResponseType.SUMMARIZE_SOURCE_PROJECT:
                # 已经成功加载 C/C++ 项目，接下来应该分析 C/C++ 项目的依赖关系
                return asyncio.run(self.analyze_dependencies(agent_response))
            elif agent_response.type == ProjectManagerResponseType.ANALYZE_DEPENDENCIES:
                # 已经成功分析 C/C++ 项目的依赖关系，接下来应该为每个模块编写上下文信息（模块名、介绍等等）
                return asyncio.run(self.create_target_project(agent_response))
            elif agent_response.type in [ProjectManagerResponseType.CREATE_TARGET_PROJECT]:
                # 已经成功创建目标项目，接下来应该分配模块转译任务给 Tech Leader
                return asyncio.run(self.assign_module_translation(agent_response))
            elif agent_response.type == ProjectManagerResponseType.ALL_MODULES_DONE:
                # TODO merge all modules
                return AgentResponse.done(self, ProjectManagerResponseType.ALL_TASKS_DONE)
        elif agent_response.type == AgentResponseStatus.ERROR:
            # TODO: exception handling
            self.logger.error(f"error occurred in run: {agent_response}")
            return agent_response

    def start(self, source_project_dir: str, target_project_dir: str, overwrite: bool = False):
        self.target_project_dir = os.path.abspath(target_project_dir)
        agent_response = asyncio.run(self.load_source_project(source_project_dir, overwrite))
        last_response = agent_response
        while True:
            agent_response = self.run(agent_response)
            if agent_response is None:
                self.logger.error(f"agent response is None, break. last_response: {last_response}")
                break
            if agent_response.type == ProjectManagerResponseType.ALL_TASKS_DONE:
                break
            last_response = agent_response

    async def load_source_project(self, project_dir: str, overwrite: bool = False) -> AgentResponse:
        """
        加载 C 项目, 识别所有源代码文件
        """
        # 将路径换算成绝对路径, 方便存储在数据库中
        project_dir = os.path.abspath(project_dir)
        if self.translator.state_manager.state.source_project:
            # 从状态管理器中已经存储了项目ID，即从过去的状态中加载项目
            self.logger.info("load project from state manager")
        else:
            # 未指定项目ID, 则通过交互的方式加载项目
            # 检索数据库中是否存在路径相同的项目
            projects = await self.translator.state_manager.list_projects_by_dirpath(project_dir)
            project = None
            if projects and overwrite:
                # TODO: 若设置了 overwrite 参数，直接在第一个项目上进行操作，否则开始新项目
                # user_choice = wait_for_choice(f"find existing projects in database, which one do you want to load?", [
                #     f"{project.name} | {project.create_time} | {project.description}"
                #     for project in projects
                # ] + ["new project"])
                project = projects[0]
            if project is None:
                # 创建新项目
                self.logger.info(f"loading source project from {project_dir}")
                await self.translator.state_manager.create_source_project(project_dir)
            else:
                await self.translator.state_manager.load_source_project_by_id(project.id)
        return AgentResponse.done(self, ProjectManagerResponseType.LOAD_SOURCE_PROJECT)

    async def summarize_source_project_file(self, project_structure: str, filepath: str, content: str, semaphore: asyncio.Semaphore = None) -> str:
        """生成文件摘要"""
        summarize_file_prompt = PromptLoader.get_prompt(
            f"{self.ROLE}/summarize_file.prompt",
            project_structure=project_structure,
            filepath=filepath,
            content=content
        )
        if semaphore:
            async with semaphore:
                summarize_response = await self.async_call_llm(messages=[
                    {"role": "user", "content": summarize_file_prompt}
                ])
        else:
            summarize_response = await self.async_call_llm(messages=[
                {"role": "user", "content": summarize_file_prompt}
            ])
        file_summary = summarize_response.choices[0].message.content
        await self.translator.state_manager.save_source_project_file(
            filepath=filepath,
            content=content,
            summary=file_summary
        )
        return file_summary

    async def summarize_source_project(self, pre_response: AgentResponse = None) -> AgentResponse:
        """生成 C/C++ 项目的摘要"""
        project_structure = self.translator.state_manager.state.source_project.pretty_structure()

        # TODO: BUG, 文件内容过大会导致卡死，需要优化
        # 生成文件摘要
        self.logger.info(f"summarize the source project files ...")
        source_project_files = self.translator.state_manager.state.source_project.list_files(
            show_content=True,
            show_summary=True,
            ignore_func=lambda filepath: not os.path.splitext(filepath)[1] in [".h", ".c"]
        )
        if all([file.summary for file in source_project_files]):
            self.logger.info("source project files already summarized, skipping ...")
        else:
            max_concurrency = 10
            concurrency_semaphore = asyncio.Semaphore(max_concurrency)
            async_tasks = []
            for file in source_project_files:
                if file.summary:
                    continue
                async_tasks.append(
                    asyncio.create_task(
                        self.summarize_source_project_file(
                            project_structure=project_structure,
                            filepath=file.path,
                            content=file.content,
                            semaphore=concurrency_semaphore
                        )
                    )
                )

            try:
                await asyncio.gather(*async_tasks)
            except Exception as e:
                self.logger.error(f"summarize source project files failed: {e}")

        # 生成项目摘要
        if self.translator.state_manager.state.source_project.description:
            self.logger.info("source project already summarized, skipping ...")
        else:
            # TODO: BUG, 文件内容过多时会超出上下文限制
            project_files = self.translator.state_manager.state.source_project.list_files(
                show_summary=True
            )
            summarize_project_prompt = PromptLoader.get_prompt(
                f"{self.ROLE}/summarize_project.prompt",
                project_name=self.translator.state_manager.state.source_project.name,
                project_structure=project_structure,
                project_files=project_files,
                show_file_summary=True,
            )
            summarize_project_response = self.call_llm(messages=[
                {"role": "user", "content": summarize_project_prompt}
            ])
            summarize_project_response_content = summarize_project_response.choices[0].message.content
            await self.translator.state_manager.update_source_project_description(summarize_project_response_content)

        return AgentResponse.done(self, ProjectManagerResponseType.SUMMARIZE_SOURCE_PROJECT)

    async def analyze_dependencies(self, pre_response: AgentResponse) -> AgentResponse:
        """
        分析 C/C++ 项目依赖关系
        """
        if self.translator.state_manager.state.module_translations:
            self.logger.info("source project dependencies already analyzed, skipping ...")
            return AgentResponse.done(self, ProjectManagerResponseType.ANALYZE_DEPENDENCIES)
        self.logger.info("analyzing source project dependencies ...")
        # 所有 .h 和 .c 文件
        source_files = self.translator.state_manager.state.source_project.list_files(
            ignore_func=lambda filepath: (os.path.splitext(filepath)[1] not in [".h", ".c", ".hpp", ".cpp"]),
            relpath=self.translator.state_manager.state.source_project.src_path
        )
        visitor = DepGraphClangNodeVisitor.from_language("c")
        for source_file in source_files:
            # TODO: 这里仅转译源代码，测试用例转译会单独考虑，后续会添加更加细致的过滤规则
            root_node = visitor.parse(
                path=os.path.join(self.translator.state_manager.state.source_project.src_path, source_file.path),
                args=["-std=c11"])
            visitor.visit(root_node)
        dep_graph = visitor.build_graph()
        for module_info, translation_unit_nodes_list_module in dep_graph.traverse_modules():
            # 将模块名转换为相对路径
            related_files = module_info["files"]
            # NOTE：这里用于测试时过滤掉一些模块
            # if any([file for file in related_files if "rb-tree" not in file]):
            #     continue
            await self.translator.state_manager.create_module_translation(
                translation_units_list=translation_unit_nodes_list_module,
                related_files=related_files
            )
        self.logger.info("source project dependencies analyzed.")
        if len(self.translator.state_manager.state.module_translations) == 0:
            self.logger.info("no modules found in the project.")
        return AgentResponse.done(self, ProjectManagerResponseType.ANALYZE_DEPENDENCIES)

    async def summarize_module(self, module_translation: ModuleTranslation, semaphore: asyncio.Semaphore = None):
        """生成模块摘要和上下文信息"""
        module_files = self.translator.state_manager.state.source_project.list_files(
            show_summary=True,
            ignore_func=lambda filepath: filepath not in module_translation.related_files
        )
        summarize_module_prompt = PromptLoader.get_prompt(
            f"{self.ROLE}/summarize_module.prompt",
            project_description=self.translator.state_manager.state.source_project.description,
            project_files=module_files,
            show_file_summary=True,
        )
        if semaphore:
            async with semaphore:
                summarize_module_response = await self.async_call_llm(messages=[
                    {"role": "user", "content": summarize_module_prompt}
                ], json_format=SummarizeModuleResponse)
        else:
            summarize_module_response = await self.async_call_llm(messages=[
                {"role": "user", "content": summarize_module_prompt}
            ], json_format=SummarizeModuleResponse)
        if summarize_module_response.format_object:
            module_translation_name = summarize_module_response.format_object.name
            module_translation_description = summarize_module_response.format_object.description
        else:
            # 根据 related_files 生成模块名
            related_filenames = list(set([os.path.splitext(os.path.basename(file))[0] for file in module_translation.related_files]))
            related_files_description = "\n".join([file.summary for file in module_files])
            module_translation_name = "_".join(related_filenames) + "_module"
            module_translation_description = f"{module_translation_name} module\n{related_files_description}"

        await self.translator.state_manager.update_module_translation_info(
            module_translation.id,
            module_translation_name,
            module_translation_description
        )

    async def create_target_project(self, pre_response: AgentResponse) -> AgentResponse:
        """创建目标项目，并初始化模块信息
        """
        if self.translator.state_manager.state.target_project:
            self.logger.info("target project already exists, skipping ...")
        else:
            self.logger.info("creating target project ...")
            # 创建一个存放转译后的项目
            # TODO: 通过大模型生成更加准确地项目信息，这里暂时使用 source_project 的信息
            project_dirpath = self.target_project_dir
            project_name = os.path.basename(project_dirpath)
            await self.translator.state_manager.create_target_project(
                name=project_name,
                dirpath=project_dirpath,
                # TODO: 这里临时使用 source_project 的描述信息
                description=self.translator.state_manager.state.source_project.description.replace(
                    self.translator.state_manager.state.source_project.name, project_name
                )
            )
        max_concurrency = 5
        concurrency_semaphore = asyncio.Semaphore(max_concurrency)
        # 为模块转译编写上下文信息
        async_tasks = []
        for module_translation in self.translator.state_manager.state.module_translations:
            if module_translation.name and module_translation.description:
                continue
            async_tasks.append(
                asyncio.create_task(
                    self.summarize_module(module_translation, concurrency_semaphore)
                )
            )
        if len(async_tasks) == 0:
            self.logger.info("all modules have been summarized, skipping ...")
        else:
            try:
                await asyncio.gather(*async_tasks)
            except Exception as e:
                self.logger.error(f"summarize module translation failed: {e}")

        return AgentResponse.done(self, ProjectManagerResponseType.CREATE_TARGET_PROJECT)

    async def assign_module_translation(self, pre_response: AgentResponse):
        """分配模块转译任务给 Tech Leader"""

        scheduler = asyncio.create_task(self.module_translation_scheduler())
        for module_translation_index, module_translation in self.translator.state_manager.state.ready_module_translations:
            await self.ready_module_translation_tasks.put(module_translation_index)
        await scheduler
        self.logger.info("all modules have been translated.")
        return AgentResponse.done(self, ProjectManagerResponseType.ALL_MODULES_DONE)

    async def module_translation_scheduler(self):
        """任务调度器，从就绪队列中获取任务并分配给 Tech Leader
        """
        last_print_timestamp = time.time()
        module_translation_coros_map = {}
        # 直到所有的任务都完成或失败
        while not all([
            module_translation.status in (ModuleTranslationStatus.DONE, ModuleTranslationStatus.FAILED)
            for module_translation in self.translator.state_manager.state.module_translations
        ]):
            try:
                # 避免一直阻塞等待
                module_translation_index = await asyncio.wait_for(self.ready_module_translation_tasks.get(), timeout=5)
                module_translation = self.translator.state_manager.state.module_translations[module_translation_index]
                module_translation_coro = asyncio.create_task(
                    self.assign_module_to_tech_leader(module_translation)
                )
                module_translation_coros_map[module_translation_coro] = module_translation
            except asyncio.TimeoutError:
                ...
            except Exception as e:
                self.logger.error(f"assign module translation failed: {e}")

            # 检查并处理已经完成的任务
            done_coros = [
                module_translation
                for module_translation in module_translation_coros_map.keys()
                if module_translation.done()
            ]
            for done_coro in done_coros:
                if done_coro.exception():
                    self.logger.error(f"module translation failed: {done_coro.exception()}, related_files: {module_translation_coros_map[done_coro].related_files}")
                module_translation_coros_map.pop(done_coro)

            # 每隔 1 分钟打印正在进行的任务
            if time.time() - last_print_timestamp > 60:
                last_print_timestamp = time.time()
                for coro, module_translation in module_translation_coros_map.items():
                    self.logger.info(f"module translation [{module_translation.name}] is {module_translation.status} ...")

    async def assign_module_to_tech_leader(self, module_translation: ModuleTranslation):
        """分配模块转译任务给 Tech Leader"""
        async with self.concurrent_semaphore:
            tech_leader = TechLeader(
                self.translator,
                name=f"{TechLeader.ROLE}_{module_translation.name}"
            )
            self.logger.info(f"assigning module translation [{module_translation.name}] to {tech_leader.name}")
            try:
                agent_response = await asyncio.to_thread(
                    tech_leader.start,
                    module_translation
                )
                if agent_response and agent_response.status == AgentResponseStatus.DONE:
                    # 正常完成任务
                    await self.translator.state_manager.update_module_translation_status(
                        module_translation.id,
                        ModuleTranslationStatus.DONE
                    )
                    self.logger.info(f"module translation [{module_translation.name}] done.")
                else:
                    # 任务失败
                    self.logger.error(f"module translation [{module_translation.name}] failed: {agent_response}")
                    await self.translator.state_manager.update_module_translation_status(
                        module_translation.id,
                        ModuleTranslationStatus.FAILED
                    )
            except Exception as e:
                error_details = traceback.format_exc()
                self.logger.error(
                    f"module translation [{module_translation.name}] failed: {e}\n{error_details}"
                )
                await self.translator.state_manager.update_module_translation_status(
                    module_translation.id,
                    ModuleTranslationStatus.FAILED
                )
