import asyncio
import hashlib
import os
import time
import traceback
from typing import Literal, Optional

from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor, as_completed, wait

# from sqlalchemy import select

from core.config import Config
from core.agents.base import AgentResponse, AgentResponseStatus, BaseAgent
# from core.agents.transpile_memory import TranspileMemory
# from core.db.session import SessionManager
# from core.db.models.project import Project
from core.utils.prompt_loader import PromptLoader
from core.agents.tech_leader import TechLeader
from core.graph.dep_graph import DGNode
from core.graph.dep_graph_visitor import DepGraphClangNodeVisitor
from core.schema.response import ProjectManagerResponseType
from core.state.state_manager import StateManager, generate_id, ModuleTranslationStatus
from core.schema.translation import TranslationTask, TranslationTaskSource, TranslationUnitNode


class CreateRustProjectFileJsonFormat(BaseModel):
    filepath: str = Field(description="Rust 项目的相对文件路径。")
    text: str = Field(description="Rust 项目的文件内容。")


class CreateRustProjectJsonFormat(BaseModel):
    type: Literal["bin", "lib"] = Field(description="Rust 项目的类型，可以是 'bin' 或 'lib'。")
    name: str = Field(description="Rust 项目的名称。")
    path: str = Field(description="Rust 项目的路径。")
    description: str = Field(description="Rust 项目的描述。")
    files: list[CreateRustProjectFileJsonFormat] = Field(description="Rust 项目的文件列表。")
    file_map: dict[str, list[str]] = Field(description="C/C++ 项目模块与 Rust 项目文件进行映射。")


def construct_module_translation_tasks(project_dir, translation_unit_nodes_list: list[list[list[DGNode]]]) -> list[
    TranslationTask]:
    """
    构建转译任务, 将 location 转为相对路径
    """
    module_translation_tasks = []
    # node_id -> task_id
    node_task_lookup_map = {}
    # 遍历所有转译的节点
    for translation_unit_nodes in translation_unit_nodes_list:
        # 遍历所有可并行转译的节点
        for translation_unit in translation_unit_nodes:
            translation_task = TranslationTask(
                source=TranslationTaskSource(
                    id=translation_unit[-1].id,
                    name=translation_unit[-1].name,
                    nodes=[
                        TranslationUnitNode(
                            filepath=os.path.relpath(node.location, project_dir),
                            id=node.id,
                            name=node.name,
                            type=node.type,
                            text=(node.extra.get("raw_comment", "") or "") + "\n" + node.text
                        )
                        for node in translation_unit
                    ]
                ),
                target=None,
                status="init",
                prerequisites=list(set(
                    node_task_lookup_map[edge.dst.id]
                    for node in translation_unit
                    for edge in node.edges
                    if edge.dst.id not in [
                        n.id for n in translation_unit
                    ]
                ))
            )
            for node in translation_unit:
                node_task_lookup_map[node.id] = translation_task.source.id
            module_translation_tasks.append(translation_task)
    return module_translation_tasks


def generate_project_id(project_dir: str) -> str:
    """基于文件的项目中每个文件的元数据生成项目唯一ID
    """
    project_dirname = os.path.basename(project_dir)
    project_hash = hashlib.sha256(project_dirname.encode()).hexdigest()
    for root, dirs, files in os.walk(project_dir):
        for file in files:
            filepath = os.path.join(root, file)
            stat = os.stat(filepath)
            metadata = f"{file}{stat.st_size}{stat.st_mtime}"
            project_hash = hashlib.sha256((project_hash + metadata).encode()).hexdigest()
    return project_hash


# class ProjectManagerService:
#     """Project Manager 服务类
#     用于封装 Project Manager 的数据库操作
#     """
#
#     def __init__(self, state_manager: StateManager):
#         self.state_manager = state_manager
#
#     async def get_project_by_project_id(self, project_id: str) -> Optional[Project]:
#         async with SessionManager() as session:
#             results = await session.execute(
#                 select(Project).where(Project.id == project_id)
#             )
#             project = results.scalars().first()
#             return project
#
#     async def insert(self, project: Project):
#         async with SessionManager() as session:
#             session.add(project)
#             await session.commit()
#             return project
#
#     async def update(self, project: Project):
#         async with SessionManager() as session:
#             session.add(project)
#             await session.commit()
#             return project


class ProjectManager(BaseAgent):
    """项目经理智能体
    负责管理转译项目的整个生命周期，包括项目的创建及初始化、项目代码追踪、团队的管理、任务的分配等。

    1. 读取 C/C++ 项目的代码，构建依赖图
    2. 总结 C/C++ 项目的内容
    3. 分配模块转译任务给 Tech Leader
    """
    ROLE = "project_manager"
    DESCRIPTION = "A powerful and efficient AI assistant responsible for managing the project."

    def __init__(
            self,
            llm_config: dict,
            *,
            name: str = None,
    ):
        super().__init__(llm_config, name=name)
        self.state_filepath = "temp/default_state.json"
        # 整个项目的状态管理器
        self.state_manager = StateManager()
        # 就绪队列，存放待转译的模块任务
        self.ready_module_translation_tasks = asyncio.Queue(maxsize=5)
        # self.service = ProjectManagerService(self.state_manager)

        # self.memory = TranspileMemory(llm_config, self.state_manager)

    def run(self, pre_response: AgentResponse) -> AgentResponse:
        """运行智能体"""
        if pre_response.status == AgentResponseStatus.DONE:
            # 前一个任务完成，根据前一个任务的类型执行下一个任务
            if pre_response.type == ProjectManagerResponseType.LOAD_SOURCE_PROJECT:
                # 已经成功加载 C/C++ 项目，接下来应该总结 C/C++ 项目的内容
                return self.summarize_source_project(pre_response)
            elif pre_response.type == ProjectManagerResponseType.SUMMARIZE_SOURCE_PROJECT:
                # 已经成功加载 C/C++ 项目，接下来应该分析 C/C++ 项目的依赖关系
                return self.analyze_dependencies(pre_response)
            elif pre_response.type == ProjectManagerResponseType.ANALYZE_DEPENDENCIES:
                # 已经成功分析完依赖，接下来将每个项目分配个 Tech Leader
                return self.assign_module(pre_response)
            # elif pre_response.type in [ProjectManagerResponseType.CREATE_RUST_PROJECT]:
            #     # 已经成功分析 C/C++ 项目的依赖关系，接下来应该将 C/C++ 项目转译为 Rust 项目
            #     return asyncio.run(self.assign_module_translation(pre_response))
            # elif pre_response.type == ProjectManagerResponseType.ALL_MODULES_DONE:
            #     self.memory.wait_all_tasks()
            #     return AgentResponse.done(self, ProjectManagerResponseType.ALL_TASKS_DONE)
        elif pre_response.type == AgentResponseStatus.ERROR:
            # TODO: 错误处理
            return AgentResponse.error(self, ProjectManagerResponseType.ALL_MODULES_DONE, error=pre_response.data)

    def start(self, project_dir: str):
        agent_response = asyncio.run(self.load_source_project(project_dir))
        while True:
            agent_response = self.run(agent_response)
            if agent_response.type == ProjectManagerResponseType.ALL_TASKS_DONE:
                break
            if agent_response is None:
                break

    async def load_source_project(self, project_dir: str) -> AgentResponse:
        """
        加载 C/C++ 项目到全局记忆中
        在数据库中创建该项目的记录
        """
        project_id = generate_project_id(project_dir)
        # project = await self.service.get_project_by_project_id(project_id)
        # if project:
        #     self.logger.info(f"Loaded project from database: {project.name}({project.id})")
        #     # TODO：存储项目状态
        # else:
        #     # 创建项目记录
        #     project = Project(
        #         id=project_id,
        #         name=os.path.basename(project_dir),
        #         dirpath=project_dir,
        #         description="",  # 项目描述先滞空，后续会自动生成
        #     )
        #     project = await self.service.insert(project)

        self.state_filepath = f"{Config.RUST_PROJECTS_PATH}/{project_id}_state.json"
        self.state_manager.bind_filepath(self.state_filepath)

        if os.path.exists(self.state_filepath):
            self.logger.info(f"Loaded project from {self.state_filepath}")
            self.state_manager.load(self.state_filepath)
        if self.state_manager.source_project:
            return AgentResponse.done(self, ProjectManagerResponseType.LOAD_SOURCE_PROJECT)
        # 加载 C/C++ 项目
        self.logger.info(f"Loading C/C++ project from {project_dir} ...")
        self.state_manager.load_source_project(
            project_name=os.path.basename(project_dir),
            project_dir=project_dir
        )
        return AgentResponse.done(self, ProjectManagerResponseType.LOAD_SOURCE_PROJECT)

    def summarize_source_project(self, pre_response: AgentResponse = None) -> AgentResponse:
        """生成 C/C++ 项目的摘要"""
        if self.state_manager.source_project.description and self.state_manager.source_project.file_summaries:
            self.logger.info("C/C++ project already summarized, skipping ...")
            return AgentResponse.done(self, ProjectManagerResponseType.SUMMARIZE_SOURCE_PROJECT)
        # 生成文件摘要
        project_structure = self.state_manager.source_project.pretty_structure()

        # TODO：文件内容过大会导致卡死
        def _summarize_file(filepath: str, content: str):
            summarize_file_prompt = PromptLoader.get_prompt(
                f"{self.ROLE}/summarize_file.prompt",
                project_structure=project_structure,
                filepath=filepath,
                content=content
            )
            summarize_response = self.call_llm(messages=[
                {"role": "user", "content": summarize_file_prompt}
            ])
            return summarize_response.choices[0].message.content

        if not self.state_manager.source_project.file_summaries:
            self.logger.info(f"Summarize the project [{self.state_manager.source_project.path}] files ...")
            file_summaries = {}
            source_project_files = self.state_manager.source_project.list_files(
                show_content=True,
                ignore_func=lambda filepath: not os.path.splitext(filepath)[1] in [".h", ".c"]
            )
            with ThreadPoolExecutor(max_workers=max(int(len(source_project_files) / 2), 1)) as executor:
                futures = {
                    executor.submit(_summarize_file, file.path, file.content): file.path
                    for file in source_project_files
                }

                for future in as_completed(futures):
                    file_summary = future.result()
                    filepath = futures[future]
                    file_summaries[filepath] = file_summary
            self.state_manager.set_source_project_file_summaries(file_summaries)
        self.logger.info(f"Summarizing the project [{self.state_manager.source_project.name}] ...")
        # 生成项目摘要
        project_files = self.state_manager.source_project.list_files(
            show_summary=True
        )
        summarize_project_prompt = PromptLoader.get_prompt(
            f"{self.ROLE}/summarize_project.prompt",
            project_name=self.state_manager.source_project.name,
            project_dir=self.state_manager.source_project.path,
            project_structure=project_structure,
            project_files=project_files,
            show_file_summary=True,
        )
        summarize_project_response = self.call_llm(messages=[
            {"role": "user", "content": summarize_project_prompt}
        ])
        summarize_project_response_content = summarize_project_response.choices[0].message.content
        self.state_manager.set_source_project_description(summarize_project_response_content)
        return AgentResponse.done(self, ProjectManagerResponseType.SUMMARIZE_SOURCE_PROJECT)

    def analyze_dependencies(self, pre_response: AgentResponse) -> AgentResponse:
        """
        分析 C/C++ 项目依赖关系
        """
        if self.state_manager.translator.modules and self.state_manager.translator.module_translations:
            self.logger.info("Dependencies already analyzed, skipping ...")
            return AgentResponse.done(self, ProjectManagerResponseType.ANALYZE_DEPENDENCIES)
        self.logger.info("Analyzing dependencies ...")
        # 所有 .h 和 .c 文件
        source_files = self.state_manager.source_project.list_files(
            ignore_func=lambda filepath: not os.path.splitext(filepath)[1] in [".h", ".c", ".hpp", ".cpp"])
        visitor = DepGraphClangNodeVisitor.from_language("c")
        for source_file in source_files:
            # TODO: 这里会跳过含有 test 的文件，仅转译源代码，测试用例转译会单独考虑，后续会添加更加细致的过滤规则
            # if "test" in source_file.path:
            #     continue
            root_node = visitor.parse(path=os.path.join(self.state_manager.source_project.path, source_file.path),
                                      # args=["-std=c++11"],
                                      expected_files=[
                                          os.path.join(self.state_manager.source_project.path, file.path)
                                          for file in source_files
                                      ])
            visitor.visit(root_node)
        dep_graph = visitor.build_graph()
        for module_info, translation_unit_nodes_list_module in dep_graph.traverse_modules():
            # 将模块名转换为相对路径
            # if "arraylist" in module_name:
            #     continue
            module_name = str(module_info["module"])

            module_translation_tasks = construct_module_translation_tasks(
                self.state_manager.source_project.path,
                translation_unit_nodes_list_module)
            self.state_manager.add_module_translation(module_name, module_translation_tasks, info=module_info)
            self.logger.debug(f"Module [{module_name}] translation tasks created.")
        if not self.state_manager.translator.modules:
            self.logger.error("No modules found in the project.")
            return AgentResponse.error(self, ProjectManagerResponseType.ANALYZE_DEPENDENCIES,
                                       error={"message": "No modules found in the project."})
        return AgentResponse.done(self, ProjectManagerResponseType.ANALYZE_DEPENDENCIES)

    def assign_module(self, pre_response: AgentResponse) -> AgentResponse:
        """
        创建 Rust 项目
        """
        modules_num = max(min(len(self.state_manager.translator.modules), 3), 1)
        executor = ThreadPoolExecutor(max_workers=modules_num)
        futures = []
        for module_name in self.state_manager.translator.modules:
            sub_state_manager = StateManager()

            sub_state_manager_bound_filepath = self.state_manager.bound_filepath.replace("state.json",
                                                                                         f"{module_name}_state.json")
            sub_state_manager.bind_filepath(sub_state_manager_bound_filepath)
            if os.path.exists(sub_state_manager_bound_filepath):
                sub_state_manager.load(sub_state_manager_bound_filepath)
            else:
                sub_state_manager.load_source_project(
                    project_name=self.state_manager.source_project.name,
                    project_dir=self.state_manager.source_project.path
                )
                sub_state_manager.set_source_project_description(self.state_manager.source_project.description)
                sub_state_manager.set_source_project_file_summaries(self.state_manager.source_project.file_summaries)
                # 转译信息
                sub_state_manager.add_module_translation(
                    module_name, self.state_manager.translator.module_translations[module_name].translation_tasks
                )
            sub_project_manager = SubProjectManager(llm_config=self.llm_config, state_manager=sub_state_manager,
                                                    name=f"{SubProjectManager.ROLE}_{module_name.replace('/', '_')}")
            future = executor.submit(
                sub_project_manager.start
            )
            futures.append(future)

        wait(futures)
        executor.shutdown(wait=True)
        return AgentResponse.done(self, ProjectManagerResponseType.ALL_TASKS_DONE)


# ===========================================================================

# TODO: 处理并发时无法多个模块同时转译的问题，这里为不同的模块创建不同的 Rust 项目
class SubProjectManager(BaseAgent):
    ROLE = "project_manager"
    DESCRIPTION = "A powerful and efficient AI assistant responsible for managing the project."

    def __init__(
            self,
            llm_config: dict,
            state_manager: StateManager,
            *,
            name: str = None,
    ):
        super().__init__(llm_config, name=name)
        self.state_filepath = "temp/default_state.json"
        # 整个项目的状态管理器
        self.state_manager = state_manager
        # 就绪队列，存放待转译的模块任务
        self.ready_module_translation_tasks = asyncio.Queue(maxsize=10)

        # self.memory = TranspileMemory(llm_config, self.state_manager)

    def run(self, pre_response: AgentResponse) -> AgentResponse:
        """运行智能体"""
        if pre_response.status == AgentResponseStatus.DONE:
            # 前一个任务完成，根据前一个任务的类型执行下一个任务
            if pre_response.type == ProjectManagerResponseType.ANALYZE_DEPENDENCIES:
                # 已经成功创建 Rust 项目，接下来应该分析 C/C++ 项目，为转译做准备
                return self.create_rust_project(pre_response)
            elif pre_response.type in [ProjectManagerResponseType.CREATE_RUST_PROJECT]:
                # 已经成功分析 C/C++ 项目的依赖关系，接下来应该将 C/C++ 项目转译为 Rust 项目
                return asyncio.run(self.assign_module_translation(pre_response))
            elif pre_response.type == ProjectManagerResponseType.ALL_MODULES_DONE:
                # self.memory.wait_all_tasks()
                return AgentResponse.done(self, ProjectManagerResponseType.ALL_TASKS_DONE)
        elif pre_response.type == AgentResponseStatus.ERROR:
            # TODO: 错误处理
            return AgentResponse.error(self, ProjectManagerResponseType.ALL_MODULES_DONE, error=pre_response.data)

    def start(self):
        agent_response = self.create_rust_project(
            AgentResponse.done(self, ProjectManagerResponseType.ANALYZE_DEPENDENCIES))
        while True:
            agent_response = self.run(agent_response)
            if agent_response.type == ProjectManagerResponseType.ALL_TASKS_DONE:
                break
            if agent_response is None:
                break

    def create_rust_project(self, pre_response: AgentResponse) -> AgentResponse:
        """
        创建 Rust 项目
        """
        if self.state_manager.target_project:
            self.logger.info("Rust project already exists, skipping ...")
            return AgentResponse.done(self, ProjectManagerResponseType.CREATE_RUST_PROJECT)
        source_project = self.state_manager.source_project
        self.logger.info(f"Creating Rust project from project [{source_project.name}] ...")
        #         # ============================== humaneval ==============================
        #         module_name = self.state_manager.translator.modules[0]
        #         rust_project_name = f"humaneval_{module_name}"
        #         rust_project_description = f"Rust project for module [{module_name}]"
        #         cargo_toml_text = f"""
        # [package]
        # name = "contest"
        # version = "0.1.0"
        # edition = "2021"
        #
        # # See more keys and their definitions at https://doc.rust-lang.org/cargo/reference/manifest.html
        #
        # [dependencies]
        # rand = "0.4"
        # regex = "1"
        # md5 = "0.7.0"
        #         """
        #         rust_project_files = [
        #             CreateRustProjectFileJsonFormat(filepath="src/main.rs", text=""),
        #             CreateRustProjectFileJsonFormat(filepath="Cargo.toml", text=cargo_toml_text)
        #         ]
        #         crate_type = "bin"
        #         file_map = {
        #             module_name: [
        #                 "src/main.rs"
        #             ]
        #         }
        #         # ============================== humaneval ==============================
#         # ============================== contest ==============================
#         module_name = self.state_manager.translator.modules[0].replace("-", "_")
#         rust_project_name = f"{module_name}"
#         rust_project_description = f"Rust project for module [{module_name}]"
#         cargo_toml_text = f"""
# [package]
# name = "contest"
# version = "0.1.0"
# edition = "2021"
#                 """
#         rust_project_files = [
#             CreateRustProjectFileJsonFormat(filepath="src/main.rs", text=""),
#             CreateRustProjectFileJsonFormat(filepath="Cargo.toml", text=cargo_toml_text)
#         ]
#         crate_type = "bin"
#         file_map = {
#             module_name: [
#                 "src/main.rs"
#             ]
#         }
#         # ============================== contest ==============================

        module = self.state_manager.translator.module_translations["0"]
        project_files = self.state_manager.source_project.list_files(
            ignore_func=lambda filepath: filepath not in module.related_c_files,
            show_summary=True
        )
        create_rust_project_prompt = PromptLoader.get_prompt(
            f"{self.ROLE}/create_rust_project.prompt",
            project_description=source_project.description,
            project_files=project_files,
            show_file_summary=True,
            # project_structure=source_project.pretty_structure(),
            # modules=self.state_manager.translator.modules
        )
        self.logger.debug(f"Create Rust project prompt: \n{create_rust_project_prompt}")
        response = self.call_llm(messages=[
            {"role": "user", "content": create_rust_project_prompt}
        ], json_format=CreateRustProjectJsonFormat)
        self.logger.debug(f"Create Rust project response: \n{response.choices[0].message.content}")
        json_obj = response.format_object
        if json_obj.type not in ("bin", "lib"):
            # 异常处理
            json_obj.type = "bin"
        rust_project_name = json_obj.name.replace("-", "_")
        rust_project_path = f"{Config.RUST_PROJECTS_PATH}/{rust_project_name}"
        rust_project_description = json_obj.description
        rust_project_files = json_obj.files
        crate_type = json_obj.type
        file_map = json_obj.file_map
        if os.path.exists(rust_project_path):
            rust_project_path = f"{rust_project_path}_{generate_id(str(time.time()))[:8]}"
        # 创建项目目录
        os.makedirs(rust_project_path, exist_ok=True)
        for rust_file in rust_project_files:
            rust_filepath = os.path.join(rust_project_path, rust_file.filepath)
            os.makedirs(os.path.dirname(rust_filepath), exist_ok=True)
            with self.state_manager.file_lock_manager.file_lock(rust_filepath):
                with open(rust_filepath, "w", encoding="utf-8") as f:
                    f.write(rust_file.text)
            self.logger.debug(f"Rust file [{rust_filepath}] created.")
        self.logger.info(f"Rust project [{rust_project_name}] created at {rust_project_path}")
        self.state_manager.create_rust_project(
            project_name=rust_project_name,
            project_dir=rust_project_path,
            crate_type=crate_type,
            project_description=rust_project_description
        )
        for module_name, related_rust_files in file_map.items():
            self.state_manager.add_module_rust_files("0", related_rust_files)
            self.logger.debug(f"Module [{module_name}] related Rust files added: {related_rust_files}.")
        return AgentResponse.done(self, ProjectManagerResponseType.CREATE_RUST_PROJECT)

    async def assign_module_translation(self, pre_response: AgentResponse):
        """分配模块转译任务给技术负责人"""
        # 初始化转译就绪队列
        for module_name in self.state_manager.translator.ready_modules:
            # TODO：Debug，这里只添加一个模块
            await self.ready_module_translation_tasks.put(module_name)
        await asyncio.create_task(
            self.module_translation_scheduler()
        )
        self.logger.info("All modules translated.")
        return AgentResponse.done(self, ProjectManagerResponseType.ALL_MODULES_DONE)

    async def module_translation_scheduler(self):
        """任务调度器，将就绪的任务分配给 Tech Leader
        """
        module_translation_future_map = {}
        while not all([module_translation.status in (ModuleTranslationStatus.DONE, ModuleTranslationStatus.FAILED) for
                       module_translation in
                       self.state_manager.translator.module_translations.values()]):
            try:
                module_name = await asyncio.wait_for(self.ready_module_translation_tasks.get(), timeout=1)
                module_translation_future = asyncio.create_task(
                    self.assign_module_to_tech_leader(module_name)
                )
                module_translation_future_map[module_translation_future] = module_name
            except asyncio.TimeoutError:
                ...

            # 检查并处理已经完成的任务
            done_futures = [module_translation for module_translation in module_translation_future_map.keys() if
                            module_translation.done()]
            for done_future in done_futures:
                module_translation_future_map.pop(done_future)

    async def assign_module_to_tech_leader(self, module_name: str):
        """分配模块转译任务给技术负责人"""
        if self.state_manager.translator.module_translations[module_name].status == "done":
            self.logger.info(f"Module translation [{module_name}] already done.")
            return
        tech_leader = TechLeader(self.llm_config, state_manager=self.state_manager,
                                 name=f"{TechLeader.ROLE}_{module_name.replace('/', '_')}")
        self.logger.info(f"Assign module translation [{module_name}] to {tech_leader.name} ...")
        try:
            agent_response = await asyncio.to_thread(
                tech_leader.start,
                module_name
            )
            if agent_response.status == AgentResponseStatus.DONE:
                self.state_manager.mark_module_translation_as(module_name, ModuleTranslationStatus.DONE)
                self.logger.info(f"Module translation [{module_name}] completed by {tech_leader.name}.")
            else:
                self.state_manager.mark_module_translation_as(module_name, ModuleTranslationStatus.FAILED)
                self.logger.error(
                    f"Module translation [{module_name}] failed by {tech_leader.name}.\n{agent_response.error}")
        except Exception as e:
            error_details = traceback.format_exc()
            self.logger.error(
                f"Module translation [{module_name}] failed by {tech_leader.name}.\nexception: {error_details}")
            self.state_manager.mark_module_translation_as(module_name,
                                                          ModuleTranslationStatus.FAILED)