import copy
import json
import os
import uuid
from contextlib import contextmanager
from enum import Enum
from typing import Callable, List, Literal, Optional
from collections import defaultdict
from threading import Lock

from pydantic import BaseModel

from core.graph.dep_graph import DGNode
from core.schema.translation import ModuleTranslation, ModuleTranslationStatus, TranslationTask, TranslationTaskSource, \
    TranslationTaskStatus, TranslationTaskTarget, TranslationUnitNode
from core.utils.file_utils import add_line_numbers
from core.utils.vfs import VirtualFileSystem


class ProjectFile(BaseModel):
    type: Literal["file"]
    path: str
    content: Optional[str] = None
    summary: Optional[str] = None


class Project:
    DEFAULT_IGNORES = [".git", ".vcs", ".gitignore", "target", "Cargo.lock"]

    def __init__(
            self,
            id: str,
            name: str,
            path: str,
            description: Optional[str] = None,
            file_summaries: Optional[dict[str, str]] = None,
            **kwargs
    ):
        self.id = id
        self.name = name
        self.path = path
        self.description = description
        self.file_summaries = file_summaries or {}
        self.details = kwargs

    def list_files(
            self,
            show_content: bool = False,
            show_summary: bool = False,
            show_line_numbers: bool = False,
            ignore_func: Callable[[str], bool] = None,
            relpath: Optional[str] = None
    ) -> List[ProjectFile]:
        """列出目录下的全部文件。

        Args:
            show_content: (bool) 是否显示文件内容。
            show_summary: (bool) 是否显示文件摘要。
            show_line_numbers: (bool) 是否显示行号。
            ignore_func: (Optional[Callable]) 忽略的文件或目录的函数。
            relpath: (Optional[str]) 相对路径。

        Returns:
            list[str]: 目录下的文件列表。
        """
        path = self.path
        file_list = []
        for root, dirs, files in os.walk(path):
            for ignore_file in Project.DEFAULT_IGNORES:
                if ignore_file in dirs:
                    dirs.remove(ignore_file)
            if relpath:
                if not root.startswith(relpath):
                    continue
            # 若 relpath 存在，那么此时 root 是 relpath 或 relpath 的子目录
            for file in files:
                if os.path.basename(file) in Project.DEFAULT_IGNORES:
                    continue
                abs_filepath = os.path.join(root, file)
                if relpath:
                    filepath = os.path.relpath(os.path.join(root, file), relpath)
                else:
                    filepath = os.path.relpath(os.path.join(root, file), self.path)
                # NOTE: filepath 是相对路径
                if ignore_func and ignore_func(filepath):
                    continue
                current_file = ProjectFile(
                    type="file",
                    path=filepath
                )
                if show_content:
                    with open(abs_filepath, "r", encoding="utf-8") as f:
                        file_content = f.read()
                    if show_line_numbers:
                        current_file.content = add_line_numbers(file_content)
                    else:
                        current_file.content = file_content
                if show_summary:
                    # 这里必须是相对于项目根目录的路径
                    current_file.summary = self.file_summaries.get(os.path.relpath(abs_filepath, self.path), "")
                file_list.append(current_file)
        return file_list

    def pretty_structure(self, ignore_func: Callable[[str], bool] = None) -> str:
        """返回项目的结构。
        """

        def inner_list_files(dirpath: str):
            file_list = []
            for entry in os.listdir(dirpath):
                if os.path.basename(entry) in Project.DEFAULT_IGNORES:
                    continue
                path = os.path.join(dirpath, entry)
                filepath = os.path.relpath(os.path.join(dirpath, entry), self.path)
                if ignore_func and ignore_func(filepath):
                    continue
                current_file = {
                    "path": filepath,
                    "children": []
                }
                if os.path.isdir(path):
                    current_file["type"] = "dir"
                    current_file["children"] = inner_list_files(path)  # 递归调用，增加缩进
                else:
                    current_file["type"] = "file"
                file_list.append(current_file)
            return file_list

        def inner_pretty_files(node: dict, indent: int = 0):
            nonlocal file_structure_str
            # 添加当前节点信息
            file_structure_str += " " * indent + f"[{node['type'].upper()}] {node['path']}\n"
            # 如果是目录，递归显示子节点
            if node["type"] == "dir":
                for child in node["children"]:
                    inner_pretty_files(child, indent + 2)

        file_structure = {
            "path": os.path.basename(self.path),
            "type": "dir",
            "children": inner_list_files(self.path)
        }
        file_structure_str = ""
        inner_pretty_files(file_structure)
        return file_structure_str

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "path": self.path,
            "description": self.description,
            "file_summaries": self.file_summaries,
            **self.details
        }

    @property
    def src_path(self):
        return os.path.join(self.path, "src")

    @property
    def test_path(self):
        return os.path.join(self.path, "test")


class TargetProject(Project):
    def __init__(
            self,
            id: str,
            name: str,
            path: str,
            description: Optional[str] = None,
            **kwargs
    ):
        super().__init__(id, name, path, description, **kwargs)

    @property
    def test_path(self):
        return os.path.join(self.path, "tests")

    def pretty_structure(self, ignore_func: Callable[[str], bool] = None) -> str:
        """返回项目的结构。
        """

        def inner_list_files(dirpath: str):
            file_list = []
            for entry in os.listdir(dirpath):
                if os.path.basename(entry) in Project.DEFAULT_IGNORES:
                    continue
                path = os.path.join(dirpath, entry)
                filepath = os.path.relpath(os.path.join(dirpath, entry), self.path)
                if ignore_func and ignore_func(filepath):
                    continue
                current_file = {
                    "path": filepath,
                    "children": []
                }
                if os.path.isdir(path):
                    current_file["type"] = "dir"
                    current_file["children"] = inner_list_files(path)  # 递归调用，增加缩进
                else:
                    current_file["type"] = "file"
                file_list.append(current_file)
            return file_list

        def inner_pretty_files(node: dict, indent: int = 0):
            nonlocal file_structure_str
            # 添加当前节点信息
            file_structure_str += " " * indent + f"[{node['type'].upper()}] {node['path']}\n"
            # 如果是目录，递归显示子节点
            if node["type"] == "dir":
                for child in node["children"]:
                    inner_pretty_files(child, indent + 2)

        file_structure = {
            "path": os.path.basename(self.path),
            "type": "dir",
            "children": inner_list_files(self.path)
        }
        file_structure_str = ""
        inner_pretty_files(file_structure)
        return file_structure_str

    def to_dict(self):
        return {
            **super().to_dict(),
        }


class FileLockManager:

    def __init__(self):
        self.file_locks: dict[str, Lock] = {}

    @contextmanager
    def file_lock(self, filepath: str):
        if filepath not in self.file_locks:
            lock = Lock()
            self.file_locks[filepath] = lock
        else:
            lock = self.file_locks[filepath]
        lock.acquire()
        try:
            yield
        finally:
            lock.release()

    def is_lock(self, filepath: str):
        if filepath not in self.file_locks:
            return False
        else:
            lock = self.file_locks[filepath]
            return lock.locked()

    def acquire_file_lock(self, filepath: str):
        if filepath not in self.file_locks:
            lock = Lock()
            self.file_locks[filepath] = lock
        else:
            lock = self.file_locks[filepath]
        lock.acquire()

    def release_file_lock(self, filepath: str):
        if filepath not in self.file_locks:
            raise ValueError(f"File lock for {filepath} not found")
        lock = self.file_locks[filepath]
        lock.release()


class State:

    def __init__(self):
        self.project_translation_id: Optional[str] = None
        self.source_project: Optional[Project] = None
        self.target_project: Optional[TargetProject] = None

        self.module_translations: list[ModuleTranslation] = []

    @property
    def ready_module_translations(self):
        """获取准备好的模块"""
        return [
            (index, module_translation)
            for index, module_translation in enumerate(self.module_translations)
            if module_translation.status not in [ModuleTranslationStatus.DONE, ModuleTranslationStatus.FAILED]
        ]

    def to_dict(self):
        return {
            "project_translation_id": self.project_translation_id,
            "source_project": self.source_project.to_dict() if self.source_project else None,
            "target_project": self.target_project.to_dict() if self.target_project else None,
            "module_translations": [module_translation.model_dump() for module_translation in self.module_translations]
        }

    def load_from_json(self, json_str: str, file_system: Optional[VirtualFileSystem] = None):
        state = json.loads(json_str)
        if "project_translation_id" in state and state["project_translation_id"]:
            self.project_translation_id = state["project_translation_id"]
        if "source_project" in state and state["source_project"]:
            self.source_project = Project(**state["source_project"])
        if "target_project" in state and state["target_project"]:
            self.target_project = TargetProject(**state["target_project"], file_system=file_system)
        if "module_translations" in state and state["module_translations"]:
            self.module_translations = [
                ModuleTranslation(**module_translation)
                for module_translation in state["module_translations"]
            ]


def optimize_translation_units(nodes: list[tuple[int, DGNode]]):
    """优化转译单元。"""

    def can_merge(current_nodes_line_count: int, node_line_count: int, target: int = 20):
        """判断节点是否可以合并。

        Args:
            current_nodes_line_count: 当前节点组总行数
            node_line_count: 待合并节点行数
            target: 合并目标行数，默认 20
        """
        return current_nodes_line_count + node_line_count <= target

    def find_best_combination(remaining_nodes: list[tuple[int, DGNode]], start_idx: int,
                              nodes_group_line_count: int = 0, nodes_group: Optional[list[DGNode]] = None):
        """寻找最佳组合。

        Args:
            remaining_nodes: 剩余节点
            start_idx: 起始索引
            nodes_group_line_count: 当前组行数
            nodes_group: 当前组
        """
        best_result = None  # 存储最优结果
        best_result_size = float('inf')  # 存储最少的组数

        # 初始化当前组（第一次调用）
        if nodes_group is None:
            nodes_group = []
        if start_idx >= len(remaining_nodes):
            return [nodes_group]
        # 尝试将当前节点加入当前组
        if can_merge(nodes_group_line_count, remaining_nodes[start_idx][0]):
            new_nodes_group = nodes_group + [remaining_nodes[start_idx][1]]
            # 递归调用，尝试将下一个节点加入当前组
            new_nodes_group_line_count = nodes_group_line_count + remaining_nodes[start_idx][0]
            result1 = find_best_combination(remaining_nodes, start_idx + 1, new_nodes_group_line_count, new_nodes_group)
            if result1 and len(result1) < best_result_size:
                best_result = result1
                best_result_size = len(result1)

        # 尝试不将当前节点加入当前组
        new_nodes_group = [remaining_nodes[start_idx][1]]
        new_nodes_group_line_count = remaining_nodes[start_idx][0]
        result2 = [nodes_group] + find_best_combination(remaining_nodes, start_idx + 1, new_nodes_group_line_count,
                                                        new_nodes_group)
        if len(result2) < best_result_size:
            best_result = result2
            best_result_size = len(result2)

        return best_result

    best_combination = find_best_combination(nodes, 0)
    return best_combination


class StateManager:

    def __init__(self, filepath: str):
        self.bound_filepath = filepath
        self.state = State()

        self._lock = Lock()

        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                if content:
                    self.state.load_from_json(content)

    async def create_source_project(self, project_dir: str):
        self.state.source_project = Project(
            id=str(uuid.uuid4()),
            name=os.path.basename(project_dir),
            path=project_dir,
            description=""
        )
        self.sync_to_disk()

    async def create_target_project(self, name: str, dirpath: str, description: str):
        os.makedirs(dirpath, exist_ok=True)
        self.state.target_project = TargetProject(
            id=str(uuid.uuid4()),
            name=name,
            path=dirpath,
            description=description
        )

        self.sync_to_disk()

    async def save_source_project_file(self, filepath: str, content: str, summary: str = ""):
        if not self.state.source_project:
            raise ValueError("source project not loaded")
        self.state.source_project.file_summaries[filepath] = summary
        self.sync_to_disk()

    async def update_source_project_description(self, description: str):
        self.state.source_project.description = description
        self.sync_to_disk()

    async def create_module_translation(
            self,
            translation_units_list: list[list[list[DGNode]]],
            related_files: list[str]
    ):
        translation_tasks = []
        # node_id -> task_id
        node_task_lookup_map = {}
        # 记录每个任务的所有节点的依赖节点
        prerequisites_nodes = defaultdict(list)
        # 遍历所有转译的转译单元组
        is_first = True
        all_nodes = []
        for translation_units in translation_units_list:
            # 将所有没有依赖的节点构建成一个任务，即第一个可并行转译的转译单元
            if is_first:
                is_first = False
                all_dep_nodes = [
                    node
                    for translation_unit in translation_units
                    for node in translation_unit
                ]
                translation_task = TranslationTask(
                    source=TranslationTaskSource(
                        name="init",
                        nodes=[
                            TranslationUnitNode(
                                filepath=os.path.relpath(node.location, self.state.source_project.path),
                                id=node.id,
                                name=node.name,
                                type=node.type,
                                text=node.text,
                            )
                            for node in all_dep_nodes
                        ],
                        description=""
                    ),
                    target=None,
                    status=TranslationTaskStatus.CREATED,
                    prerequisites=[]
                )
                prerequisites_nodes[translation_task.id] = [
                    edge.dst.id
                    for node in all_dep_nodes
                    for edge in node.edges
                    if edge.dst.id not in [n.id for n in all_dep_nodes]
                ]
                translation_tasks.append(translation_task)
                for node in all_dep_nodes:
                    node_task_lookup_map[node.id] = translation_task.id
            else:
                # 遍历所有可并行转译的转译单元，合并代码长度不超过 20 行的同类型的转译单元
                to_merge_nodes = defaultdict(list)
                new_translation_units = []
                for translation_unit in translation_units:
                    line_count = sum([len(node.text.split("\n")) for node in translation_unit])
                    if len(translation_unit) == 1 and line_count <= 20:
                        to_merge_nodes[translation_unit[0].type].append((line_count, translation_unit[0]))
                    else:
                        new_translation_units.append(translation_unit)
                for nodes in to_merge_nodes.values():
                    translation_units = optimize_translation_units(nodes)
                    new_translation_units.extend(translation_units)
                # 遍历所有转译单元
                # TODO: 新的转译单元中顺序可能发生变化，这里需要进行排序
                for translation_unit in new_translation_units:
                    translation_unit_node_ids = [node.id for node in translation_unit]
                    all_nodes.extend([
                        node
                        for node in translation_unit
                    ])
                    translation_task = TranslationTask(
                        source=TranslationTaskSource(
                            name="_".join(set([node.name for node in translation_unit]))[:50],
                            nodes=[
                                TranslationUnitNode(
                                    filepath=os.path.relpath(node.location, self.state.source_project.path),
                                    id=node.id,
                                    name=node.name,
                                    type=node.type,
                                    text=node.text,
                                )
                                for node in translation_unit
                            ],
                            description=""
                        ),
                        target=None,
                        status=TranslationTaskStatus.CREATED,
                        prerequisites=[]
                    )
                    prerequisites_nodes[translation_task.id] = list(set(
                        edge.dst.id
                        for node in translation_unit
                        for edge in node.edges
                        if edge.dst.id not in translation_unit_node_ids
                    ))
                    translation_tasks.append(translation_task)
                    for node in translation_unit:
                        node_task_lookup_map[node.id] = translation_task.id
        # 完善依赖关系
        for translation_task in translation_tasks:
            translation_task.prerequisites = list(set([
                node_task_lookup_map[node_id]
                for node_id in prerequisites_nodes[translation_task.id]
            ]))

        # 保存模块转译
        module_translation = ModuleTranslation(
            translation_tasks=translation_tasks,
            related_files=[os.path.relpath(file, self.state.source_project.path) for file in related_files],
            status=ModuleTranslationStatus.CREATED
        )
        self.state.module_translations.append(module_translation)
        self.sync_to_disk()

    async def update_module_translation_info(
            self,
            module_translation_id: str,
            name: str,
            description: str,
    ):
        module_translation_path = os.path.join(self.state.target_project.path, name)
        module_translation = self.get_module_translation_by_id(module_translation_id)
        if module_translation:
            module_translation.name = name
            module_translation.description = description
            module_translation.path = module_translation_path

            self.sync_to_disk()

    async def update_module_translation_status(self, module_translation_id: str, status: ModuleTranslationStatus):
        module_translation = self.get_module_translation_by_id(module_translation_id)
        if module_translation:
            module_translation.status = status
            self.sync_to_disk()
        if status == ModuleTranslationStatus.DONE or status == ModuleTranslationStatus.FAILED:
            # 备份这个 state 文件
            back_filepath = os.path.join(
                os.path.dirname(self.bound_filepath),
                module_translation.name + "_state.json"
            )
            with open(back_filepath, "w", encoding="utf-8") as f:
                back_state = copy.deepcopy(self.state)
                back_state.module_translations = [module_translation]
                f.write(json.dumps(back_state.to_dict(), ensure_ascii=False, indent=4, cls=self.EnumEncoder))

    async def update_translation_task_status(
            self,
            module_translation_id: str,
            translation_task_id: str,
            status: TranslationTaskStatus
    ):
        module_translation = self.get_module_translation_by_id(module_translation_id)
        if module_translation:
            translation_task = module_translation.get_translation_task_by_id(translation_task_id)
            if translation_task:
                translation_task.status = status
                self.sync_to_disk()

    def get_module_translation_by_id(self, module_translation_id: str):
        for module_translation in self.state.module_translations:
            if module_translation.id == module_translation_id:
                return module_translation
        return None

    async def add_module_translation_related_rust_files(self, module_translation_id: str,
                                                        related_rust_files: list[str]):
        module_translation = self.get_module_translation_by_id(module_translation_id)
        if module_translation:
            for rust_file in related_rust_files:
                if rust_file not in module_translation.related_rust_files:
                    module_translation.related_rust_files.append(rust_file)

            self.sync_to_disk()

    async def set_translation_task_target(self, module_translation_id: str, translation_task_id: str,
                                          target: TranslationTaskTarget):
        module_translation = self.get_module_translation_by_id(module_translation_id)
        if module_translation:
            translation_task = module_translation.get_translation_task_by_id(translation_task_id)
            if translation_task:
                translation_task.target = target

                self.sync_to_disk()

    class EnumEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, Enum):
                return obj.value
            return super().default(obj)

    def sync_to_disk(self):
        if os.path.dirname(self.bound_filepath) != "":
            os.makedirs(os.path.dirname(self.bound_filepath), exist_ok=True)
        with self._lock:
            with open(self.bound_filepath, "w", encoding="utf-8") as f:
                f.write(json.dumps(self.state.to_dict(), ensure_ascii=False, indent=4, cls=self.EnumEncoder))
