import hashlib
import json
import os
import warnings
from contextlib import contextmanager
from enum import Enum
from typing import Callable, List, Literal, Optional
from collections import defaultdict
from threading import Lock

from pydantic import BaseModel

from core.schema.translation import TranslationTask
from core.version.vcs import VCS


class ProjectFile(BaseModel):
    type: Literal["file"]
    path: str
    content: Optional[str] = None
    summary: Optional[str] = None


class Project:
    DEFAULT_IGNORES = [".git", ".vcs", ".gitignore", "target", "Cargo.lock"]

    def __init__(
        self,
        name: str,
        path: str,
        description: Optional[str] = None,
        file_summaries: Optional[dict[str, str]] = None,
        **kwargs
    ):
        self.name = name
        self.path = path
        self.description = description
        self.file_summaries = file_summaries or {}
        self.details = kwargs

    def list_files(
        self,
        show_content: bool = False,
        show_summary: bool = False,
        ignore_func: Callable[[str], bool] = None
    ) -> List[ProjectFile]:
        """列出目录下的全部文件。

        Args:
            show_content: (bool) 是否显示文件内容。
            show_summary: (bool) 是否显示文件摘要。
            ignore_func: (Optional[Callable]) 忽略的文件或目录的函数。

        Returns:
            list[str]: 目录下的文件列表。
        """
        file_list = []
        for root, dirs, files in os.walk(self.path):
            for ignore_file in Project.DEFAULT_IGNORES:
                if ignore_file in dirs:
                    dirs.remove(ignore_file)
            for file in files:
                if os.path.basename(file) in Project.DEFAULT_IGNORES:
                    continue
                filepath = os.path.relpath(os.path.join(root, file), self.path)
                if ignore_func and ignore_func(filepath):
                    continue
                current_file = ProjectFile(
                    type="file",
                    path=filepath
                )
                if show_content:
                    with open(os.path.join(self.path, filepath), "r", encoding="utf-8") as f:
                        current_file.content = f.read()
                if show_summary:
                    current_file.summary = self.file_summaries.get(filepath, "")
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
            "name": self.name,
            "path": self.path,
            "description": self.description,
            "file_summaries": self.file_summaries,
            **self.details
        }


class RustProject(Project):
    def __init__(self, name: str, path: str, crate_type: str, description: Optional[str] = None, **kwargs):
        super().__init__(name, path, description, **kwargs)
        self.crate_type = crate_type

    def to_dict(self):
        return {
            **super().to_dict(),
            "crate_type": self.crate_type
        }


def generate_id(string, length=16):
    hash_object = hashlib.sha256(string.encode())
    unique_id = hash_object.hexdigest()
    return unique_id[:length]


class ModuleTranslationStatus(Enum):
    INIT = "init"
    TRANSPILE = "transpile"
    TEST = "test"
    BENCHMARK = "benchmark"
    DONE = "done"
    FAILED = "failed"

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value


class ModuleTranslation(BaseModel):
    """模块转译"""

    module_name: str
    """模块名"""
    module_description: str
    """模块介绍"""
    translation_tasks: list[TranslationTask]
    """转译任务"""
    related_c_files: List[str] = []
    """相关 C 文件"""
    related_rust_files: list[str] = []
    """相关 Rust 文件"""
    related_test_files: list[str] = []
    """相关测试文件"""
    related_bench_files: list[str] = []
    """相关基准测试文件"""
    status: ModuleTranslationStatus = ModuleTranslationStatus.INIT
    """状态"""

    class Config:
        json_encoders = {
            ModuleTranslationStatus: lambda v: v.value
        }

    def get_translation_task_by_task_id(self, task_id: str):
        for translation_task in self.translation_tasks:
            if translation_task.source.id == task_id:
                return translation_task
        return None


class Translator(BaseModel):
    module_translations: dict[str, ModuleTranslation] = {}
    module_infos: dict[str, dict] = defaultdict(dict)
    """模块名 -> 模块转译"""

    def add_module_translation(self, module_name: str, module_translation_tasks: list[TranslationTask],
                               status: Literal["init", "done", "pending"] = "init", info: dict = None):
        related_c_files = set(
            node.filepath
            for task in module_translation_tasks
            for node in task.source.nodes
        )
        self.module_translations[module_name] = ModuleTranslation(
            module_name=module_name,
            module_description="",
            translation_tasks=module_translation_tasks,
            related_c_files=list(related_c_files),
            status=status
        )
        self.module_infos[module_name] = info or {}


    def add_module_rust_files(self, module_name: str, rust_files: list[str]):
        for rust_file in rust_files:
            if rust_file not in self.module_translations[module_name].related_rust_files:
                self.module_translations[module_name].related_rust_files.append(rust_file)

    def add_module_test_files(self, module_name: str, test_files: list[str]):
        for test_file in test_files:
            if test_file not in self.module_translations[module_name].related_test_files:
                self.module_translations[module_name].related_test_files.append(test_file)

    def add_module_bench_files(self, module_name: str, bench_files: list[str]):
        for bench_file in bench_files:
            if bench_file not in self.module_translations[module_name].related_bench_files:
                self.module_translations[module_name].related_bench_files.append(bench_file)

    @property
    def modules(self):
        return list(self.module_translations.keys())

    @property
    def ready_modules(self):
        """获取准备好的模块"""
        return [
            module
            for module, module_translation in self.module_translations.items()
            if module_translation.status != ModuleTranslationStatus.DONE
        ]


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


class StateManager:
    """
    TODO: 使用本地数据库
    """

    def __init__(self):
        self.bound_filepath: Optional[str] = None
        # C/C++ 项目
        self.source_project: Optional[Project] = None
        # Rust 项目
        self.target_project: Optional[Project] = None
        self.target_vcs: Optional[VCS] = None
        # 转换器
        self.translator: Translator = Translator()
        # 文件锁管理器
        self.file_lock_manager = FileLockManager()

    def update(self):
        """更新状态并写入文件。"""
        if self.bound_filepath:
            with self.file_lock_manager.file_lock(self.bound_filepath):
                self.dump(self.bound_filepath)
        else:
            warnings.warn("State file not bound, state not saved.")

    def mark_module_translation_as(self, module_name: str, status: ModuleTranslationStatus):
        if module_name not in self.translator.module_translations:
            raise ValueError(f"Module {module_name} not found")
        self.translator.module_translations[module_name].status = status
        self.update()

    def mark_translation_task_as(self, module_name: str, task_id: str,
                                 status: Literal["init", "running", "completion", "done", "failed"]):
        if module_name not in self.translator.module_translations:
            raise ValueError(f"Module {module_name} not found")
        module_translation = self.translator.module_translations[module_name]
        translation_task = module_translation.get_translation_task_by_task_id(task_id)
        if not translation_task:
            raise ValueError(f"Translation task {task_id} not found in module {module_name}")
        translation_task.status = status
        self.update()

    # def set_benchmark_module(self, module: str, benchmark_file: str):
    #     self.module_benchmarks[module] = benchmark_file
    #     if self.bound_filepath:
    #         self.dump(self.bound_filepath)

    def add_module_rust_files(self, module_name, related_rust_files: list[str]):
        self.translator.add_module_rust_files(module_name, related_rust_files)
        self.update()

    def add_module_test_files(self, module_name, related_test_files: list[str]):
        self.translator.add_module_test_files(module_name, related_test_files)
        self.update()

    def add_module_bench_files(self, module_name, related_bench_files: list[str]):
        self.translator.add_module_bench_files(module_name, related_bench_files)
        self.update()

    def add_module_translation(self, module_name: str, module_translation_tasks: list[TranslationTask], **kwargs):
        self.translator.add_module_translation(module_name, module_translation_tasks, **kwargs)
        self.update()

    def bind_filepath(self, filepath: str):
        self.bound_filepath = filepath

    def set_source_project_description(self, description: str):
        if self.source_project:
            self.source_project.description = description
            self.update()

    def set_source_project_file_summaries(self, file_summaries: dict[str, str]):
        if self.source_project:
            self.source_project.file_summaries = file_summaries
            self.update()

    def load_source_project(self, project_name: str, project_dir: str):
        """
        创建项目
        """
        self.source_project = Project(
            name=project_name,
            path=project_dir
        )
        if self.bound_filepath:
            self.dump(self.bound_filepath)
        return self.source_project

    def create_rust_project(self, project_name: str, project_dir: str, crate_type: str, project_description: str):
        """
        创建项目
        """
        self.target_project = RustProject(
            name=project_name,
            path=project_dir,
            description=project_description,
            crate_type=crate_type
        )
        self.target_vcs = VCS(project_dir)
        self.target_vcs.init()
        self.target_vcs.add([
            os.path.join(project_dir, file.path)
            for file in self.target_project.list_files()
        ])
        self.target_vcs.commit("Initial commit")
        if self.bound_filepath:
            self.dump(self.bound_filepath)
        return self.target_project

    def dump(self, filepath: str):
        """保存状态到文件。"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "source_project": self.source_project.to_dict() if self.source_project else None,
                "target_project": self.target_project.to_dict() if self.target_project else None,
                "translator": json.loads(self.translator.model_dump_json()),
            }, ensure_ascii=False, indent=4))

    def load(self, filepath: str):
        """从文件加载状态。"""
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            if not content:
                return
            state = json.loads(content)
            self.source_project = Project(**state["source_project"]) if "source_project" in state and state[
                "source_project"] else None
            self.target_project = Project(**state["target_project"]) if "target_project" in state and state[
                "target_project"] else None
            self.translator = Translator.model_validate(state["translator"]) if "translator" in state and state[
                "translator"] else Translator()
            if self.target_project:
                self.target_vcs = VCS(self.target_project.path)
