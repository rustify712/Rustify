import json
import uuid
from enum import Enum as PyEnum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class TranslationUnitNode(BaseModel):
    filepath: str = Field(description="转译节点所在文件路径")
    id: str = Field(description="节点 ID")
    name: str = Field(description="节点名称")
    type: str = Field(description="节点类型")
    text: str = Field(description="节点代码")


class TranslationTaskSource(BaseModel):
    name: str = Field(description="待转译的 C/C++ 节点名称")
    nodes: List[TranslationUnitNode] = Field(description="待转译的 C/C++ 节点")
    description: Optional[str] = Field(description="节点描述", default="")

class TranslationTaskTargetItem(BaseModel):
    name: str = Field(description="转译后的 Rust 节点名称")
    type: str = Field(description="转译后的 Rust 节点类型")
    text: str = Field(description="转译后的 Rust 代码")
    description: Optional[str] = Field(description="节点描述", default="")
    filepath: Optional[str] = Field(description="转译后的 Rust 代码所在文件路径", default="")

class TranslationTaskTarget(BaseModel):
    files: List[TranslationTaskTargetItem]

class TranslationTaskStatus(PyEnum):
    CREATED = "created"
    """任务被创建"""
    TRANSLATING = "translating"
    """执行转译中"""
    FIXING = "fixing"
    """转译代码修复中"""
    DONE = "done"
    """转译任务完成"""
    FAILED = "failed"
    """转译任务失败"""

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value

class TranslationTask(BaseModel):
    id: str = Field(description="转译任务 ID", default_factory=lambda: str(uuid.uuid4()))
    source: TranslationTaskSource = Field(description="待转译的 C 节点")
    target: Optional[TranslationTaskTarget] = Field(description="转译后的 Rust 节点")
    prerequisites: list[str] = Field(description="转译任务的前置任务的ID", default=[])
    status: TranslationTaskStatus = Field(default=False, description="转译任务执行状态")

    class Config:
        json_encoders = {
            TranslationTaskStatus: lambda v: v.value
        }

    def __hash__(self):
        return hash(self.source.id)

class ModuleTranslationStatus(PyEnum):
    CREATED = "created"
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
    id: str = Field(description="模块 ID", default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(description="待转译模块名", default="")
    description: str = Field(description="模块介绍", default="")
    path: str = Field(description="模块路径", default="")
    translation_tasks: list[TranslationTask] = Field(description="转译任务列表", default=[])
    related_files: list[str] = Field(description="相关文件列表")
    status: ModuleTranslationStatus = Field(description="模块转译状态", default=ModuleTranslationStatus.CREATED)

    related_rust_files: list[str] = Field(description="相关 Rust 文件列表", default=[])

    class Config:
        json_encoders = {
            ModuleTranslationStatus: lambda v: v.value
        }
        # NOTE: 允许 DGNode 类型（非 BaseModel 类）
        arbitrary_types_allowed = True

    def get_translation_task_by_id(self, translation_task_id: str):
        for translation_task in self.translation_tasks:
            if translation_task.id == translation_task_id:
                return translation_task
        return None