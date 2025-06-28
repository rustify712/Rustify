from typing import Optional
from uuid import UUID
from datetime import datetime
from enum import Enum as PyEnum
from sqlalchemy import Enum, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import JSON, Text

from core.db.models.base import BaseTable
from core.db.models.project import ProjectEntity
from core.schema.translation import ModuleTranslationStatus, TranslationTaskStatus


class TranslationTaskEntity(BaseTable):
    __tablename__ = "translation_tasks"

    translation_id: Mapped[str] = mapped_column(ForeignKey("module_translations.id", ondelete="CASCADE"))
    """转译模块 ID"""
    source: Mapped[dict] = mapped_column(JSON)
    """待转译的 C 节点"""
    target: Mapped[Optional[dict]] = mapped_column(JSON, default=None)
    """转译后的 Rust 节点"""
    prerequisites: Mapped[list[str]] = mapped_column(JSON, default=[])
    """转译任务的前置任务的ID"""
    status: Mapped[TranslationTaskStatus] = mapped_column(
        Enum(TranslationTaskStatus), default=TranslationTaskStatus.CREATED)
    """转译任务执行状态"""
    create_time: Mapped[datetime] = mapped_column(server_default=func.now())
    """创建时间"""

class ModuleTranslationEntity(BaseTable):
    __tablename__ = "module_translations"

    project_id: Mapped[UUID] = mapped_column(ForeignKey("projects.id"))
    """项目ID"""
    name: Mapped[str] = mapped_column(Text, default="")
    """待转译模块名"""
    description: Mapped[str] = mapped_column(Text, default="")
    """模块介绍"""
    path: Mapped[str] = mapped_column(Text, default="")
    """模块路径"""
    related_files: Mapped[list[str]] = mapped_column(JSON, default=[])
    """相关文件列表"""
    status: Mapped[ModuleTranslationStatus] = mapped_column(
        Enum(ModuleTranslationStatus), default=ModuleTranslationStatus.CREATED
    )
    """模块转译状态"""
    related_rust_files: Mapped[list[str]] = mapped_column(JSON, default=[])
    """相关 Rust 文件列表"""
    # related_test_files: Mapped[list[str]] = mapped_column(JSON, default=[])
    # """相关测试文件列表"""
    # related_bench_files: Mapped[list[str]] = mapped_column(JSON, default=[])
    # """相关基准测试文件列表"""

    translation_tasks: Mapped[list[TranslationTaskEntity]] = relationship("TranslationTaskEntity", foreign_keys=[TranslationTaskEntity.translation_id])

class ProjectTranslationStatus(PyEnum):
    CREATED = "created"
    """已创建"""
    RUNNING = "running"
    """进行中"""
    DONE = "done"
    """已完成"""
    CANCELLED = "cancelled"
    """已取消"""
    FAILED = "failed"
    """已失败"""


class ProjectTranslation(BaseTable):
    __tablename__ = "project_translations"

    source_project_id: Mapped[UUID] = mapped_column(ForeignKey("projects.id"))
    """源项目ID"""
    source_lang: Mapped[str] = mapped_column()
    """源语言"""
    target_project_id: Mapped[UUID] = mapped_column(ForeignKey("projects.id"))
    """目标项目ID"""
    target_lang: Mapped[str] = mapped_column()
    """目标语言"""
    status: Mapped[ProjectTranslationStatus] = mapped_column(
        Enum(ProjectTranslationStatus), default=ProjectTranslationStatus.CREATED
    )
    """状态"""
    create_time: Mapped[datetime] = mapped_column(server_default=func.now())
    """创建时间"""

    source_project: Mapped[ProjectEntity] = relationship(ProjectEntity, foreign_keys=[source_project_id])
    """源项目"""
    target_project: Mapped[ProjectEntity] = relationship(ProjectEntity, foreign_keys=[target_project_id])
    """目标项目"""


class TranslationTable(BaseTable):
    """Translation table model"""
    __tablename__ = "translation_table"

    source_lang: Mapped[str] = mapped_column(Text, nullable=False)
    """源代码语言"""
    source_name: Mapped[str] = mapped_column(Text, nullable=True)
    """源代码名称"""
    source_text: Mapped[str] = mapped_column(Text, nullable=False)
    """源代码文本"""
    source_description: Mapped[str] = mapped_column(Text, nullable=False)
    """源代码描述"""
    source_module: Mapped[str] = mapped_column(Text, nullable=True)
    """源代码模块"""
    source_project: Mapped[str] = mapped_column(Text, nullable=True)
    """源代码项目"""

    target_lang: Mapped[str] = mapped_column(Text, nullable=False)
    """目标代码语言"""
    target_name: Mapped[str] = mapped_column(Text, nullable=True)
    """目标代码名称"""
    target_text: Mapped[str] = mapped_column(Text, nullable=False)
    """目标代码文本"""
    target_description: Mapped[str] = mapped_column(Text, nullable=False)
    """目标代码描述"""

    description: Mapped[str] = mapped_column(Text, nullable=False)
    """转译描述"""
    chat_messages: Mapped[list[dict]] = mapped_column(JSON, nullable=True)
    """
    聊天消息，存储为字典列表。
    示例：[
        {"role": "user", "message": "Hello!"},
        {"role": "assistant", "message": "Hi there!"}
    ]
    """
