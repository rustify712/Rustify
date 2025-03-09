from uuid import UUID
from datetime import datetime
from enum import Enum as PyEnum
from sqlalchemy import Enum, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import JSON, Text

from core.db.models.base import BaseTable
from core.db.models.project import Project


class ProjectTranslationStatus(PyEnum):
    PENDING = "pending"
    """待处理"""
    RUNNING = "running"
    """进行中"""
    COMPLETED = "completed"
    """已完成"""
    CANCELLED = "cancelled"
    """已取消"""
    FAULTED = "faulted"
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
        Enum(ProjectTranslationStatus), default=ProjectTranslationStatus.PENDING
    )
    """状态"""
    create_time: Mapped[datetime] = mapped_column(server_default=func.now())
    """创建时间"""

    source_project: Mapped[Project] = relationship(Project, foreign_keys=[source_project_id])
    """源项目"""
    target_project: Mapped[Project] = relationship(Project, foreign_keys=[target_project_id])
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
