from datetime import datetime
from uuid import UUID

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from core.db.models.base import BaseTable


class FileContentEntity(BaseTable):
    __tablename__ = "file_contents"

    file_id: Mapped[UUID] = mapped_column(ForeignKey("files.id", ondelete="CASCADE", use_alter=True))
    """文件ID"""
    content: Mapped[str] = mapped_column()
    """文件内容"""
    summary: Mapped[str] = mapped_column()
    """文件内容摘要"""
    create_time: Mapped[datetime] = mapped_column(server_default=func.now())
    """创建时间"""
    update_time: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())
    """更新时间"""

    # Relationships
    file: Mapped["FileEntity"] = relationship("FileEntity", foreign_keys=[file_id])
    """文件"""


class FileEntity(BaseTable):
    __tablename__ = "files"

    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    """项目ID"""
    path: Mapped[str] = mapped_column()
    """文件路径"""
    meta: Mapped[dict] = mapped_column(default=dict, server_default="{}")
    """元数据"""
    create_time: Mapped[datetime] = mapped_column(server_default=func.now())
    """创建时间"""
    update_time: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())
    """更新时间"""

    # Relationships
    project: Mapped["ProjectEntity"] = relationship("ProjectEntity", back_populates="files", foreign_keys=[project_id])
    """项目"""
    content: Mapped[FileContentEntity] = relationship("FileContentEntity", back_populates="file", foreign_keys=[FileContentEntity.file_id])
    """文件内容"""


class ProjectEntity(BaseTable):
    __tablename__ = "projects"

    name: Mapped[str] = mapped_column()
    """项目名称"""
    description: Mapped[str] = mapped_column()
    """项目描述"""
    dirpath: Mapped[str] = mapped_column()
    """项目目录路径"""
    create_time: Mapped[datetime] = mapped_column(server_default=func.now())
    """创建时间"""

    # Relationships
    files: Mapped[list[FileEntity]] = relationship("FileEntity", foreign_keys=[FileEntity.project_id])
    """文件列表"""
