from datetime import datetime
from uuid import UUID

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from core.db.models.base import BaseTable


class FileContent(BaseTable):
    __tablename__ = "file_contents"

    file_id: Mapped[UUID] = mapped_column(ForeignKey("files.id", ondelete="CASCADE", use_alter=True))
    """文件ID"""
    content: Mapped[str] = mapped_column()
    """文件内容"""
    create_time: Mapped[datetime] = mapped_column(server_default=func.now())
    """创建时间"""
    update_time: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())
    """更新时间"""

    # Relationships
    file: Mapped["File"] = relationship("File", foreign_keys=[file_id])
    """文件"""


class File(BaseTable):
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
    project: Mapped["Project"] = relationship("Project", back_populates="files", foreign_keys=[project_id])
    """项目"""
    content: Mapped[FileContent] = relationship("FileContent", back_populates="file", foreign_keys=[FileContent.file_id])
    """文件内容"""


class Project(BaseTable):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(primary_key=True)
    """项目ID"""
    name: Mapped[str] = mapped_column()
    """项目名称"""
    description: Mapped[str] = mapped_column()
    """项目描述"""
    dirpath: Mapped[str] = mapped_column()
    """项目目录路径"""
    create_time: Mapped[datetime] = mapped_column(server_default=func.now())
    """创建时间"""

    # Relationships
    files: Mapped[list[File]] = relationship("File", foreign_keys=[File.project_id])
    """文件列表"""
