# https://github.com/Pythagora-io/gpt-pilot/blob/main/core/db/models/base.py
# DeclarativeBase enables declarative configuration of
# database models within SQLAlchemy.
#
# It also sets up a registry for the classes that inherit from it,
# so that SQLAlechemy understands how they map to database tables.
from typing import Optional, Self
from uuid import UUID, uuid4

from sqlalchemy import MetaData, select, delete
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, mapped_column, Mapped, ColumnProperty
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.types import JSON


class BaseTable(AsyncAttrs, DeclarativeBase):
    """Base class for all SQL database models."""
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)

    # Mapping of Python types to SQLAlchemy types.
    type_annotation_map = {
        list[dict]: JSON,
        list[str]: JSON,
        dict: JSON,
    }

    metadata = MetaData(
        # Naming conventions for constraints, foreign keys, etc.
        # ix: Index
        # uq: Unique constraint
        # ck: Check constraint
        # fk: Foreign key
        # pk: Primary key
        naming_convention={
            "ix": "ix_%(column_0_label)s",
            "uq": "uq_%(table_name)s_%(column_0_name)s",
            "ck": "ck_%(table_name)s_`%(constraint_name)s`",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        }
    )

    def __repr__(self) -> str:
        """Return a string representation of the model with all attributes."""
        # 获取所有列并生成属性=值对
        column_values = {
            prop.key: getattr(self, prop.key)
            for prop in self.__mapper__.iterate_properties
            if isinstance(prop, ColumnProperty)
        }

        # 格式化为字符串
        columns_repr = ", ".join(f"{key}={value}" for key, value in column_values.items())
        return f"<{self.__class__.__name__}({columns_repr})>"

    @classmethod
    async def get_by_id(cls, session: "AsyncSession", tid: str) -> Optional[Self]:
        if not isinstance(tid, UUID):
            tid = UUID(tid)
        result = await session.execute(select(cls).where(cls.id == tid))
        return result.scalar_one_or_none()

    @classmethod
    async def delete_by_id(cls, session: "AsyncSession", tid: str) -> int:
        if not isinstance(tid, UUID):
            tid = UUID(tid)
        result = await session.execute(delete(cls).where(cls.id == tid))
        await session.commit()
        return result.rowcount()
