from __future__ import annotations

from decimal import Decimal
from uuid import UUID

from sqlalchemy import CheckConstraint, ForeignKey, Index, Numeric, Text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import Uuid

from src.connections.database.base import Base


class Winery(Base):
    __tablename__ = "wineries"

    winery_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    address: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    vineyard_area: Mapped[str | None] = mapped_column(Text)
    region: Mapped[str | None] = mapped_column(Text, index=True)
    locality: Mapped[str | None] = mapped_column(Text)
    climate: Mapped[str | None] = mapped_column(Text)

    wines: Mapped[list[Wine]] = relationship(back_populates="winery")


class Wine(Base):
    __tablename__ = "wines"
    __table_args__ = (
        CheckConstraint("rating IS NULL OR rating BETWEEN 0 AND 5", name="ck_wines_rating_range"),
        Index("ix_wines_name_producer", "name", "producer"),
    )

    wine_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    winery_id: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("wineries.winery_id", ondelete="SET NULL"),
        index=True,
    )
    name: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    producer: Mapped[str | None] = mapped_column(Text, index=True)
    rating: Mapped[Decimal | None] = mapped_column(Numeric(3, 2))
    color: Mapped[str | None] = mapped_column(Text)
    wine_type: Mapped[str | None] = mapped_column(Text, index=True)
    region: Mapped[str | None] = mapped_column(Text, index=True)
    grape_varieties: Mapped[list[str]] = mapped_column(
        ARRAY(Text),
        nullable=False,
        default=list,
    )
    shade: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    serving_temperature: Mapped[str | None] = mapped_column(Text)
    alcohol: Mapped[str | None] = mapped_column(Text)
    food_pairings: Mapped[list[str]] = mapped_column(
        ARRAY(Text),
        nullable=False,
        default=list,
    )

    winery: Mapped[Winery | None] = relationship(back_populates="wines")
