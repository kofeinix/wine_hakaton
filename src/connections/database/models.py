from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import Uuid

from src.connections.database.base import Base


class Producer(Base):
    __tablename__ = "producers"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)

    wines: Mapped[list[Wine]] = relationship(back_populates="producer")


class Region(Base):
    __tablename__ = "regions"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    country: Mapped[str | None] = mapped_column(String(255), index=True)

    wines: Mapped[list[Wine]] = relationship(back_populates="region")


class Grape(Base):
    __tablename__ = "grapes"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)

    wine_links: Mapped[list[WineGrape]] = relationship(back_populates="grape")
    aliases: Mapped[list[GrapeAlias]] = relationship(
        back_populates="grape",
        cascade="all, delete-orphan",
    )


class GrapeAlias(Base):
    __tablename__ = "grape_aliases"

    grape_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("grapes.id", ondelete="CASCADE"),
        nullable=False,
        primary_key=True,
        index=True,
    )
    alias: Mapped[str] = mapped_column(String(255), nullable=False, primary_key=True, index=True)

    grape: Mapped[Grape] = relationship(back_populates="aliases")


class Wine(Base):
    __tablename__ = "wines"
    __table_args__ = (
        Index("ix_wines_color_sugar", "color", "sugar"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    sku: Mapped[str] = mapped_column(String(500), nullable=False, unique=True, index=True)
    name: Mapped[str] = mapped_column(String(500), nullable=False, index=True)
    producer_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("producers.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    region_id: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("regions.id", ondelete="SET NULL"),
        index=True,
    )
    year: Mapped[int | None] = mapped_column(Integer)
    color: Mapped[str | None] = mapped_column(String(50), index=True)
    sugar: Mapped[str | None] = mapped_column(String(50), index=True)
    alcohol: Mapped[float | None] = mapped_column(Float)
    price: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    stock: Mapped[int | None] = mapped_column(Integer)
    description: Mapped[str | None] = mapped_column(Text)
    source_url: Mapped[str | None] = mapped_column(String(1000))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    rating: Mapped[Decimal | None] = mapped_column(Numeric(3, 2))

    producer: Mapped[Producer] = relationship(back_populates="wines")
    region: Mapped[Region | None] = relationship(back_populates="wines")
    grape_links: Mapped[list[WineGrape]] = relationship(
        back_populates="wine",
        cascade="all, delete-orphan",
    )
    images: Mapped[list[WineImage]] = relationship(
        back_populates="wine",
        cascade="all, delete-orphan",
        order_by="desc(WineImage.is_main), WineImage.id",
    )


class WineGrape(Base):
    __tablename__ = "wine_grapes"

    wine_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("wines.id", ondelete="CASCADE"),
        nullable=False,
        primary_key=True,
        index=True,
    )
    grape_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("grapes.id", ondelete="RESTRICT"),
        nullable=False,
        primary_key=True,
        index=True,
    )
    percentage: Mapped[float | None] = mapped_column(Float)

    wine: Mapped[Wine] = relationship(back_populates="grape_links")
    grape: Mapped[Grape] = relationship(back_populates="wine_links")


class WineImage(Base):
    __tablename__ = "wine_images"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    wine_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("wines.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source_url: Mapped[str | None] = mapped_column(String(1000))
    is_main: Mapped[bool] = mapped_column(Boolean, nullable=False, index=True)
    minio_path: Mapped[str] = mapped_column(String(1000), nullable=False, index=True)

    wine: Mapped[Wine] = relationship(back_populates="images")
