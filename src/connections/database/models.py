from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    false,
    func,
)
from sqlalchemy.orm import Mapped, foreign, mapped_column, relationship
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


class ColorAlias(Base):
    __tablename__ = "color_aliases"

    color: Mapped[str] = mapped_column(String(50), nullable=False, primary_key=True, index=True)
    alias: Mapped[str] = mapped_column(String(255), nullable=False, primary_key=True, index=True)


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
    color_aliases: Mapped[list[ColorAlias]] = relationship(
        primaryjoin=lambda: Wine.color == foreign(ColorAlias.color),
        viewonly=True,
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
    # сгенерированная сцена (flux_*): на фронте показывается отдельно, по кнопке
    is_generated: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=false()
    )
    minio_path: Mapped[str] = mapped_column(String(1000), nullable=False, index=True)

    wine: Mapped[Wine] = relationship(back_populates="images")


# --- пользователи ---------------------------------------------------------------


class User(Base):
    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    email: Mapped[str] = mapped_column(String(320), nullable=False, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class SearchHistory(Base):
    """Поиск по фото. Владелец — пользователь или анонимная cookie (временная история)."""

    __tablename__ = "search_history"
    __table_args__ = (
        Index("ix_search_history_user_created", "user_id", "created_at"),
        Index("ix_search_history_pending_notify", "notified_at", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    anon_id: Mapped[str | None] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )
    top_wine_id: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("wines.id", ondelete="SET NULL")
    )
    confidence: Mapped[float | None] = mapped_column(Float)
    results: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)
    # когда фоновая задача обработала поиск (напомнила или решила не напоминать)
    notified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Favorite(Base):
    __tablename__ = "favorites"

    user_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    wine_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("wines.id", ondelete="CASCADE"), primary_key=True
    )
    search_id: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("search_history.id", ondelete="SET NULL")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class Notification(Base):
    """Сводное напоминание: "вы недавно смотрели вина, что-то взяли?"."""

    __tablename__ = "notifications"
    __table_args__ = (Index("ix_notifications_user_created", "user_id", "created_at"),)

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    kind: Mapped[str] = mapped_column(String(50), nullable=False, default="viewed_wines")
    message: Mapped[str] = mapped_column(Text, nullable=False)
    # промежуток, за который собраны просмотренные вина
    period_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    period_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    read_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    wines: Mapped[list[NotificationWine]] = relationship(
        cascade="all, delete-orphan", order_by="desc(NotificationWine.viewed_at)"
    )


class NotificationWine(Base):
    """Вина, вошедшие в напоминание."""

    __tablename__ = "notification_wines"

    notification_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("notifications.id", ondelete="CASCADE"), primary_key=True
    )
    wine_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("wines.id", ondelete="CASCADE"), primary_key=True
    )
    search_id: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("search_history.id", ondelete="SET NULL")
    )
    viewed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class WineReview(Base):
    """Оценка и комментарий пользователя к вину (одна на пару пользователь–вино)."""

    __tablename__ = "wine_reviews"
    __table_args__ = (Index("ix_wine_reviews_wine", "wine_id"),)

    user_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    wine_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("wines.id", ondelete="CASCADE"), primary_key=True
    )
    rating: Mapped[int | None] = mapped_column(Integer)  # 1..5
    comment: Mapped[str | None] = mapped_column(Text)
    # из какого напоминания оставлен отзыв (если из него)
    notification_id: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("notifications.id", ondelete="SET NULL")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
