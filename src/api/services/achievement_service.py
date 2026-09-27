"""Достижения: пересчёт по данным пользователя, выдача, уведомления, топ пользователей.

После любого действия фронтенд вызывает POST /achievements/check: сервис собирает снимок данных
пользователя (Snapshot), считает прогресс каждого достижения из каталога, выдаёт новые и решает,
о каком прогрессе сообщить. Так же подхватываются достижения, полученные без участия пользователя
(лайки под его комментарием) — при следующей проверке.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert

from src.api.schemas import (
    AchievementItem,
    AchievementsResponse,
    AchievementUpdate,
    LeaderboardEntry,
    LeaderboardResponse,
)
from src.api.services.achievement_catalog import (
    CATEGORIES,
    SCAN_MIN_CONFIDENCE,
    Achievement,
    Scan,
    Snapshot,
    build_catalog,
)
from src.api.services.sommelier_service import SommelierService
from src.connections.database.models import (
    Favorite,
    SearchHistory,
    User,
    UserAchievement,
    UserAchievementState,
    WineReview,
    WineReviewReaction,
)
from src.connections.database.postgres import DatabaseClient

logger = logging.getLogger(__name__)

DEFAULT_TIMEZONE = "Europe/Moscow"
PROGRESS_MILESTONES = (0.1, 0.25, 0.5, 0.75, 0.9)
PROGRESS_QUIET = timedelta(minutes=10)  # чаще не напоминаем о прогрессе одного достижения
MAX_PROGRESS_UPDATES = 2  # за одну проверку — не больше двух уведомлений о прогрессе
FIRST_CHECK_TOAST_LIMIT = 3  # при первой проверке больше — одной сводкой, а не лентой
LEADERBOARD_SIZE = 20


# лайки под каждым комментарием (комментарий = автор + вино)
_LIKES_PER_COMMENT = (
    select(
        WineReviewReaction.review_user_id,
        WineReviewReaction.wine_id,
        func.count().label("likes"),
    )
    .where(WineReviewReaction.value == 1)
    .group_by(WineReviewReaction.review_user_id, WineReviewReaction.wine_id)
    .subquery()
)



@dataclass
class _State:
    progress: int = 0
    notified_progress: int = 0
    notified_at: datetime | None = None


class AchievementService:
    def __init__(self, database: DatabaseClient, sommelier: SommelierService) -> None:
        self.database = database
        self.sommelier = sommelier  # каталог вин в памяти: регион, цвет, сахар, сорта
        self._catalog: list[Achievement] | None = None

    async def catalog(self) -> list[Achievement]:
        if self._catalog is None:
            self._catalog = build_catalog(await self.sommelier.entries())
        return self._catalog

    # --- проверка после действия ------------------------------------------------------------

    async def check(self, user_id: UUID, timezone: str | None = None) -> list[AchievementUpdate]:
        """Пересчитать достижения пользователя; вернуть, что показать во всплывающих уведомлениях."""
        catalog = await self.catalog()
        now = datetime.now(UTC)
        async with self.database.session() as session:
            user = await session.get(User, user_id)
            if user is None:
                return []
            if timezone and timezone != user.timezone and _zone(timezone) is not None:
                user.timezone = timezone
            snapshot = await self._snapshot(session, user_id, _zone(user.timezone) or ZoneInfo(DEFAULT_TIMEZONE))
            earned = {
                code: at
                for code, at in (
                    await session.execute(
                        select(UserAchievement.code, UserAchievement.earned_at).where(UserAchievement.user_id == user_id)
                    )
                ).all()
            }
            states = {
                row.code: _State(row.progress, row.notified_progress, row.notified_at)
                for row in (
                    await session.scalars(select(UserAchievementState).where(UserAchievementState.user_id == user_id))
                ).all()
            }
            first_check = not states and not earned

            progress = _progress(catalog, snapshot, set(earned))
            new_codes = [item.code for item in catalog if item.code not in earned and progress[item.code] >= item.target]
            for code in new_codes:
                earned[code] = now
            visible = _visible_codes(catalog, progress, set(earned))

            updates = [
                AchievementUpdate(type="earned", **_describe(item, progress[item.code]))
                for item in catalog
                if item.code in new_codes
            ]
            progress_updates = []
            for item in catalog:
                state = states.get(item.code, _State())
                value = progress[item.code]
                if item.code in earned or item.code not in visible or value <= state.progress:
                    continue
                if _should_notify(item, value, state, now):
                    progress_updates.append((value / item.target, item, value))
            # ближе к цели — интереснее; остальные дождутся следующего раза
            progress_updates.sort(key=lambda entry: -entry[0])
            notified = {item.code for _, item, _ in progress_updates[:MAX_PROGRESS_UPDATES]}
            updates += [
                AchievementUpdate(type="progress", **_describe(item, value))
                for _, item, value in progress_updates[:MAX_PROGRESS_UPDATES]
            ]

            if new_codes:
                await session.execute(
                    insert(UserAchievement)
                    .values([{"user_id": user_id, "code": code, "earned_at": now} for code in new_codes])
                    .on_conflict_do_nothing()
                )
            rows = []
            for item in catalog:
                state = states.get(item.code, _State())
                value = progress[item.code]
                # при первой проверке пишем всё, даже нули: следующая проверка уже не «первая»
                if value == state.progress and item.code not in notified and not first_check:
                    continue
                rows.append(
                    {
                        "user_id": user_id,
                        "code": item.code,
                        "progress": value,
                        "notified_progress": value if item.code in notified else state.notified_progress,
                        "notified_at": now if item.code in notified else state.notified_at,
                    }
                )
            if rows:
                statement = insert(UserAchievementState).values(rows)
                await session.execute(
                    statement.on_conflict_do_update(
                        index_elements=[UserAchievementState.user_id, UserAchievementState.code],
                        set_={
                            "progress": statement.excluded.progress,
                            "notified_progress": statement.excluded.notified_progress,
                            "notified_at": statement.excluded.notified_at,
                            "updated_at": func.now(),
                        },
                    )
                )
            await session.commit()

        if first_check:
            # первая проверка у пользователя со старой историей: не заваливаем ленту уведомлений
            earned_now = [update for update in updates if update.type == "earned"]
            if len(earned_now) > FIRST_CHECK_TOAST_LIMIT:
                return [
                    AchievementUpdate(
                        type="summary",
                        code="summary",
                        title=f"Получено достижений: {len(earned_now)}",
                        description="За то, что вы уже успели сделать. Все — в кабинете, вкладка «Достижения».",
                        category="meta",
                        progress=len(earned_now),
                        target=len(catalog),
                    )
                ]
            return earned_now
        return updates

    # --- кабинет --------------------------------------------------------------------------

    async def list(self, user_id: UUID) -> AchievementsResponse:
        catalog = await self.catalog()
        async with self.database.session() as session:
            earned = {
                code: at
                for code, at in (
                    await session.execute(
                        select(UserAchievement.code, UserAchievement.earned_at).where(UserAchievement.user_id == user_id)
                    )
                ).all()
            }
            progress = {
                code: value
                for code, value in (
                    await session.execute(
                        select(UserAchievementState.code, UserAchievementState.progress).where(
                            UserAchievementState.user_id == user_id
                        )
                    )
                ).all()
            }
            holders = dict(
                (await session.execute(select(UserAchievement.code, func.count()).group_by(UserAchievement.code))).all()
            )
            players = await session.scalar(select(func.count(func.distinct(UserAchievement.user_id)))) or 0
        values = {item.code: min(progress.get(item.code, 0), item.target) for item in catalog}
        for code in earned:
            if code in values:
                values[code] = next(item.target for item in catalog if item.code == code)
        visible = _visible_codes(catalog, values, set(earned))
        items = [
            AchievementItem(
                **_describe(item, values[item.code]),
                category_label=CATEGORIES[item.category],
                secret=item.hidden and item.series is None,  # ступени серий — не «секреты»
                earned=item.code in earned,
                earned_at=earned.get(item.code),
                earned_by_percent=round(holders.get(item.code, 0) / players * 100, 2) if players else 0.0,
            )
            for item in catalog
            if item.code in visible
        ]
        return AchievementsResponse(
            earned_count=sum(1 for item in catalog if item.code in earned),
            total_count=len(catalog),
            items=items,
        )

    async def leaderboard(self, user_id: UUID | None) -> LeaderboardResponse:
        async with self.database.session() as session:
            rows = (
                await session.execute(
                    select(
                        User.id,
                        User.email,
                        User.avatar_url,
                        func.count(UserAchievement.code).label("earned"),
                        func.max(UserAchievement.earned_at).label("last"),
                    )
                    .join(UserAchievement, UserAchievement.user_id == User.id)
                    .group_by(User.id)
                    # больше достижений; при равенстве — кто раньше добрал, затем кто раньше пришёл
                    .order_by(func.count(UserAchievement.code).desc(), func.max(UserAchievement.earned_at), User.created_at)
                )
            ).all()
        entries = [
            LeaderboardEntry(
                rank=rank,
                display_name="Вы" if row.id == user_id else _mask(row.email),
                avatar_url=row.avatar_url,
                earned_count=row.earned,
                is_me=row.id == user_id,
            )
            for rank, row in enumerate(rows, start=1)
        ]
        return LeaderboardResponse(
            items=entries[:LEADERBOARD_SIZE],
            me=next((entry for entry in entries if entry.is_me), None),
        )

    # --- снимок данных пользователя ---------------------------------------------------------

    async def _snapshot(self, session, user_id: UUID, zone: ZoneInfo) -> Snapshot:
        wines = {entry.card.id: entry for entry in await self.sommelier.entries()}
        snapshot = Snapshot()
        for wine_id, created_at, confidence in (
            await session.execute(
                select(SearchHistory.top_wine_id, SearchHistory.created_at, SearchHistory.confidence).where(
                    SearchHistory.user_id == user_id,
                    SearchHistory.top_wine_id.is_not(None),
                    SearchHistory.confidence >= SCAN_MIN_CONFIDENCE,
                    SearchHistory.from_camera.is_(True),  # только снятые камерой, не из галереи
                )
            )
        ).all():
            entry = wines.get(str(wine_id))
            if entry is not None:
                snapshot.scans.append(Scan(entry, created_at.astimezone(zone), confidence))

        for rating, comment, notification_id in (
            await session.execute(
                select(WineReview.rating, WineReview.comment, WineReview.notification_id).where(
                    WineReview.user_id == user_id
                )
            )
        ).all():
            if comment:
                snapshot.comments += 1
            if rating:
                snapshot.ratings.add(rating)
            if notification_id is not None:
                snapshot.review_from_notification = True

        given = dict(
            (
                await session.execute(
                    select(WineReviewReaction.value, func.count())
                    .where(WineReviewReaction.user_id == user_id)
                    .group_by(WineReviewReaction.value)
                )
            ).all()
        )
        snapshot.likes_given, snapshot.dislikes_given = given.get(1, 0), given.get(-1, 0)
        snapshot.max_likes_on_comment = await session.scalar(
            select(func.coalesce(func.max(_LIKES_PER_COMMENT.c.likes), 0)).where(
                _LIKES_PER_COMMENT.c.review_user_id == user_id
            )
        )

        favorites = (
            await session.execute(select(Favorite.search_id).where(Favorite.user_id == user_id))
        ).scalars().all()
        snapshot.favorites = len(favorites)
        snapshot.favorite_after_scan = any(search_id is not None for search_id in favorites)
        return snapshot


# --- правила ----------------------------------------------------------------------------------


def _progress(catalog: list[Achievement], snapshot: Snapshot, already: set[str]) -> dict[str, int]:
    """Прогресс по всем достижениям; мета-достижения — после остальных, по числу полученных."""
    progress = {item.code: min(item.progress(snapshot), item.target) for item in catalog if not item.meta}
    snapshot.earned = already | {item.code for item in catalog if not item.meta and progress[item.code] >= item.target}
    for item in catalog:
        if item.meta:
            progress[item.code] = min(item.progress(snapshot), item.target)
    return progress


def _visible_codes(catalog: list[Achievement], progress: dict[str, int], earned: set[str]) -> set[str]:
    visible = set(earned)
    series_open: dict[str, bool] = {}  # серия -> получено ли предыдущее достижение
    for item in catalog:
        if item.code in earned:
            if item.series:
                series_open[item.series] = True
            continue
        if item.series:
            if item.series in series_open and series_open[item.series] is None:
                continue  # следующее в серии уже показано — дальше не раскрываем
            if not item.hidden or series_open.get(item.series):
                visible.add(item.code)
            series_open[item.series] = None
        elif not item.hidden:
            visible.add(item.code)
    return visible


def _should_notify(item: Achievement, value: int, state: _State, now: datetime) -> bool:
    if state.notified_progress == 0:
        return True  # первый прогресс — достижение только что открылось
    crossed = any(state.notified_progress / item.target < mark <= value / item.target for mark in PROGRESS_MILESTONES)
    quiet = state.notified_at is None or now - state.notified_at >= PROGRESS_QUIET
    return crossed or quiet


def _describe(item: Achievement, value: int) -> dict:
    return {
        "code": item.code,
        "title": item.title,
        "description": item.description,
        "category": item.category,
        "progress": value,
        "target": item.target,
    }


def _zone(name: str | None) -> ZoneInfo | None:
    if not name:
        return None
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return None


def _mask(email: str) -> str:
    local = email.split("@", 1)[0]
    return f"{local[:1].upper()}•••" if local else "Пользователь"
