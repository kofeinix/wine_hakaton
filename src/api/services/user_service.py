"""Аккаунт, история поиска, избранное и напоминания."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import selectinload

from src.api.auth import hash_password, verify_password
from src.api.repositories.wine_repository import WineRepository
from src.api.schemas import (
    FavoriteItem,
    NotificationDetail,
    NotificationItem,
    ReviewItem,
    ReviewResponse,
    SearchHistoryItem,
    SearchResultRef,
    ViewedWine,
    WineResponse,
    PublicReview,
    WineReviewsResponse,
)
from src.api.services.wine_service import wine_to_response
from src.connections.database.models import (
    Favorite,
    Notification,
    NotificationWine,
    SearchHistory,
    User,
    Wine,
    WineReview,
    WineReviewReaction,
)
from src.connections.database.postgres import DatabaseClient
from src.settings.settings import AuthSettings, NotificationSettings

logger = logging.getLogger(__name__)

HISTORY_RESULTS_LIMIT = 10  # сколько кандидатов поиска храним в истории


class EmailTakenError(Exception):
    pass


@dataclass(frozen=True)
class ViewedRef:
    """Просмотр вина: top-1 результата поиска."""

    wine_id: UUID
    viewed_at: datetime
    search_id: UUID | None


@dataclass(frozen=True)
class Owner:
    """Владелец истории: пользователь или анонимная cookie."""

    user_id: UUID | None = None
    anon_id: str | None = None

    @property
    def is_anonymous(self) -> bool:
        return self.user_id is None


def parse_uuid(value: str | None) -> UUID | None:
    try:
        return UUID(str(value)) if value else None
    except ValueError:
        return None


def normalize_email(email: str) -> str:
    return email.strip().lower()


def _now() -> datetime:
    return datetime.now(UTC)


class UserService:
    def __init__(
        self,
        database: DatabaseClient,
        auth_settings: AuthSettings,
        notification_settings: NotificationSettings,
    ) -> None:
        self.database = database
        self.wines = WineRepository(database)
        self.auth = auth_settings
        self.notifications = notification_settings

    # --- аккаунт --------------------------------------------------------------------

    async def register(self, email: str, password: str, anon_id: str | None) -> User:
        user = User(email=normalize_email(email), password_hash=hash_password(password))
        async with self.database.session() as session:
            session.add(user)
            try:
                await session.flush()
            except IntegrityError as exc:
                raise EmailTakenError(email) from exc
            await self._claim_anonymous_history(session, user.id, anon_id)
            await session.commit()
        return user

    async def login(self, email: str, password: str, anon_id: str | None) -> User | None:
        async with self.database.session() as session:
            user = await session.scalar(select(User).where(User.email == normalize_email(email)))
            if user is None or not verify_password(password, user.password_hash):
                return None
            await self._claim_anonymous_history(session, user.id, anon_id)
            await session.commit()
        return user

    async def get_user(self, user_id: UUID) -> User | None:
        async with self.database.session() as session:
            return await session.get(User, user_id)

    async def update_profile(
        self,
        user_id: UUID,
        *,
        avatar_url: str | None = None,
        review_notification_period_minutes: int | None = None,
    ) -> User | None:
        async with self.database.session() as session:
            user = await session.get(User, user_id)
            if user is None:
                return None
            if avatar_url is not None:
                user.avatar_url = avatar_url
            if review_notification_period_minutes is not None:
                user.review_notification_period_minutes = review_notification_period_minutes
            await session.commit()
            return user

    async def _claim_anonymous_history(self, session, user_id: UUID, anon_id: str | None) -> None:
        """Временная история по cookie переходит в аккаунт при входе/регистрации."""
        if not anon_id:
            return
        result = await session.execute(
            update(SearchHistory)
            .where(SearchHistory.anon_id == anon_id, SearchHistory.created_at >= self._anon_cutoff())
            .values(user_id=user_id, anon_id=None)
        )
        if result.rowcount:
            logger.info("Moved %s anonymous searches to user %s", result.rowcount, user_id)

    def _anon_cutoff(self) -> datetime:
        return _now() - timedelta(hours=self.auth.anon_history_ttl_hours)

    # --- история --------------------------------------------------------------------

    async def record_search(
        self,
        owner: Owner,
        results: list[tuple[str, float]],
        confidence: float | None,
    ) -> str | None:
        """Сохранить поиск в историю. Ошибка записи не должна ломать поиск -> None."""
        if owner.user_id is None and owner.anon_id is None:
            return None
        refs = [{"wine_id": wine_id, "score": round(float(score), 6)} for wine_id, score in results]
        top_wine_id = parse_uuid(refs[0]["wine_id"]) if refs else None
        try:
            async with self.database.session() as session:
                if top_wine_id is not None and await session.get(Wine, top_wine_id) is None:
                    top_wine_id = None  # вина нет в БД — не ломаем FK
                entry = SearchHistory(
                    user_id=owner.user_id,
                    anon_id=None if owner.user_id else owner.anon_id,
                    top_wine_id=top_wine_id,
                    confidence=confidence,
                    results=refs[:HISTORY_RESULTS_LIMIT],
                )
                session.add(entry)
                await session.commit()
                return str(entry.id)
        except Exception:
            logger.exception("Failed to record search history")
            return None

    async def history(self, owner: Owner, limit: int, offset: int) -> list[SearchHistoryItem]:
        query = select(SearchHistory).order_by(SearchHistory.created_at.desc()).limit(limit).offset(offset)
        if owner.user_id is not None:
            query = query.where(SearchHistory.user_id == owner.user_id)
        elif owner.anon_id is not None:
            query = query.where(
                SearchHistory.anon_id == owner.anon_id, SearchHistory.created_at >= self._anon_cutoff()
            )
        else:
            return []
        async with self.database.session() as session:
            entries = list((await session.scalars(query)).all())
        cards = await self._wine_cards([entry.top_wine_id for entry in entries])
        return [
            SearchHistoryItem(
                id=str(entry.id),
                created_at=entry.created_at,
                confidence=entry.confidence,
                top_wine=cards.get(entry.top_wine_id),
                results=[SearchResultRef(**ref) for ref in entry.results or []],
            )
            for entry in entries
        ]

    async def delete_history(self, owner: Owner, search_id: UUID) -> bool:
        condition = (
            SearchHistory.user_id == owner.user_id
            if owner.user_id is not None
            else SearchHistory.anon_id == owner.anon_id
        )
        async with self.database.session() as session:
            result = await session.execute(
                delete(SearchHistory).where(SearchHistory.id == search_id, condition)
            )
            await session.commit()
        return bool(result.rowcount)

    # --- избранное ------------------------------------------------------------------

    async def add_favorite(self, user_id: UUID, wine_id: UUID, search_id: UUID | None) -> bool:
        """False — вина нет в каталоге. Повторное добавление не ошибка."""
        async with self.database.session() as session:
            if await session.get(Wine, wine_id) is None:
                return False
            if search_id is not None:
                owned = await session.scalar(
                    select(SearchHistory.id).where(
                        SearchHistory.id == search_id, SearchHistory.user_id == user_id
                    )
                )
                search_id = owned  # чужой или несуществующий поиск не привязываем
            await session.execute(
                insert(Favorite)
                .values(user_id=user_id, wine_id=wine_id, search_id=search_id)
                .on_conflict_do_nothing(index_elements=[Favorite.user_id, Favorite.wine_id])
            )
            await session.commit()
        return True

    async def remove_favorite(self, user_id: UUID, wine_id: UUID) -> bool:
        async with self.database.session() as session:
            result = await session.execute(
                delete(Favorite).where(Favorite.user_id == user_id, Favorite.wine_id == wine_id)
            )
            await session.commit()
        return bool(result.rowcount)

    async def favorites(self, user_id: UUID) -> list[FavoriteItem]:
        async with self.database.session() as session:
            rows = list(
                (
                    await session.scalars(
                        select(Favorite).where(Favorite.user_id == user_id).order_by(Favorite.created_at.desc())
                    )
                ).all()
            )
        cards = await self._wine_cards([row.wine_id for row in rows])
        return [
            FavoriteItem(
                wine=cards[row.wine_id],
                search_id=str(row.search_id) if row.search_id else None,
                created_at=row.created_at,
            )
            for row in rows
            if row.wine_id in cards
        ]

    # --- просмотренные вина и отзывы ----------------------------------------------

    async def viewed_wines(self, user_id: UUID, since: datetime, until: datetime) -> list[ViewedWine]:
        """Вина, которые пользователь смотрел за промежуток (последний просмотр каждого)."""
        async with self.database.session() as session:
            searches = list(
                (
                    await session.scalars(
                        select(SearchHistory)
                        .where(
                            SearchHistory.user_id == user_id,
                            SearchHistory.top_wine_id.is_not(None),
                            SearchHistory.created_at >= since,
                            SearchHistory.created_at <= until,
                        )
                        .order_by(SearchHistory.created_at.desc())
                    )
                ).all()
            )
        return await self._viewed_items(user_id, _latest_views(searches))

    async def upsert_review(
        self,
        user_id: UUID,
        wine_id: UUID,
        rating: int | None,
        comment: str | None,
        notification_id: UUID | None,
    ) -> ReviewResponse | None:
        """Создать или заменить отзыв. None — вина нет в каталоге."""
        comment = comment.strip() or None if comment is not None else None
        async with self.database.session() as session:
            if await session.get(Wine, wine_id) is None:
                return None
            if notification_id is not None:
                notification_id = await session.scalar(
                    select(Notification.id).where(
                        Notification.id == notification_id, Notification.user_id == user_id
                    )
                )
            values = {"rating": rating, "comment": comment}
            if notification_id is not None:
                values["notification_id"] = notification_id
            review = await session.scalar(
                insert(WineReview)
                .values(user_id=user_id, wine_id=wine_id, **values)
                .on_conflict_do_update(
                    index_elements=[WineReview.user_id, WineReview.wine_id],
                    set_={**values, "updated_at": func.now()},
                )
                .returning(WineReview)
            )
            await session.commit()
        return _review_response(review)

    async def delete_review(self, user_id: UUID, wine_id: UUID) -> bool:
        async with self.database.session() as session:
            result = await session.execute(
                delete(WineReview).where(WineReview.user_id == user_id, WineReview.wine_id == wine_id)
            )
            await session.commit()
        return bool(result.rowcount)

    async def wine_reviews(
        self, wine_id: UUID, viewer_id: UUID | None = None, limit: int = 50
    ) -> WineReviewsResponse | None:
        """Все отзывы о вине: сводка по оценкам и лента (свой — первым). None — вина нет в каталоге."""
        async with self.database.session() as session:
            if await session.get(Wine, wine_id) is None:
                return None
            rows = (
                await session.execute(
                    select(WineReview, User.email, User.avatar_url)
                    .join(User, User.id == WineReview.user_id)
                    .where(WineReview.wine_id == wine_id)
                    .order_by(WineReview.updated_at.desc())
                )
            ).all()
            author_ids = [review.user_id for review, _, _ in rows]
            review_counts = dict(
                (
                    await session.execute(
                        select(WineReview.user_id, func.count())
                        .where(WineReview.user_id.in_(author_ids), WineReview.comment.is_not(None))
                        .group_by(WineReview.user_id)
                    )
                ).all()
            ) if author_ids else {}
            reaction_rows = (
                await session.execute(
                    select(
                        WineReviewReaction.review_user_id,
                        WineReviewReaction.value,
                        func.count(),
                    )
                    .where(WineReviewReaction.wine_id == wine_id)
                    .group_by(WineReviewReaction.review_user_id, WineReviewReaction.value)
                )
            ).all()
            reaction_counts: dict[UUID, dict[int, int]] = {}
            for author_id, value, count in reaction_rows:
                reaction_counts.setdefault(author_id, {})[value] = int(count)
            my_reactions = {}
            if viewer_id is not None:
                my_reactions = dict(
                    (
                        await session.execute(
                            select(WineReviewReaction.review_user_id, WineReviewReaction.value).where(
                                WineReviewReaction.wine_id == wine_id,
                                WineReviewReaction.user_id == viewer_id,
                            )
                        )
                    ).all()
                )
        ratings = [review.rating for review, _, _ in rows if review.rating]
        items = [
            PublicReview(
                user_id=str(review.user_id),
                author="Вы" if review.user_id == viewer_id else _mask_email(email),
                avatar_url=avatar_url,
                author_review_count=int(review_counts.get(review.user_id, 0)),
                author_frame=_review_frame(int(review_counts.get(review.user_id, 0))),
                is_mine=review.user_id == viewer_id,
                rating=review.rating,
                comment=review.comment,
                likes=reaction_counts.get(review.user_id, {}).get(1, 0),
                dislikes=reaction_counts.get(review.user_id, {}).get(-1, 0),
                my_reaction=my_reactions.get(review.user_id),
                created_at=review.created_at,
                updated_at=review.updated_at,
            )
            for review, email, avatar_url in rows
        ]
        items.sort(key=lambda item: not item.is_mine)  # стабильно: свой первым, остальные по дате
        return WineReviewsResponse(
            wine_id=str(wine_id),
            count=len(rows),
            rated_count=len(ratings),
            average_rating=round(sum(ratings) / len(ratings), 2) if ratings else None,
            distribution={value: ratings.count(value) for value in range(1, 6)},
            items=items[:limit],
        )

    async def reviews(self, user_id: UUID) -> list[ReviewItem]:
        async with self.database.session() as session:
            rows = list(
                (
                    await session.scalars(
                        select(WineReview)
                        .where(WineReview.user_id == user_id)
                        .order_by(WineReview.updated_at.desc())
                    )
                ).all()
            )
        cards = await self._wine_cards([row.wine_id for row in rows])
        return [
            ReviewItem(**_review_response(row).model_dump(), wine=cards[row.wine_id])
            for row in rows
            if row.wine_id in cards
        ]

    async def set_review_reaction(
        self,
        user_id: UUID,
        wine_id: UUID,
        review_user_id: UUID,
        value: int,
    ) -> bool:
        """Поставить лайк/дизлайк отзыву. False — отзыва нет."""
        if value not in (-1, 1) or user_id == review_user_id:
            return False
        async with self.database.session() as session:
            exists = await session.scalar(
                select(WineReview.user_id).where(
                    WineReview.user_id == review_user_id,
                    WineReview.wine_id == wine_id,
                    WineReview.comment.is_not(None),
                )
            )
            if exists is None:
                return False
            await session.execute(
                insert(WineReviewReaction)
                .values(user_id=user_id, review_user_id=review_user_id, wine_id=wine_id, value=value)
                .on_conflict_do_update(
                    index_elements=[
                        WineReviewReaction.user_id,
                        WineReviewReaction.review_user_id,
                        WineReviewReaction.wine_id,
                    ],
                    set_={"value": value, "updated_at": func.now()},
                )
            )
            await session.commit()
        return True

    async def delete_review_reaction(self, user_id: UUID, wine_id: UUID, review_user_id: UUID) -> bool:
        async with self.database.session() as session:
            result = await session.execute(
                delete(WineReviewReaction).where(
                    WineReviewReaction.user_id == user_id,
                    WineReviewReaction.review_user_id == review_user_id,
                    WineReviewReaction.wine_id == wine_id,
                )
            )
            await session.commit()
        return bool(result.rowcount)

    # --- уведомления ---------------------------------------------------------------

    async def list_notifications(
        self, user_id: UUID, unread_only: bool, limit: int
    ) -> tuple[list[NotificationItem], int]:
        query = select(Notification).options(selectinload(Notification.wines)).where(Notification.user_id == user_id)
        if unread_only:
            query = query.where(Notification.read_at.is_(None))
        async with self.database.session() as session:
            rows = list((await session.scalars(query.order_by(Notification.created_at.desc()).limit(limit))).all())
            unread = await session.scalar(
                select(func.count())
                .select_from(Notification)
                .where(Notification.user_id == user_id, Notification.read_at.is_(None))
            )
        return [_notification_item(row) for row in rows], int(unread or 0)

    async def open_notification(self, user_id: UUID, notification_id: UUID) -> NotificationDetail | None:
        """Детали напоминания (список вин с избранным и отзывами); отмечает прочитанным."""
        async with self.database.session() as session:
            notification = await session.scalar(
                select(Notification)
                .options(selectinload(Notification.wines))
                .where(Notification.id == notification_id, Notification.user_id == user_id)
            )
            if notification is None:
                return None
            if notification.read_at is None:
                notification.read_at = _now()
                await session.commit()
        refs = [ViewedRef(item.wine_id, item.viewed_at, item.search_id) for item in notification.wines]
        return NotificationDetail(
            **_notification_item(notification).model_dump(),
            wines=await self._viewed_items(user_id, refs),
        )

    async def mark_notifications_read(self, user_id: UUID, notification_id: UUID | None = None) -> int:
        condition = [Notification.user_id == user_id, Notification.read_at.is_(None)]
        if notification_id is not None:
            condition.append(Notification.id == notification_id)
        async with self.database.session() as session:
            result = await session.execute(update(Notification).where(*condition).values(read_at=_now()))
            await session.commit()
        return int(result.rowcount or 0)

    async def process_due_reminders(self, batch_size: int = 200) -> int:
        """Сводные напоминания "вы недавно смотрели вина, что-то взяли?".

        Сессия пользователя считается законченной, если с его последнего поиска прошло
        NOTIFICATIONS__DELAY_MINUTES. Тогда все его ещё не обработанные поиски собираются в
        одно напоминание: вина (top-1 каждого поиска, без повторов) за последние
        NOTIFICATIONS__MAX_AGE_HOURS, которые он ещё не добавил в избранное и не оценил.
        Каждый поиск обрабатывается один раз (notified_at). Заодно удаляется
        просроченная анонимная история.
        """
        now = _now()
        quiet_since = now - timedelta(minutes=self.notifications.delay_minutes)
        window_start = now - timedelta(hours=self.notifications.max_age_hours)
        created = 0
        async with self.database.session() as session:
            user_ids = list(
                (
                    await session.scalars(
                        select(SearchHistory.user_id)
                        .where(SearchHistory.user_id.is_not(None), SearchHistory.notified_at.is_(None))
                        .group_by(SearchHistory.user_id)
                        .having(func.max(SearchHistory.created_at) <= quiet_since)
                        .limit(batch_size)
                    )
                ).all()
            )
            for user_id in user_ids:
                searches = list(
                    (
                        await session.scalars(
                            select(SearchHistory)
                            .where(SearchHistory.user_id == user_id, SearchHistory.notified_at.is_(None))
                            .order_by(SearchHistory.created_at.desc())
                            .with_for_update(skip_locked=True)  # безопасно при нескольких репликах
                        )
                    ).all()
                )
                for search in searches:
                    search.notified_at = now
                views = _latest_views([s for s in searches if s.created_at >= window_start and s.top_wine_id])
                if not views:
                    continue
                wine_ids = [view.wine_id for view in views]
                handled = set(
                    (
                        await session.scalars(
                            select(Favorite.wine_id).where(Favorite.user_id == user_id, Favorite.wine_id.in_(wine_ids))
                        )
                    ).all()
                ) | set(
                    (
                        await session.scalars(
                            select(WineReview.wine_id).where(
                                WineReview.user_id == user_id, WineReview.wine_id.in_(wine_ids)
                            )
                        )
                    ).all()
                )
                views = [view for view in views if view.wine_id not in handled]
                if not views:
                    continue
                session.add(
                    Notification(
                        user_id=user_id,
                        kind="viewed_wines",
                        message=self.notifications.message_template.format(count=len(views)),
                        period_start=min(view.viewed_at for view in views),
                        period_end=max(view.viewed_at for view in views),
                        wines=[
                            NotificationWine(wine_id=view.wine_id, search_id=view.search_id, viewed_at=view.viewed_at)
                            for view in views
                        ],
                    )
                )
                created += 1
            await session.execute(
                delete(SearchHistory).where(
                    SearchHistory.user_id.is_(None), SearchHistory.created_at < self._anon_cutoff()
                )
            )
            await session.commit()
        created += await self.process_due_review_reaction_notifications(batch_size=batch_size)
        if created:
            logger.info("Created %s notifications", created)
        return created

    async def process_due_review_reaction_notifications(self, batch_size: int = 200) -> int:
        """Сводка автору: сколько людей оценили его комментарии за выбранный период."""
        now = _now()
        created = 0
        async with self.database.session() as session:
            users = list(
                (
                    await session.scalars(
                        select(User)
                        .where(
                            User.review_notification_period_minutes > 0,
                            # только те, у кого период уже истёк, иначе первые batch_size
                            # пользователей навсегда займут выборку
                            or_(
                                User.review_reactions_notified_at.is_(None),
                                User.review_reactions_notified_at
                                <= now - func.make_interval(0, 0, 0, 0, 0, User.review_notification_period_minutes),
                            ),
                        )
                        .order_by(User.review_reactions_notified_at.asc().nulls_first())
                        .limit(batch_size)
                    )
                ).all()
            )
            for user in users:
                configured_start = now - timedelta(minutes=user.review_notification_period_minutes)
                if user.review_reactions_notified_at and user.review_reactions_notified_at > configured_start:
                    continue
                period_start = user.review_reactions_notified_at or configured_start
                if period_start >= now:
                    continue
                reactors_count = await session.scalar(
                    select(func.count(func.distinct(WineReviewReaction.user_id))).where(
                        WineReviewReaction.review_user_id == user.id,
                        WineReviewReaction.updated_at > period_start,
                        WineReviewReaction.updated_at <= now,
                    )
                )
                user.review_reactions_notified_at = now
                reactors_count = int(reactors_count or 0)
                if not reactors_count:
                    continue
                session.add(
                    Notification(
                        user_id=user.id,
                        kind="review_reactions",
                        message=(
                            f"За {_period_label(user.review_notification_period_minutes)} ваши комментарии "
                            f"оценили {reactors_count} {_people_word(reactors_count)}"
                        ),
                        period_start=period_start,
                        period_end=now,
                    )
                )
                created += 1
            await session.commit()
        return created

    async def _viewed_items(self, user_id: UUID, refs: list[ViewedRef]) -> list[ViewedWine]:
        wine_ids = [ref.wine_id for ref in refs]
        if not wine_ids:
            return []
        async with self.database.session() as session:
            favorites = set(
                (
                    await session.scalars(
                        select(Favorite.wine_id).where(Favorite.user_id == user_id, Favorite.wine_id.in_(wine_ids))
                    )
                ).all()
            )
            reviews = {
                review.wine_id: review
                for review in (
                    await session.scalars(
                        select(WineReview).where(WineReview.user_id == user_id, WineReview.wine_id.in_(wine_ids))
                    )
                ).all()
            }
        cards = await self._wine_cards(wine_ids)
        return [
            ViewedWine(
                wine=cards[ref.wine_id],
                viewed_at=ref.viewed_at,
                search_id=str(ref.search_id) if ref.search_id else None,
                is_favorite=ref.wine_id in favorites,
                review=_review_response(reviews[ref.wine_id]) if ref.wine_id in reviews else None,
            )
            for ref in refs
            if ref.wine_id in cards
        ]

    # --- общее ------------------------------------------------------------------------

    async def _wine_cards(self, wine_ids: list[UUID | None]) -> dict[UUID, WineResponse]:
        ids = [str(wine_id) for wine_id in dict.fromkeys(wine_ids) if wine_id is not None]
        if not ids:
            return {}
        return {wine.id: wine_to_response(wine) for wine in await self.wines.load_wines_by_ids(ids)}


def _latest_views(searches: list[SearchHistory]) -> list[ViewedRef]:
    """Последний просмотр каждого вина (поиски в любом порядке), новые сверху."""
    latest: dict[UUID, ViewedRef] = {}
    for search in searches:
        if search.top_wine_id is None:
            continue
        current = latest.get(search.top_wine_id)
        if current is None or search.created_at > current.viewed_at:
            latest[search.top_wine_id] = ViewedRef(search.top_wine_id, search.created_at, search.id)
    return sorted(latest.values(), key=lambda ref: ref.viewed_at, reverse=True)


def _mask_email(email: str) -> str:
    """Публичное имя автора: первая буква логина, остальное скрыто."""
    local = email.split("@", 1)[0]
    return f"{local[:1].upper()}•••" if local else "Пользователь"


def _review_frame(count: int) -> str:
    if count > 100:
        return "diamond"
    if count >= 50:
        return "gold"
    if count >= 10:
        return "silver"
    if count >= 1:
        return "bronze"
    return "none"


def _period_label(minutes: int) -> str:
    """Для фразы «За … ваши комментарии оценили»: минуту, час, сутки, 7 дн."""
    if minutes % 1440 == 0:
        days = minutes // 1440
        return "сутки" if days == 1 else f"{days} дн."
    if minutes % 60 == 0:
        hours = minutes // 60
        return "час" if hours == 1 else f"{hours} ч"
    return "минуту" if minutes == 1 else f"{minutes} мин"


def _people_word(count: int) -> str:
    mod10 = count % 10
    mod100 = count % 100
    if mod10 == 1 and mod100 != 11:
        return "человек"
    return "человека" if mod10 in (2, 3, 4) and not 12 <= mod100 <= 14 else "человек"


def _review_response(review: WineReview) -> ReviewResponse:
    return ReviewResponse(
        wine_id=str(review.wine_id),
        rating=review.rating,
        comment=review.comment,
        created_at=review.created_at,
        updated_at=review.updated_at,
    )


def _notification_item(notification: Notification) -> NotificationItem:
    return NotificationItem(
        id=str(notification.id),
        kind=notification.kind,
        message=notification.message,
        period_start=notification.period_start,
        period_end=notification.period_end,
        wines_count=len(notification.wines),
        created_at=notification.created_at,
        read_at=notification.read_at,
    )
