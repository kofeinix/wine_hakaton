"""Аккаунт, история поиска, избранное и напоминания."""

from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status

from src.api.auth import (
    CurrentUserId,
    OptionalUserId,
    clear_anon_cookie,
    create_access_token,
    read_anon_id,
)
from src.api.schemas import (
    Credentials,
    FavoriteRequest,
    FavoritesResponse,
    NotificationDetail,
    NotificationsResponse,
    ReviewRequest,
    ReviewResponse,
    ReviewsResponse,
    SearchHistoryResponse,
    TokenResponse,
    UserResponse,
    ViewedWinesResponse,
)
from src.api.services.user_service import EmailTakenError, Owner, UserService, parse_uuid

router = APIRouter()


def get_user_service(request: Request) -> UserService:
    return request.app.state.user_service


UserServiceDep = Annotated[UserService, Depends(get_user_service)]


def _user_response(user) -> UserResponse:
    return UserResponse(id=str(user.id), email=user.email, created_at=user.created_at)


def _token_response(request: Request, user) -> TokenResponse:
    settings = request.app.state.connection_manager.settings.auth
    return TokenResponse(access_token=create_access_token(user.id, settings), user=_user_response(user))


def _require_uuid(value: str, what: str):
    parsed = parse_uuid(value)
    if parsed is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"{what} not found")
    return parsed


# --- auth ------------------------------------------------------------------------------


@router.post(
    "/auth/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["auth"],
    summary="Регистрация",
    description="Создаёт пользователя и возвращает JWT. Временная история по cookie переносится в аккаунт.",
    responses={409: {"description": "Email уже зарегистрирован"}},
)
async def register(body: Credentials, request: Request, response: Response, users: UserServiceDep) -> TokenResponse:
    if "@" not in body.email:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Invalid email")
    try:
        user = await users.register(body.email, body.password, read_anon_id(request))
    except EmailTakenError:
        raise HTTPException(status.HTTP_409_CONFLICT, "Email already registered") from None
    clear_anon_cookie(request, response)
    return _token_response(request, user)


@router.post(
    "/auth/login",
    response_model=TokenResponse,
    tags=["auth"],
    summary="Вход",
    description="Возвращает JWT (заголовок `Authorization: Bearer <token>`). Временная история по cookie переносится в аккаунт.",
    responses={401: {"description": "Неверный email или пароль"}},
)
async def login(body: Credentials, request: Request, response: Response, users: UserServiceDep) -> TokenResponse:
    user = await users.login(body.email, body.password, read_anon_id(request))
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid email or password")
    clear_anon_cookie(request, response)
    return _token_response(request, user)


@router.get("/auth/me", response_model=UserResponse, tags=["auth"], summary="Текущий пользователь")
async def me(user_id: CurrentUserId, users: UserServiceDep) -> UserResponse:
    user = await users.get_user(user_id)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User not found")
    return _user_response(user)


# --- история ---------------------------------------------------------------------------


@router.get(
    "/history",
    response_model=SearchHistoryResponse,
    tags=["history"],
    summary="История поиска",
    description=(
        "С токеном — история аккаунта. Без токена — временная история по cookie "
        "(хранится AUTH__ANON_HISTORY_TTL_HOURS)."
    ),
)
async def history(
    request: Request,
    user_id: OptionalUserId,
    users: UserServiceDep,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> SearchHistoryResponse:
    owner = Owner(user_id=user_id, anon_id=None if user_id else read_anon_id(request))
    items = await users.history(owner, limit=limit, offset=offset)
    return SearchHistoryResponse(items=items, anonymous=owner.is_anonymous)


@router.get(
    "/history/wines",
    response_model=ViewedWinesResponse,
    tags=["history"],
    summary="Просмотренные вина за промежуток",
    description=(
        "Вина, которые пользователь смотрел (top-1 каждого поиска, последний просмотр), с отметкой "
        "избранного и отзывом. По умолчанию — последние 24 часа."
    ),
)
async def viewed_wines(
    user_id: CurrentUserId,
    users: UserServiceDep,
    since: datetime | None = Query(default=None, description="ISO-время начала; по умолчанию сутки назад"),
    until: datetime | None = Query(default=None, description="ISO-время конца; по умолчанию сейчас"),
) -> ViewedWinesResponse:
    until = _aware(until) or datetime.now(UTC)
    since = _aware(since) or until - timedelta(hours=24)
    return ViewedWinesResponse(items=await users.viewed_wines(user_id, since, until))


def _aware(value: datetime | None) -> datetime | None:
    return value.replace(tzinfo=UTC) if value is not None and value.tzinfo is None else value


@router.delete(
    "/history/{search_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["history"],
    summary="Удалить запись истории",
)
async def delete_history(search_id: str, request: Request, user_id: OptionalUserId, users: UserServiceDep) -> Response:
    owner = Owner(user_id=user_id, anon_id=None if user_id else read_anon_id(request))
    if owner.user_id is None and owner.anon_id is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Search not found")
    if not await users.delete_history(owner, _require_uuid(search_id, "Search")):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Search not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# --- избранное -------------------------------------------------------------------------


@router.get("/favorites", response_model=FavoritesResponse, tags=["favorites"], summary="Избранные вина")
async def favorites(user_id: CurrentUserId, users: UserServiceDep) -> FavoritesResponse:
    return FavoritesResponse(items=await users.favorites(user_id))


@router.post(
    "/favorites",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["favorites"],
    summary="Добавить вино в избранное",
    description="Повторное добавление не ошибка. `search_id` из ответа поиска связывает избранное с поиском.",
    responses={404: {"description": "Вино не найдено"}},
)
async def add_favorite(body: FavoriteRequest, user_id: CurrentUserId, users: UserServiceDep) -> Response:
    wine_id = _require_uuid(body.wine_id, "Wine")
    if not await users.add_favorite(user_id, wine_id, parse_uuid(body.search_id)):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Wine not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete(
    "/favorites/{wine_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["favorites"],
    summary="Убрать вино из избранного",
)
async def remove_favorite(wine_id: str, user_id: CurrentUserId, users: UserServiceDep) -> Response:
    if not await users.remove_favorite(user_id, _require_uuid(wine_id, "Favorite")):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Favorite not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# --- отзывы ---------------------------------------------------------------------------


@router.get("/reviews", response_model=ReviewsResponse, tags=["reviews"], summary="Мои оценки и комментарии")
async def reviews(user_id: CurrentUserId, users: UserServiceDep) -> ReviewsResponse:
    return ReviewsResponse(items=await users.reviews(user_id))


@router.put(
    "/reviews/{wine_id}",
    response_model=ReviewResponse,
    tags=["reviews"],
    summary="Оценить вино / оставить комментарий",
    description=(
        "Создаёт или заменяет отзыв пользователя к вину: оценка 1–5 и/или комментарий. "
        "`notification_id` — если отзыв оставлен из напоминания."
    ),
    responses={404: {"description": "Вино не найдено"}, 422: {"description": "Нет ни оценки, ни комментария"}},
)
async def put_review(wine_id: str, body: ReviewRequest, user_id: CurrentUserId, users: UserServiceDep) -> ReviewResponse:
    if body.rating is None and not (body.comment or "").strip():
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Provide rating and/or comment")
    review = await users.upsert_review(
        user_id,
        _require_uuid(wine_id, "Wine"),
        rating=body.rating,
        comment=body.comment,
        notification_id=parse_uuid(body.notification_id),
    )
    if review is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Wine not found")
    return review


@router.delete(
    "/reviews/{wine_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["reviews"],
    summary="Удалить отзыв",
)
async def delete_review(wine_id: str, user_id: CurrentUserId, users: UserServiceDep) -> Response:
    if not await users.delete_review(user_id, _require_uuid(wine_id, "Review")):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Review not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# --- уведомления -----------------------------------------------------------------------


@router.get(
    "/notifications",
    response_model=NotificationsResponse,
    tags=["notifications"],
    summary="Уведомления",
    description=(
        "Сводные напоминания «вы недавно смотрели вина, что-то взяли?»: одно на сессию поисков, "
        "через NOTIFICATIONS__DELAY_MINUTES после последнего поиска. Список вин — в GET /notifications/{id}."
    ),
)
async def notifications(
    user_id: CurrentUserId,
    users: UserServiceDep,
    unread_only: bool = Query(default=False),
    limit: int = Query(default=50, ge=1, le=200),
) -> NotificationsResponse:
    items, unread = await users.list_notifications(user_id, unread_only=unread_only, limit=limit)
    return NotificationsResponse(items=items, unread=unread)


@router.get(
    "/notifications/{notification_id}",
    response_model=NotificationDetail,
    tags=["notifications"],
    summary="Открыть напоминание",
    description=(
        "Вина из напоминания: карточка, когда смотрел, в избранном ли, оценка и комментарий. "
        "Кнопки: POST /favorites, PUT /reviews/{wine_id}. Отмечает напоминание прочитанным."
    ),
    responses={404: {"description": "Напоминание не найдено"}},
)
async def open_notification(notification_id: str, user_id: CurrentUserId, users: UserServiceDep) -> NotificationDetail:
    detail = await users.open_notification(user_id, _require_uuid(notification_id, "Notification"))
    if detail is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Notification not found")
    return detail


@router.post(
    "/notifications/{notification_id}/read",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["notifications"],
    summary="Отметить уведомление прочитанным",
)
async def read_notification(notification_id: str, user_id: CurrentUserId, users: UserServiceDep) -> Response:
    await users.mark_notifications_read(user_id, _require_uuid(notification_id, "Notification"))
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/notifications/read-all",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["notifications"],
    summary="Отметить все уведомления прочитанными",
)
async def read_all_notifications(user_id: CurrentUserId, users: UserServiceDep) -> Response:
    await users.mark_notifications_read(user_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
