"""Аккаунт, история поиска, избранное и напоминания."""

import base64
import io
from datetime import UTC, datetime, timedelta
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, Response, UploadFile, status
from PIL import Image, ImageOps

from src.api.auth import (
    CurrentUserId,
    OptionalUserId,
    clear_anon_cookie,
    create_access_token,
    read_anon_id,
)
from src.api.schemas import (
    AchievementCheckRequest,
    AchievementCheckResponse,
    AchievementsResponse,
    LeaderboardResponse,
    Credentials,
    FavoriteRequest,
    FavoritesResponse,
    NotificationDetail,
    NotificationsResponse,
    ProfileUpdateRequest,
    ReviewPhoto,
    ReviewReactionRequest,
    ReviewRequest,
    ReviewResponse,
    ReviewsResponse,
    SearchHistoryResponse,
    TokenResponse,
    UserResponse,
    ViewedWinesResponse,
    WineReviewsResponse,
)
from src.api.services.achievement_service import AchievementService
from src.api.services.user_service import (
    MAX_REVIEW_PHOTOS,
    EmailTakenError,
    NicknameTakenError,
    Owner,
    UserService,
    parse_uuid,
    valid_nickname,
)
from src.settings.logging_setup import set_actor

router = APIRouter()
logger = structlog.get_logger(__name__)


def get_user_service(request: Request) -> UserService:
    return request.app.state.user_service


UserServiceDep = Annotated[UserService, Depends(get_user_service)]


def get_achievement_service(request: Request) -> AchievementService:
    return request.app.state.achievement_service


AchievementServiceDep = Annotated[AchievementService, Depends(get_achievement_service)]


def _user_response(user) -> UserResponse:
    return UserResponse(
        id=str(user.id),
        email=user.email,
        nickname=user.nickname,
        avatar_url=user.avatar_url,
        review_notification_period_minutes=user.review_notification_period_minutes,
        created_at=user.created_at,
    )


AVATAR_MAX_UPLOAD_BYTES = 10 * 1024 * 1024
AVATAR_SIZE = 256


REVIEW_PHOTO_MAX_SIDE = 1600
PHOTO_CACHE_SECONDS = 7 * 24 * 3600


async def _read_upload(image: UploadFile) -> bytes:
    data = await image.read()
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Файл пустой")
    if len(data) > AVATAR_MAX_UPLOAD_BYTES:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Файл больше 10 МБ")
    return data


def _to_jpeg(data: bytes, *, square: int | None = None, max_side: int | None = None) -> bytes:
    """Перекодировать загрузку в JPEG: учесть поворот из EXIF, убрать метаданные, уменьшить."""
    try:
        with Image.open(io.BytesIO(data)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
    except Exception as exc:  # ultralytics подменяет Image.open, и ошибки бывают не только PIL-овые
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Не удалось прочитать изображение") from exc
    if square:
        image = ImageOps.fit(image, (square, square), Image.Resampling.LANCZOS)
    elif max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


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
        logger.info("register", ok=False)
        raise HTTPException(status.HTTP_409_CONFLICT, "Email already registered") from None
    set_actor(f"user:{user.id}")
    logger.info("register", ok=True, user_id=str(user.id))
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
        logger.info("login", ok=False)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid email or password")
    set_actor(f"user:{user.id}")
    logger.info("login", ok=True, user_id=str(user.id))
    clear_anon_cookie(request, response)
    return _token_response(request, user)


@router.get("/auth/me", response_model=UserResponse, tags=["auth"], summary="Текущий пользователь")
async def me(user_id: CurrentUserId, users: UserServiceDep) -> UserResponse:
    user = await users.get_user(user_id)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User not found")
    return _user_response(user)


@router.patch("/profile", response_model=UserResponse, tags=["auth"], summary="Настройки профиля")
async def update_profile(body: ProfileUpdateRequest, user_id: CurrentUserId, users: UserServiceDep) -> UserResponse:
    nickname = " ".join(body.nickname.split()) if body.nickname is not None else None
    if nickname is not None and not valid_nickname(nickname):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Ник: 3–24 символа — буквы, цифры, пробел, «_», «.», «-»; начинается с буквы или цифры",
        )
    try:
        user = await users.update_profile(
            user_id,
            review_notification_period_minutes=body.review_notification_period_minutes,
            nickname=nickname,
        )
    except NicknameTakenError as exc:
        logger.info("profile_update", ok=False)
        raise HTTPException(status.HTTP_409_CONFLICT, "Этот ник уже занят") from exc
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User not found")
    fields = list(body.model_dump(exclude_unset=True))
    logger.info("profile_update", ok=True, fields=fields)
    return _user_response(user)


@router.post("/profile/avatar", response_model=UserResponse, tags=["auth"], summary="Загрузить аватар")
async def upload_avatar(
    user_id: CurrentUserId,
    users: UserServiceDep,
    image: UploadFile = File(description="Изображение аватара"),
) -> UserResponse:
    # аватар приходит в каждом отзыве, поэтому это маленькая миниатюра прямо в data URL
    jpeg = _to_jpeg(await _read_upload(image), square=AVATAR_SIZE)
    avatar_url = f"data:image/jpeg;base64,{base64.b64encode(jpeg).decode('ascii')}"
    user = await users.update_profile(user_id, avatar_url=avatar_url)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User not found")
    logger.info("avatar_update", bytes=len(jpeg))
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
    items, total = await users.history(owner, limit=limit, offset=offset)
    return SearchHistoryResponse(items=items, total=total, anonymous=owner.is_anonymous)


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
    logger.info("history_delete", search_id=search_id)
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
    logger.info("favorite_add", wine_id=str(wine_id), search_id=body.search_id)
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
    logger.info("favorite_remove", wine_id=wine_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# --- отзывы ---------------------------------------------------------------------------


@router.get("/reviews", response_model=ReviewsResponse, tags=["reviews"], summary="Мои оценки и комментарии")
async def reviews(user_id: CurrentUserId, users: UserServiceDep) -> ReviewsResponse:
    return ReviewsResponse(items=await users.reviews(user_id))


@router.get(
    "/wines/{wine_id}/reviews",
    response_model=WineReviewsResponse,
    tags=["reviews"],
    summary="Все отзывы о вине",
    description=(
        "Средняя оценка пользователей сервиса, распределение по бокалам и лента отзывов. "
        "Доступно без входа; с токеном свой отзыв помечен is_mine и идёт первым. Авторы замаскированы."
    ),
    responses={404: {"description": "Вино не найдено"}},
)
async def wine_reviews(
    wine_id: str,
    users: UserServiceDep,
    user_id: OptionalUserId,
    limit: int = Query(default=50, ge=1, le=200),
) -> WineReviewsResponse:
    result = await users.wine_reviews(_require_uuid(wine_id, "Wine"), viewer_id=user_id, limit=limit)
    if result is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Wine not found")
    return result


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
    # текст комментария в лог не пишем — только его длину
    logger.info(
        "review_save",
        wine_id=wine_id,
        rating=body.rating,
        comment_chars=len((body.comment or "").strip()),
        from_notification=bool(body.notification_id),
    )
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
    logger.info("review_delete", wine_id=wine_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/reviews/{wine_id}/photos",
    response_model=ReviewPhoto,
    status_code=status.HTTP_201_CREATED,
    tags=["reviews"],
    summary="Прикрепить фото к своему отзыву",
    description=f"До {MAX_REVIEW_PHOTOS} фото; сначала нужно сохранить отзыв (`PUT /reviews/{{wine_id}}`).",
    responses={404: {"description": "Отзыва нет"}, 409: {"description": "Лимит фото"}},
)
async def add_review_photo(
    wine_id: str,
    user_id: CurrentUserId,
    users: UserServiceDep,
    image: UploadFile = File(description="Фото"),
) -> ReviewPhoto:
    jpeg = _to_jpeg(await _read_upload(image), max_side=REVIEW_PHOTO_MAX_SIDE)
    try:
        photo = await users.add_review_photo(user_id, _require_uuid(wine_id, "Wine"), jpeg)
    except ValueError as exc:
        logger.info("review_photo_add", ok=False, wine_id=wine_id, reason=str(exc))
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    if photo is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Review not found")
    logger.info("review_photo_add", ok=True, wine_id=wine_id, photo_id=photo.id, bytes=len(jpeg))
    return photo


@router.delete(
    "/reviews/{wine_id}/photos/{photo_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["reviews"],
    summary="Удалить фото из своего отзыва",
)
async def delete_review_photo(wine_id: str, photo_id: str, user_id: CurrentUserId, users: UserServiceDep) -> Response:
    if not await users.delete_review_photo(
        user_id, _require_uuid(wine_id, "Wine"), _require_uuid(photo_id, "Photo")
    ):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Photo not found")
    logger.info("review_photo_delete", wine_id=wine_id, photo_id=photo_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/review-photos/{photo_id}",
    tags=["reviews"],
    summary="Фото из отзыва",
    response_class=Response,
    responses={200: {"content": {"image/jpeg": {}}}, 404: {"description": "Photo not found"}},
)
async def get_review_photo(photo_id: str, users: UserServiceDep) -> Response:
    photo = await users.get_review_photo(_require_uuid(photo_id, "Photo"))
    if photo is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Photo not found")
    file_data, content_type = photo
    return Response(
        content=file_data.getvalue(),
        media_type=content_type,
        headers={"Cache-Control": f"public, max-age={PHOTO_CACHE_SECONDS}, immutable"},
    )


@router.put(
    "/wines/{wine_id}/reviews/{review_user_id}/reaction",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["reviews"],
    summary="Лайкнуть или дизлайкнуть комментарий",
    responses={404: {"description": "Комментарий не найден"}},
)
async def put_review_reaction(
    wine_id: str,
    review_user_id: str,
    body: ReviewReactionRequest,
    user_id: CurrentUserId,
    users: UserServiceDep,
) -> Response:
    if body.value not in (-1, 1):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Reaction must be 1 or -1")
    if not await users.set_review_reaction(
        user_id,
        _require_uuid(wine_id, "Wine"),
        _require_uuid(review_user_id, "Review"),
        body.value,
    ):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Review not found")
    logger.info("review_reaction_set", wine_id=wine_id, review_author=review_user_id, value=body.value)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete(
    "/wines/{wine_id}/reviews/{review_user_id}/reaction",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["reviews"],
    summary="Убрать реакцию с комментария",
)
async def delete_review_reaction(
    wine_id: str,
    review_user_id: str,
    user_id: CurrentUserId,
    users: UserServiceDep,
) -> Response:
    await users.delete_review_reaction(
        user_id,
        _require_uuid(wine_id, "Wine"),
        _require_uuid(review_user_id, "Review"),
    )
    logger.info("review_reaction_remove", wine_id=wine_id, review_author=review_user_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# --- достижения ------------------------------------------------------------------------


@router.post(
    "/achievements/check",
    response_model=AchievementCheckResponse,
    tags=["achievements"],
    summary="Пересчитать достижения после действия",
    description=(
        "Фронтенд вызывает после скана, отзыва, реакции, избранного и при входе. Прогресс считается по данным "
        "пользователя, поэтому здесь же подхватываются достижения, полученные без его участия (лайки под "
        "комментарием). В ответе — что показать во всплывающих уведомлениях: полученные и заметный прогресс."
    ),
)
async def check_achievements(
    body: AchievementCheckRequest, user_id: CurrentUserId, achievements: AchievementServiceDep
) -> AchievementCheckResponse:
    updates = await achievements.check(user_id, body.timezone)
    earned = [update.code for update in updates if update.type == "earned"]
    if earned:
        logger.info("achievements_earned", codes=earned)
    return AchievementCheckResponse(updates=updates)


@router.get(
    "/achievements",
    response_model=AchievementsResponse,
    tags=["achievements"],
    summary="Мои достижения",
    description="Полученные и открытые достижения; скрытые неоткрытые не возвращаются вовсе.",
)
async def list_achievements(user_id: CurrentUserId, achievements: AchievementServiceDep) -> AchievementsResponse:
    return await achievements.list(user_id)


@router.get(
    "/achievements/leaderboard",
    response_model=LeaderboardResponse,
    tags=["achievements"],
    summary="Топ пользователей по достижениям",
)
async def achievements_leaderboard(
    user_id: CurrentUserId,
    achievements: AchievementServiceDep,
    limit: Annotated[int, Query(ge=1, le=100, description="Размер страницы")] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> LeaderboardResponse:
    return await achievements.leaderboard(user_id, limit=limit, offset=offset)


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
    offset: int = Query(default=0, ge=0),
) -> NotificationsResponse:
    items, unread, total = await users.list_notifications(user_id, unread_only=unread_only, limit=limit, offset=offset)
    return NotificationsResponse(items=items, unread=unread, total=total)


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
    logger.info("notifications_read_all")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete(
    "/notifications",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["notifications"],
    summary="Удалить все уведомления",
)
async def delete_all_notifications(user_id: CurrentUserId, users: UserServiceDep) -> Response:
    await users.delete_notifications(user_id)
    logger.info("notifications_delete_all")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete(
    "/notifications/{notification_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["notifications"],
    summary="Удалить уведомление",
    responses={404: {"description": "Уведомление не найдено"}},
)
async def delete_notification(notification_id: str, user_id: CurrentUserId, users: UserServiceDep) -> Response:
    if not await users.delete_notifications(user_id, _require_uuid(notification_id, "Notification")):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Notification not found")
    logger.info("notification_delete", notification_id=notification_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
