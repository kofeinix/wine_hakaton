from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class WinePhoto(BaseModel):
    id: str
    url: str = Field(description="GET-ссылка на файл фото (бэкенд отдаёт его из MinIO)")
    is_main: bool


class WineResponse(BaseModel):
    id: str
    slug: str
    name: str
    producer: str | None = None
    region: str | None = None
    country: str | None = None
    year: int | None = None
    color: str | None = None
    sugar: str | None = None
    alcohol: str | None = None
    serving_temperature: str | None = None
    shade: str | None = None
    price: float | None = None
    currency: str = "RUB"
    rating: float | None = None
    description: str | None = None
    grapes: list[str] = Field(default_factory=list)
    food_pairings: list[str] = Field(default_factory=list)
    source_url: str | None = None
    image_url: str | None = Field(default=None, description="Главное фото (первое из photos)")
    photos: list[WinePhoto] = Field(default_factory=list, description="Настоящие фото: main, yandex, vivino")
    generated_photos: list[WinePhoto] = Field(
        default_factory=list, description="Сгенерированные сцены (полка, стол, в руке) — показывать по кнопке"
    )


class SlugResponse(BaseModel):
    slug: str | None = Field(description="Slug найденного вина; null — не найдено")


class SearchMatch(BaseModel):
    rank: int
    slug: str
    wine_id: str
    visual_score: float = Field(description="Скор визуального поиска (SigLIP2 + Qdrant)")
    ocr_score: float = Field(description="OCR-бонус: совпадение текста этикетки с карточкой вина")
    final_score: float = Field(description="visual_score + ocr_score — по нему отсортированы результаты")
    similarity: float = Field(
        default=0.0,
        description="Сходство 0–1: final_score / максимально возможный скор в этом поиске (search.max_score)",
    )
    wine: WineResponse | None = None


class CropResponse(BaseModel):
    available: bool
    box: tuple[int, int, int, int] | None = None
    confidence: float | None = None
    width: int | None = None
    height: int | None = None
    source_view: str | None = None


class DetectionResponse(BaseModel):
    box: tuple[int, int, int, int] = Field(description="x1, y1, x2, y2 в координатах всего фото")
    confidence: float


class DetectionsResponse(BaseModel):
    bottles: list[DetectionResponse] = Field(default_factory=list)
    labels: list[DetectionResponse] = Field(default_factory=list)


class ImageSize(BaseModel):
    width: int
    height: int


class OcrInfo(BaseModel):
    applied: bool = Field(description="OCR распознал текст и участвовал в ранжировании")
    source_view: str | None = Field(default=None, description="С какого кропа читали: label_crop / bottle_crop / original")
    text: str = ""
    cached: bool = Field(default=False, description="Текст взят из кеша (это фото уже распознавали)")
    skipped: bool = Field(
        default=False,
        description="OCR не участвовал: не понадобился или не успел (причина — в reason)",
    )
    reason: str | None = Field(
        default=None,
        description=(
            "ok / skipped_confident_visual (визуальный результат уверенный) / timeout_budget (LLM не уложилась "
            "в SEARCH__OCR_BUDGET_SECONDS) / llm_unavailable / ocr_failed / empty_ocr_text / no_source_image"
        ),
    )


class SearchInfo(BaseModel):
    active_views: list[str] = Field(description="По каким кропам искали в Qdrant")
    label_photo_mode: bool = Field(description="Фото этикетки крупным планом — искали только по ней")
    multi_wine_mode: bool = Field(
        default=False, description="На фото много этикеток (полка) — весь кадр в поиске не участвовал"
    )
    labels_detected: int = Field(default=0, description="Сколько этикеток нашёл детектор на фото")
    candidates: int = Field(description="Сколько вин-кандидатов переранжировал OCR")
    max_score: float = Field(
        default=0.0,
        description="Максимально возможный final_score для этих ракурсов (с OCR-бонусом, если OCR применён)",
    )


class SearchTimings(BaseModel):
    total_ms: float
    crops_ms: float = Field(description="YOLO: кропы бутылки и этикетки")
    ocr_ms: float = Field(description="Распознавание текста vision-LLM (параллельно с эмбеддингами и Qdrant)")
    embedding_ms: float = Field(description="SigLIP2-эмбеддинги кропов")
    vector_search_ms: float = Field(description="Поиск в Qdrant")
    rerank_ms: float = Field(description="OCR-реранк кандидатов")


class SearchResponse(BaseModel):
    search_id: str | None = Field(default=None, description="id записи истории; передайте в POST /favorites")
    status: str = Field(description="found / not_found")
    results: list[SearchMatch] = Field(default_factory=list)
    ocr: OcrInfo
    crops: dict[str, CropResponse] = Field(default_factory=dict)
    image: ImageSize | None = Field(default=None, description="Размер всего загруженного фото")
    detections: DetectionsResponse = Field(
        default_factory=DetectionsResponse,
        description="Все найденные бутылки и этикетки — чтобы выбрать нужную, если на фото несколько вин",
    )
    search: SearchInfo
    timings_ms: SearchTimings
    diagnostics: dict[str, Any] | None = Field(default=None, description="Полная отладка — только при debug=true")


class Credentials(BaseModel):
    email: str = Field(min_length=3, max_length=320, examples=["user@example.com"])
    password: str = Field(min_length=6, max_length=128, examples=["secret123"])


class UserResponse(BaseModel):
    id: str
    email: str
    nickname: str | None = Field(default=None, description="Публичное имя: шапка, отзывы, рейтинг")
    avatar_url: str | None = None
    review_notification_period_minutes: int = Field(default=24 * 60, ge=1)
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class SearchResultRef(BaseModel):
    wine_id: str
    score: float


class SearchHistoryItem(BaseModel):
    id: str
    created_at: datetime
    confidence: float | None = None
    top_wine: WineResponse | None = None
    results: list[SearchResultRef] = Field(default_factory=list)


class SearchHistoryResponse(BaseModel):
    items: list[SearchHistoryItem]
    anonymous: bool = Field(description="true — временная история по cookie, без аккаунта")


class FavoriteRequest(BaseModel):
    wine_id: str
    search_id: str | None = Field(default=None, description="search_id из ответа поиска")


class FavoriteItem(BaseModel):
    wine: WineResponse
    search_id: str | None = None
    created_at: datetime


class FavoritesResponse(BaseModel):
    items: list[FavoriteItem]


class ReviewRequest(BaseModel):
    rating: int | None = Field(default=None, ge=1, le=5, description="Оценка 1–5")
    comment: str | None = Field(default=None, max_length=2000)
    notification_id: str | None = Field(default=None, description="Напоминание, из которого оставлен отзыв")


class ReviewPhoto(BaseModel):
    id: str
    url: str


class ReviewResponse(BaseModel):
    wine_id: str
    rating: int | None = None
    comment: str | None = None
    created_at: datetime
    updated_at: datetime
    photos: list[ReviewPhoto] = Field(default_factory=list)


class ReviewItem(ReviewResponse):
    wine: WineResponse
    likes: int = 0
    dislikes: int = 0


class ReviewsResponse(BaseModel):
    items: list[ReviewItem]


class PublicReview(BaseModel):
    user_id: str = Field(description="ID автора отзыва для реакций")
    author: str = Field(description="Ник автора (email не раскрываем)")
    avatar_url: str | None = None
    author_review_count: int = Field(default=0, description="Сколько комментариев оставил автор")
    author_frame: str = Field(default="none", description="none / bronze / silver / gold / diamond")
    author_reviews_total: int = Field(default=0, description="Сколько всего отзывов оставил автор")
    is_mine: bool = False
    rating: int | None = None
    comment: str | None = None
    likes: int = 0
    dislikes: int = 0
    my_reaction: int | None = Field(default=None, description="1 — лайк, -1 — дизлайк")
    photos: list[ReviewPhoto] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class WineReviewsResponse(BaseModel):
    wine_id: str
    count: int = Field(description="Всего отзывов")
    rated_count: int = Field(description="Сколько отзывов с оценкой")
    average_rating: float | None = Field(default=None, description="Средняя оценка пользователей сервиса, 1–5")
    distribution: dict[int, int] = Field(default_factory=dict, description="Сколько оценок по каждому числу бокалов")
    items: list[PublicReview] = Field(default_factory=list, description="Свой отзыв первым, затем новые сверху")


class ViewedWine(BaseModel):
    """Вино, которое пользователь смотрел: карточка + его избранное и отзыв."""

    wine: WineResponse
    viewed_at: datetime
    search_id: str | None = None
    is_favorite: bool
    review: ReviewResponse | None = None


class ViewedWinesResponse(BaseModel):
    items: list[ViewedWine]


class NotificationItem(BaseModel):
    id: str
    kind: str
    message: str
    period_start: datetime
    period_end: datetime
    wines_count: int
    created_at: datetime
    read_at: datetime | None = None


class NotificationDetail(NotificationItem):
    wines: list[ViewedWine]


class NotificationsResponse(BaseModel):
    items: list[NotificationItem]
    unread: int


class ProfileUpdateRequest(BaseModel):
    review_notification_period_minutes: int | None = Field(default=None, ge=1, le=60 * 24 * 30)
    nickname: str | None = Field(default=None, description="Публичный ник: 3–24 символа, уникальный")


class ReviewReactionRequest(BaseModel):
    value: int = Field(description="1 — лайк, -1 — дизлайк")


class SommelierOption(BaseModel):
    value: str
    label: str
    count: int = Field(default=0, description="Сколько вин останется, если выбрать вариант")


class SommelierFacets(BaseModel):
    occasions: list[SommelierOption]
    dishes: list[SommelierOption]
    tastes: list[SommelierOption]
    types: list[SommelierOption]
    colors: list[SommelierOption]
    sugars: list[SommelierOption]
    regions: list[SommelierOption]
    grapes: list[SommelierOption]


class SommelierMatch(BaseModel):
    wine: WineResponse
    score: float
    reasons: list[str] = Field(description="Почему вино подошло — из полей карточки, каждую причину можно проверить")


class SommelierSearchResponse(BaseModel):
    total: int
    items: list[SommelierMatch]
    corrected_query: str | None = Field(
        default=None, description="Исправленный запрос, если по исходному ничего не нашлось (опечатка)"
    )
    facets: SommelierFacets


class AchievementUpdate(BaseModel):
    type: str = Field(description="earned — получено, progress — продвинулся, summary — сводка при первой проверке")
    code: str
    title: str
    description: str
    category: str
    progress: int
    target: int


class AchievementCheckRequest(BaseModel):
    timezone: str | None = Field(default=None, max_length=64, description="Часовой пояс браузера, например Europe/Moscow")


class AchievementCheckResponse(BaseModel):
    updates: list[AchievementUpdate]


class AchievementItem(BaseModel):
    code: str
    title: str
    description: str
    category: str
    category_label: str
    progress: int
    target: int
    secret: bool = Field(description="Скрытое: показывается только тем, кто его открыл")
    earned: bool
    earned_at: datetime | None = None
    earned_by_percent: float = Field(description="Доля пользователей с хотя бы одним достижением, получивших это")


class AchievementsResponse(BaseModel):
    earned_count: int
    total_count: int = Field(description="Все достижения, включая скрытые (их самих в items нет)")
    items: list[AchievementItem]


class LeaderboardEntry(BaseModel):
    rank: int
    display_name: str = Field(description="Ник пользователя (email не раскрываем)")
    avatar_url: str | None = None
    earned_count: int
    is_me: bool = False


class LeaderboardResponse(BaseModel):
    items: list[LeaderboardEntry]
    total: int = Field(default=0, description="Сколько пользователей в рейтинге (есть хоть одно достижение)")
    me: LeaderboardEntry | None = None


class WineTermIndex(BaseModel):
    id: str
    term: str
    letter: str
    stems: list[str] = Field(description="Основы слов термина; фраза — несколько основ подряд")
    exact: bool = Field(default=False, description="Только точное слово, без окончаний")


class WineTermsResponse(BaseModel):
    endings: list[str] = Field(description="Допустимые окончания: слово = основа + окончание")
    items: list[WineTermIndex]


class WineTermDetail(BaseModel):
    id: str
    term: str
    letter: str
    definition: str
    source_url: str | None = None
