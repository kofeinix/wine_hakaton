import asyncio
import io
import time
from contextlib import asynccontextmanager, suppress

import structlog
from fastapi import FastAPI, Request
from fastapi.exception_handlers import http_exception_handler, request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import RedirectResponse
from PIL import Image
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.routes import eval_router, router
from src.api.services import WineService
from src.api.services.achievement_service import AchievementService
from src.api.services.notification_worker import ReminderWorker
from src.api.services.sommelier_service import SommelierService
from src.api.services.term_service import TermService
from src.api.services.user_service import UserService
from src.api.middleware import RequestLoggingMiddleware
from src.api.user_routes import router as user_router
from src.container.manager import ConnectionManager
from src.settings.settings import DEFAULT_JWT_SECRET

logger = structlog.get_logger(__name__)


def create_app(connection_manager: ConnectionManager) -> FastAPI:
    settings = connection_manager.settings
    user_service = UserService(
        connection_manager.database,
        settings.auth,
        settings.notifications,
        storage=lambda: connection_manager.minio,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # users / search_history / favorites / notifications: create_all создаёт только
        # отсутствующие таблицы, каталог не трогает
        await connection_manager.database.create_tables()
        if assigned := await user_service.backfill_nicknames():
            logger.info("Assigned nicknames to %s existing users", assigned)
        if settings.auth.jwt_secret == DEFAULT_JWT_SECRET:
            logger.warning("AUTH__JWT_SECRET is not set: using the insecure default secret")
        # модели (YOLO, SigLIP2) грузятся при первом поиске — прогреваем в фоне, чтобы первый
        # настоящий запрос не ждал загрузки (проверочный скрипт даёт на ответ 10 с)
        warmup_task = asyncio.create_task(_warm_up(app.state.wine_service))
        worker_task = None
        if settings.notifications.enabled:
            worker = ReminderWorker(user_service, settings.notifications.check_interval_seconds)
            worker_task = asyncio.create_task(worker.run())
        try:
            yield
        finally:
            warmup_task.cancel()
            if worker_task is not None:
                worker_task.cancel()
                with suppress(asyncio.CancelledError):
                    await worker_task

    app = FastAPI(
        lifespan=lifespan,
        title="WineHakaton API",
        version="0.1.0",
        summary="Wine label recognition and catalog search API.",
        description=(
            "API for matching uploaded wine bottle images against the wine catalog. "
            "The image search pipeline crops bottle and label views with YOLO, extracts OCR text "
            "with a local vision model, embeds image views with SigLIP2, searches Qdrant by "
            "vector similarity, and enriches results from Postgres."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        swagger_ui_parameters={
            "displayRequestDuration": True,
            "defaultModelsExpandDepth": 2,
            "persistAuthorization": True,
        },
        openapi_tags=[
            {
                "name": "search",
                "description": "Image-based wine label recognition and matching.",
            },
            {
                "name": "wines",
                "description": "Wine catalog lookup and photo delivery.",
            },
            {"name": "auth", "description": "Регистрация и вход (email + пароль, JWT)."},
            {"name": "history", "description": "История поиска: аккаунт или временная по cookie."},
            {"name": "favorites", "description": "Избранные вина (нужен вход)."},
            {"name": "reviews", "description": "Оценки и комментарии к винам (нужен вход)."},
            {"name": "notifications", "description": "Напоминания «вы смотрели вина, что-то взяли?» (нужен вход)."},
            {
                "name": "system",
                "description": "Service health checks.",
            },
        ],
    )
    app.state.connection_manager = connection_manager
    app.state.wine_service = WineService(connection_manager)
    app.state.sommelier_service = SommelierService(connection_manager.database)
    app.state.term_service = TermService(connection_manager.database)
    app.state.achievement_service = AchievementService(connection_manager.database, app.state.sommelier_service)
    app.state.user_service = user_service

    @app.get(
        "/",
        include_in_schema=False,
    )
    async def swagger_redirect() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    @app.get(
        "/health",
        tags=["system"],
        summary="Health check",
        description="Returns a lightweight liveness response for the API process.",
    )
    async def health_check() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(router, prefix="/api/v1")
    app.include_router(eval_router)
    app.include_router(user_router, prefix="/api/v1")
    app.add_middleware(RequestLoggingMiddleware, anon_cookie_name=settings.auth.anon_cookie_name)

    @app.exception_handler(StarletteHTTPException)
    async def log_http_exception(request: Request, exc: StarletteHTTPException):
        # строка доступа покажет статус, здесь — почему (detail не содержит пользовательских данных)
        if exc.status_code != 404 or request.url.path.startswith("/api/"):
            logger.info("http_error", status=exc.status_code, path=request.url.path, detail=exc.detail)
        return await http_exception_handler(request, exc)

    @app.exception_handler(RequestValidationError)
    async def log_validation_error(request: Request, exc: RequestValidationError):
        # только где и что не так, без присланных значений (там могут быть email и пароль)
        problems = [
            {"field": ".".join(str(part) for part in error.get("loc", ())), "type": error.get("type")}
            for error in exc.errors()
        ]
        logger.info("validation_error", path=request.url.path, problems=problems)
        return await request_validation_exception_handler(request, exc)

    return app


async def _warm_up(wine_service: WineService) -> None:
    """Один поиск по пустой картинке: загружает модели и прогревает Qdrant до первого пользователя."""
    buffer = io.BytesIO()
    Image.new("RGB", (640, 640), (200, 190, 180)).save(buffer, format="JPEG")
    started = time.perf_counter()
    try:
        await wine_service.search(buffer.getvalue())
        logger.info("Search warm-up done in %.1fs", time.perf_counter() - started)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning("Search warm-up failed; the first search will load models itself", exc_info=True)
