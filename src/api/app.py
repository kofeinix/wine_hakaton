import asyncio
import logging
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from src.api.routes import router
from src.api.services import WineService
from src.api.services.notification_worker import ReminderWorker
from src.api.services.user_service import UserService
from src.api.user_routes import router as user_router
from src.container.manager import ConnectionManager
from src.settings.settings import DEFAULT_JWT_SECRET

logger = logging.getLogger(__name__)


def create_app(connection_manager: ConnectionManager) -> FastAPI:
    settings = connection_manager.settings
    user_service = UserService(connection_manager.database, settings.auth, settings.notifications)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # users / search_history / favorites / notifications: create_all создаёт только
        # отсутствующие таблицы, каталог не трогает
        await connection_manager.database.create_tables()
        if settings.auth.jwt_secret == DEFAULT_JWT_SECRET:
            logger.warning("AUTH__JWT_SECRET is not set: using the insecure default secret")
        worker_task = None
        if settings.notifications.enabled:
            worker = ReminderWorker(user_service, settings.notifications.check_interval_seconds)
            worker_task = asyncio.create_task(worker.run())
        try:
            yield
        finally:
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
    app.include_router(user_router, prefix="/api/v1")
    return app
