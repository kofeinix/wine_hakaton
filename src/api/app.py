from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from src.api.routes import router
from src.api.services import WineService
from src.container.manager import ConnectionManager


def create_app(connection_manager: ConnectionManager) -> FastAPI:
    app = FastAPI(
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
            {
                "name": "system",
                "description": "Service health checks.",
            },
        ],
    )
    app.state.connection_manager = connection_manager
    app.state.wine_service = WineService(connection_manager)

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
    return app
