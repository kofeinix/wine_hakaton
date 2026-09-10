from fastapi import FastAPI

from src.api.routes import router
from src.api.service import WineCatalogService
from src.container.manager import ConnectionManager


def create_app(connection_manager: ConnectionManager) -> FastAPI:
    app = FastAPI(
        title="WineHakaton API",
        version="0.1.0",
        description="Vivino-like API template for wine label search and reviews.",
    )
    app.state.connection_manager = connection_manager
    app.state.wine_service = WineCatalogService(connection_manager)

    @app.get("/health", tags=["system"])
    async def health_check() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(router, prefix="/api/v1")
    return app
