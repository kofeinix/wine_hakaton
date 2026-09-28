import logging

import uvicorn

from src.api.app import create_app
from src.container.manager import ConnectionManager

logger = logging.getLogger(__name__)


class FastApiServer:
    def __init__(
        self,
        connection_manager: ConnectionManager,
        host: str = "0.0.0.0",
        port: int = 8000,
    ) -> None:
        self._app = create_app(connection_manager)
        self._config = uvicorn.Config(
            self._app,
            host=host,
            port=port,
            log_level="info",
            # без своего log_config uvicorn пишет через корневой логгер в общем формате (setup_logging)
            log_config=None,
            access_log=False,
        )
        self._server = uvicorn.Server(self._config)

    async def start(self) -> None:
        logger.info("Starting FastAPI server on %s:%s", self._config.host, self._config.port)
        await self._server.serve()

    async def stop(self) -> None:
        logger.info("Stopping FastAPI server")
        self._server.should_exit = True
