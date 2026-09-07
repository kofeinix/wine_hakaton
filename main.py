import asyncio
import sys

from src.container.app_container import AppContainer
from src.settings import setup_logging
from src.settings.settings import all_settings


async def main() -> int:
    """
    Application entry point.

    Returns:
        Exit code: 0 for success, 1 for error
    """
    setup_logging()
    settings = all_settings
    app = AppContainer(settings)
    await app.run_app()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
