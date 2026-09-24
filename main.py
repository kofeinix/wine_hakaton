import asyncio
import sys

from src.container.app_container import AppContainer
from src.settings import setup_logging
from src.settings.settings import all_settings

# import pydevd_pycharm
# pydevd_pycharm.settrace('host.docker.internal', port=56789, stdout_to_server=True, stderr_to_server=True)

async def main() -> int:
    """
    Application entry point.

    Returns:
        Exit code: 0 for success, 1 for error
    """
    settings = all_settings
    setup_logging(settings.log_level)
    app = AppContainer(settings)
    await app.run_app()
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
