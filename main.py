import os

# Ultralytics при импорте ставит OMP_NUM_THREADS=1, если переменная не задана, — и torch (YOLO и SigLIP2)
# считает в один поток: в Docker на CPU кропы 1,6 с вместо 0,5 с. Задаём до первого импорта torch;
# своё значение можно передать через окружение.
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 1))

import asyncio  # noqa: E402
import sys  # noqa: E402

from src.container.app_container import AppContainer  # noqa: E402
from src.settings import setup_logging  # noqa: E402
from src.settings.settings import all_settings  # noqa: E402


async def main() -> int:
    """
    Application entry point.

    Returns:
        Exit code: 0 for success, 1 for error
    """
    settings = all_settings
    setup_logging(settings.log_level, settings.log_format)
    app = AppContainer(settings)
    await app.run_app()
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
