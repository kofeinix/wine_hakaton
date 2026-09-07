from __future__ import annotations

import logging
import sys


def setup_logging(level: int = logging.DEBUG) -> None:
    """Настраивает корневой логгер и формат вывода логов.

    Args:
        level: Уровень логирования (по умолчанию ``DEBUG``).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
