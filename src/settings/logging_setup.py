"""Логи через structlog.

Свой код пишет события с полями: ``logger.info("image_search", top1_slug=..., timings_ms=...)``.
Логи библиотек и модулей на стандартном ``logging`` (uvicorn, старые ``%s``-сообщения) проходят через
те же процессоры, поэтому формат у всех строк один: JSON (``LOG_FORMAT=json``) или цветной текст (``text``).
"""

from __future__ import annotations

import hashlib
import logging
import re
import sys
from contextvars import ContextVar
from dataclasses import dataclass

import structlog

# шумные библиотеки: httpx пишет INFO на каждый запрос к LLM, urllib3 (MinIO) — на каждое соединение
NOISY_LOGGERS = ("httpx", "httpcore", "openai", "urllib3", "PIL", "multipart", "python_multipart", "asyncio")
# библиотеки со своим обработчиком в stdout (мимо общего формата): переводим их на корневой логгер
OWN_HANDLER_LOGGERS = ("ultralytics", "transformers")
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


@dataclass
class RequestContext:
    """Кто и в каком запросе пишет лог. Изменяемый объект: зависимость авторизации дописывает
    пользователя, и это видно middleware, хотя эндпоинт выполняется в другом контексте."""

    request_id: str
    actor: str = "anon"


_request_context: ContextVar[RequestContext | None] = ContextVar("request_context", default=None)


def current_request_context() -> RequestContext | None:
    return _request_context.get()


def bind_request_context(context: RequestContext):
    return _request_context.set(context)


def reset_request_context(token) -> None:
    _request_context.reset(token)


def set_actor(actor: str) -> None:
    """Пометить текущий запрос: ``user:<uuid>`` или ``anon:<хеш cookie>``."""
    context = _request_context.get()
    if context is not None:
        context.actor = actor


def anon_actor(anon_id: str | None) -> str:
    # сам anon_id — ключ к временной истории, в лог только короткий хеш
    if not anon_id:
        return "anon"
    return "anon:" + hashlib.sha256(anon_id.encode("utf-8")).hexdigest()[:10]


def add_request_context(_logger, _method_name: str, event_dict: dict) -> dict:
    """Процессор structlog: ``request_id`` и ``actor`` текущего HTTP-запроса."""
    context = _request_context.get()
    if context is not None:
        event_dict.setdefault("request_id", context.request_id)
        event_dict.setdefault("actor", context.actor)
    return event_dict


def _drop_colors(_logger, _method_name: str, event_dict: dict) -> dict:
    # uvicorn дублирует сообщение с ANSI-цветами в extra, Ultralytics красит сам текст — в JSON это мусор
    event_dict.pop("color_message", None)
    event = event_dict.get("event")
    if isinstance(event, str) and "\x1b" in event:
        event_dict["event"] = _ANSI_ESCAPE.sub("", event)
    return event_dict


_LEADING_KEYS = ("ts", "level", "logger", "event", "request_id", "actor")


def _order_keys(_logger, _method_name: str, event_dict: dict) -> dict:
    # время, уровень, событие и автор — первыми: строку удобно читать и глазами
    ordered = {key: event_dict.pop(key) for key in _LEADING_KEYS if key in event_dict}
    ordered.update(event_dict)
    return ordered


def setup_logging(level: str = "INFO", fmt: str = "json") -> None:
    """Настраивает structlog и стандартный logging на общий вывод в stdout.

    Args:
        level: Уровень логирования (по умолчанию ``INFO``).
        fmt: ``json`` — строка JSON на запись (для разбора), ``text`` — читаемый текст.
    """
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.stdlib.ExtraAdder(),
        structlog.processors.TimeStamper(fmt="iso", utc=True, key="ts"),
        add_request_context,
        _drop_colors,
        structlog.processors.StackInfoRenderer(),
    ]
    # трассировки без локальных переменных: в них бывают тела запросов (email, пароль при входе)
    if fmt.lower() == "text":
        final_processors = [
            structlog.dev.ConsoleRenderer(
                colors=sys.stdout.isatty(), exception_formatter=structlog.dev.plain_traceback
            )
        ]
    else:
        # трассировка — структурой в поле exception, чтобы запись оставалась одной строкой JSON
        final_processors = [
            structlog.processors.ExceptionRenderer(
                structlog.tracebacks.ExceptionDictTransformer(show_locals=False)
            ),
            _order_keys,
            structlog.processors.JSONRenderer(ensure_ascii=False, default=str),
        ]

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared_processors,
            processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, *final_processors],
        )
    )
    logging.basicConfig(level=level.upper(), handlers=[handler], force=True)
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    try:
        # transformers ставит свой обработчик лениво, при первом логе: отключаем через его API
        from transformers.utils import logging as transformers_logging

        transformers_logging.disable_default_handler()
        transformers_logging.enable_propagation()
        # полоса «Loading weights» пишется в stderr построчно и ломает разбор логов
        transformers_logging.disable_progress_bar()
    except ImportError:
        pass
    for name in OWN_HANDLER_LOGGERS:
        library_logger = logging.getLogger(name)
        library_logger.handlers.clear()
        library_logger.propagate = True
    # warnings.warn (например, InsecureKeyLengthWarning из PyJWT) — тоже записями лога, а не текстом в stderr
    logging.captureWarnings(True)
    # доступ логирует наш middleware (с request_id и пользователем), строки uvicorn.access дублировали бы его
    logging.getLogger("uvicorn.access").disabled = True
