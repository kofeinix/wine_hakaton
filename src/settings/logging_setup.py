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
from dataclasses import dataclass, field
from typing import Any, Callable

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
    request_log_fields: dict[str, Any] = field(default_factory=dict)


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


def set_request_log_fields(**fields: Any) -> None:
    """Добавить итоговые поля запроса, которые middleware запишет в ``http_request``.

    Этим пользуются долгие сценарии вроде поиска: отдельные stage-логи остаются подробными,
    а access-строка получает короткую сводку результата.
    """
    context = _request_context.get()
    if context is not None:
        context.request_log_fields.update({key: value for key, value in fields.items() if value is not None})


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


def _http_request_message(event_dict: dict) -> str:
    duration_ms = event_dict.get("duration_ms")
    duration = f"{duration_ms} ms" if duration_ms is not None else "unknown time"
    status = event_dict.get("status", "?")
    method = event_dict.get("method", "?")
    path = event_dict.get("path", "?")
    trace = event_dict.get("request_id", "?")
    search = ""
    if event_dict.get("search_top1_slug"):
        search = (
            f"; top1={event_dict['search_top1_slug']} score={event_dict.get('search_top1_score')}; "
            f"ocr={event_dict.get('search_ocr_status')}"
        )
    return f"{method} {path} -> {status} за {duration}{search}; trace={trace}"


def _image_search_message(event_dict: dict) -> str:
    top = event_dict.get("top1_slug") or "ничего"
    score = event_dict.get("top1_score")
    suffix = f" score={score}" if score is not None else ""
    return (
        f"Поиск по фото завершен: top1={top}{suffix}, "
        f"кандидатов={event_dict.get('candidates', 0)}, OCR={event_dict.get('ocr_status')}"
    )


def _visual_search_message(event_dict: dict) -> str:
    gap = event_dict.get("gap")
    gap_text = f", отрыв={gap}" if gap is not None else ""
    return f"Визуальный поиск дал {event_dict.get('candidates', 0)} кандидатов{gap_text}"


def _ocr_text_message(event_dict: dict) -> str:
    applied = "использован" if event_dict.get("applied") else "не использован"
    chars = len(event_dict.get("normalized_text") or "")
    return f"OCR {applied}: источник={event_dict.get('source_view')}, нормализовано {chars} символов"


def _ocr_rerank_message(event_dict: dict) -> str:
    return f"OCR-реранк пересчитал {event_dict.get('candidates', 0)} кандидатов"


def _llm_call_message(event_dict: dict) -> str:
    ok = "успешен" if event_dict.get("ok") else "упал"
    return f"LLM-вызов {ok}: model={event_dict.get('model')}, {event_dict.get('duration_ms')} ms"


def _simple_message(text: str) -> Callable[[dict], str]:
    return lambda _event_dict: text


_EVENT_MESSAGES: dict[str, Callable[[dict], str]] = {
    "http_request": _http_request_message,
    "image_search_started": lambda event_dict: (
        f"Начат поиск по фото: {event_dict.get('image_bytes')} байт, "
        f"limit={event_dict.get('limit')}, только главные фото={event_dict.get('main_photos_only')}"
    ),
    "image_crops": lambda event_dict: (
        f"Кропы построены: image={event_dict.get('image_size')}, "
        f"bottle={event_dict.get('bottle_crop')}, label={event_dict.get('label_crop')}"
    ),
    "image_views_selected": lambda event_dict: (
        f"Выбраны ракурсы поиска: {', '.join(event_dict.get('views') or [])}; "
        f"labels={event_dict.get('labels_detected')}, shelf_mode={event_dict.get('multi_wine_mode')}"
    ),
    "visual_search": _visual_search_message,
    "ocr_skipped": lambda event_dict: (
        f"OCR пропущен: {event_dict.get('reason')}; "
        f"visual_gap={event_dict.get('visual_gap')}, top_score={event_dict.get('top_score')}"
    ),
    "ocr_timeout": lambda event_dict: f"OCR не успел за бюджет {event_dict.get('budget_s')} с",
    "ocr_failed": lambda event_dict: f"OCR упал на source_view={event_dict.get('source_view')}",
    "ocr_text": _ocr_text_message,
    "ocr_rerank": _ocr_rerank_message,
    "image_search": _image_search_message,
    "history_record": lambda event_dict: (
        f"Поиск сохранен в историю: search_id={event_dict.get('search_id')}, source={event_dict.get('source')}"
    ),
    "llm_call": _llm_call_message,
    "http_error": lambda event_dict: (
        f"HTTP-ошибка {event_dict.get('status')} на {event_dict.get('path')}: {event_dict.get('detail')}"
    ),
    "validation_error": lambda event_dict: (
        f"Ошибка валидации на {event_dict.get('path')}: {len(event_dict.get('problems') or [])} проблем"
    ),
    "login": lambda event_dict: "Вход выполнен" if event_dict.get("ok") else "Вход отклонен",
    "register": lambda event_dict: "Регистрация выполнена" if event_dict.get("ok") else "Регистрация отклонена",
    "profile_update": lambda event_dict: (
        f"Профиль обновлен: {event_dict.get('fields')}" if event_dict.get("ok") else "Профиль не обновлен"
    ),
    "favorite_add": lambda event_dict: f"Вино добавлено в избранное: {event_dict.get('wine_id')}",
    "favorite_remove": lambda event_dict: f"Вино удалено из избранного: {event_dict.get('wine_id')}",
    "review_save": _simple_message("Отзыв сохранен"),
    "review_delete": lambda event_dict: f"Отзыв удален: wine_id={event_dict.get('wine_id')}",
    "achievements_earned": lambda event_dict: f"Получены достижения: {event_dict.get('codes')}",
}


def _add_human_message(_logger, _method_name: str, event_dict: dict) -> dict:
    """Добавляет человеческую фразу, не меняя машинное имя события в ``event``."""
    if event_dict.get("message"):
        return event_dict
    event = event_dict.get("event")
    if not isinstance(event, str):
        return event_dict
    builder = _EVENT_MESSAGES.get(event)
    if builder is not None:
        try:
            event_dict["message"] = builder(event_dict)
        except Exception:
            event_dict["message"] = event
    elif event and " " in event:
        # Обычный logging уже пишет готовую фразу в поле event.
        event_dict["message"] = event
    return event_dict


_LEADING_KEYS = ("ts", "level", "logger", "event", "message", "request_id", "actor")


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
        _add_human_message,
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
