"""Журнал HTTP-запросов: request_id, кто обратился, что отправил и что получил.

Личных данных не пишем: ни IP, ни email, ни тела запросов — только метод, путь, размеры,
статус и время. Пользователь — его UUID, анонимный посетитель — короткий хеш cookie.
"""

from __future__ import annotations

import logging
import re
import secrets
from http.cookies import SimpleCookie
from time import perf_counter

import structlog
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from src.settings.logging_setup import (
    RequestContext,
    anon_actor,
    bind_request_context,
    reset_request_context,
)

logger = structlog.get_logger("api.access")

REQUEST_ID_HEADER = "x-request-id"
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
# частые и неинтересные запросы — в DEBUG, чтобы не забивать журнал (фото каталога грузятся пачками)
_QUIET_PATHS = ("/health", "/api/v1/photos/", "/api/v1/reviews/photos/", "/docs", "/openapi.json", "/redoc")
_SLOW_REQUEST_SECONDS = 5.0


class RequestLoggingMiddleware:
    def __init__(self, app: ASGIApp, anon_cookie_name: str) -> None:
        self.app = app
        self.anon_cookie_name = anon_cookie_name

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {key.decode("latin-1").lower(): value.decode("latin-1") for key, value in scope["headers"]}
        incoming_id = headers.get(REQUEST_ID_HEADER, "")
        request_id = incoming_id if _VALID_REQUEST_ID.match(incoming_id) else secrets.token_hex(4)
        context = RequestContext(request_id=request_id, actor=anon_actor(self._anon_id(headers)))
        token = bind_request_context(context)

        method = scope["method"]
        path = scope["path"]
        query_keys = _query_keys(scope.get("query_string", b""))
        state = {"status": 500, "sent": 0, "received": 0}
        started = perf_counter()

        async def receive_wrapper() -> Message:
            message = await receive()
            if message["type"] == "http.request":
                state["received"] += len(message.get("body", b""))
            return message

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                state["status"] = message["status"]
                message.setdefault("headers", [])
                message["headers"] = [*message["headers"], (REQUEST_ID_HEADER.encode(), request_id.encode())]
            elif message["type"] == "http.response.body":
                state["sent"] += len(message.get("body", b""))
            await send(message)

        try:
            await self.app(scope, receive_wrapper, send_wrapper)
        except Exception:
            logger.exception(
                "http_request", method=method, path=path, status=500, duration_ms=round(_ms(started), 1)
            )
            raise
        else:
            elapsed = perf_counter() - started
            status = state["status"]
            if status >= 500:
                level = logging.ERROR
            elif status >= 400 or elapsed >= _SLOW_REQUEST_SECONDS:
                level = logging.WARNING
            elif path.startswith(_QUIET_PATHS) or method in ("HEAD", "OPTIONS"):
                level = logging.DEBUG
            else:
                level = logging.INFO
            logger.log(
                level,
                "http_request",
                method=method,
                path=path,
                query_keys=query_keys,
                status=status,
                duration_ms=round(elapsed * 1000, 1),
                bytes_in=state["received"],
                bytes_out=state["sent"],
                **context.request_log_fields,
            )
        finally:
            reset_request_context(token)

    def _anon_id(self, headers: dict[str, str]) -> str | None:
        raw = headers.get("cookie")
        if not raw:
            return None
        cookie = SimpleCookie()
        try:
            cookie.load(raw)
        except Exception:
            return None
        morsel = cookie.get(self.anon_cookie_name)
        return morsel.value if morsel else None


def _query_keys(query_string: bytes) -> list[str]:
    # только имена параметров: значения (текст поиска, фильтры) пишут сами обработчики, где это уместно
    keys = []
    for part in query_string.decode("latin-1").split("&"):
        key = part.split("=", 1)[0]
        if key and key not in keys:
            keys.append(key)
    return keys


def _ms(started: float) -> float:
    return (perf_counter() - started) * 1000
