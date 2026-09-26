"""Упрощённая авторизация для хакатона: email + пароль (bcrypt), JWT access-токен.

Анонимный пользователь получает cookie с случайным id — по ней хранится временная
история поиска; при входе/регистрации она переносится в аккаунт.
"""

from __future__ import annotations

import secrets
from datetime import UTC, datetime, timedelta
from typing import Annotated
from uuid import UUID

import bcrypt
import jwt
from fastapi import Depends, HTTPException, Request, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.settings.settings import AuthSettings

_bearer = HTTPBearer(auto_error=False, description="JWT из /api/v1/auth/login или /auth/register")


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def create_access_token(user_id: UUID, settings: AuthSettings) -> str:
    now = datetime.now(UTC)
    payload = {
        "sub": str(user_id),
        "iat": now,
        "exp": now + timedelta(minutes=settings.access_token_ttl_minutes),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str, settings: AuthSettings) -> UUID | None:
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        return UUID(payload["sub"])
    except (jwt.PyJWTError, KeyError, ValueError):
        return None


def _auth_settings(request: Request) -> AuthSettings:
    return request.app.state.connection_manager.settings.auth


async def optional_user_id(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> UUID | None:
    """id пользователя из Bearer-токена; без токена — None. Невалидный токен — 401."""
    if credentials is None:
        return None
    user_id = decode_access_token(credentials.credentials, _auth_settings(request))
    if user_id is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token")
    return user_id


async def current_user_id(user_id: Annotated[UUID | None, Depends(optional_user_id)]) -> UUID:
    if user_id is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user_id


OptionalUserId = Annotated[UUID | None, Depends(optional_user_id)]
CurrentUserId = Annotated[UUID, Depends(current_user_id)]


def read_anon_id(request: Request) -> str | None:
    value = request.cookies.get(_auth_settings(request).anon_cookie_name)
    return value if value and len(value) <= 64 else None


def ensure_anon_id(request: Request, response: Response) -> str:
    """anon_id из cookie или новый (cookie ставится в ответ)."""
    anon_id = read_anon_id(request)
    if anon_id is None:
        settings = _auth_settings(request)
        anon_id = secrets.token_urlsafe(24)
        response.set_cookie(
            settings.anon_cookie_name,
            anon_id,
            max_age=settings.anon_history_ttl_hours * 3600,
            httponly=True,
            samesite="lax",
        )
    return anon_id


def clear_anon_cookie(request: Request, response: Response) -> None:
    response.delete_cookie(_auth_settings(request).anon_cookie_name)
