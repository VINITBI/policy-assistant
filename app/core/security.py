"""Password hashing (bcrypt) and JWT access tokens. The JWT `jti` is the login-session id,
so every token maps to an auditable row in `login_sessions` and can be revoked on logout."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

from app.core.config import get_settings
from app.core.exceptions import SessionExpiredError

settings = get_settings()


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8")[:72], bcrypt.gensalt(rounds=12)).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8")[:72], password_hash.encode("utf-8"))
    except ValueError:
        return False


def create_access_token(*, user_id: str, username: str, role: str, login_session_id: str) -> tuple[str, datetime]:
    expires = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_minutes)
    payload = {"sub": user_id, "username": username, "role": role, "jti": login_session_id, "exp": expires,
               "iat": datetime.now(timezone.utc)}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm), expires


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except jwt.ExpiredSignatureError as exc:
        raise SessionExpiredError(detail="token expired") from exc
    except jwt.PyJWTError as exc:
        raise SessionExpiredError("Please log in to continue.", detail=f"bad token: {exc}") from exc
