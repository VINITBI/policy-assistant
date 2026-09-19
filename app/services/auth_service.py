"""Login / logout / session validation with lockout and full audit trail."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.exceptions import AccountLockedError, AuthenticationError, SessionExpiredError, ValidationError
from app.core.logging import get_logger
from app.core.observability import LOGIN_EVENTS
from app.core.security import create_access_token, decode_token, hash_password, verify_password
from app.db.models import LoginSession, User
from app.services import audit
from app.services.audit import Actions

log = get_logger(__name__)
settings = get_settings()


def _aware(dt: datetime | None) -> datetime | None:
    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def bootstrap_users(db: Session) -> None:
    """Create default admin + employee accounts on first start (change passwords via .env)."""
    for username, password, role, name in (
        (settings.bootstrap_admin_username, settings.bootstrap_admin_password, "admin", "Administrator"),
        (settings.bootstrap_user_username, settings.bootstrap_user_password, "user", "Employee"),
    ):
        if not db.scalar(select(User).where(User.username == username)):
            db.add(User(username=username, password_hash=hash_password(password), role=role, full_name=name))
            log.info("bootstrap account created", extra={"account": username, "role": role})
    db.commit()


def create_user(db: Session, *, username: str, password: str, role: str, full_name: str = "") -> User:
    if role not in ("admin", "user"):
        raise ValidationError("Role must be 'admin' or 'user'.")
    if len(password) < 8:
        raise ValidationError("Password must be at least 8 characters.")
    if db.scalar(select(User).where(User.username == username)):
        raise ValidationError(f"Username '{username}' is already taken.")
    user = User(username=username, password_hash=hash_password(password), role=role, full_name=full_name)
    db.add(user)
    db.commit()
    return user


def login(db: Session, *, username: str, password: str, ip: str, user_agent: str) -> tuple[str, LoginSession, User]:
    username = (username or "").strip()
    user = db.scalar(select(User).where(User.username == username))
    now = datetime.now(timezone.utc)

    if user is None or not user.is_active:
        LOGIN_EVENTS.labels(result="failed").inc()
        audit.record(Actions.LOGIN_FAILED, db=db, username=username or "-", outcome="failure", ip=ip,
                     details={"reason": "unknown_or_inactive_user", "user_agent": user_agent})
        raise AuthenticationError()

    locked_until = _aware(user.locked_until)
    if locked_until and locked_until > now:
        LOGIN_EVENTS.labels(result="locked").inc()
        audit.record(Actions.LOGIN_LOCKED, db=db, user_id=user.id, username=user.username, role=user.role,
                     outcome="blocked", ip=ip, details={"locked_until": locked_until.isoformat()})
        raise AccountLockedError()

    if not verify_password(password, user.password_hash):
        user.failed_attempts += 1
        details = {"reason": "bad_password", "failed_attempts": user.failed_attempts, "user_agent": user_agent}
        if user.failed_attempts >= settings.max_failed_logins:
            user.locked_until = now + timedelta(minutes=settings.lockout_minutes)
            user.failed_attempts = 0
            details["locked_until"] = user.locked_until.isoformat()
        db.commit()
        LOGIN_EVENTS.labels(result="failed").inc()
        audit.record(Actions.LOGIN_FAILED, db=db, user_id=user.id, username=user.username, role=user.role,
                     outcome="failure", ip=ip, details=details)
        raise AuthenticationError()

    user.failed_attempts = 0
    user.locked_until = None
    user.last_login_at = now
    ls = LoginSession(user_id=user.id, username=user.username, role=user.role, ip_address=ip,
                      user_agent=(user_agent or "")[:256], expires_at=now)
    db.add(ls)
    db.flush()
    token, expires = create_access_token(user_id=user.id, username=user.username, role=user.role,
                                         login_session_id=ls.id)
    ls.expires_at = expires
    db.commit()
    LOGIN_EVENTS.labels(result="success").inc()
    audit.record(Actions.LOGIN_SUCCESS, db=db, user_id=user.id, username=user.username, role=user.role, ip=ip,
                 resource=f"login_session:{ls.id}", details={"user_agent": user_agent})
    return token, ls, user


def validate_token(db: Session, token: str) -> tuple[User, LoginSession]:
    """Decode JWT and verify the backing login session is still active (supports server-side logout)."""
    claims = decode_token(token)
    ls = db.get(LoginSession, claims.get("jti", ""))
    if ls is None or ls.status != "active":
        raise SessionExpiredError(detail="login session not active")
    now = datetime.now(timezone.utc)
    if _aware(ls.expires_at) < now:
        ls.status = "expired"
        db.commit()
        audit.record(Actions.SESSION_EXPIRED, db=db, user_id=ls.user_id, username=ls.username, role=ls.role,
                     resource=f"login_session:{ls.id}")
        raise SessionExpiredError()
    user = db.get(User, claims["sub"])
    if user is None or not user.is_active:
        raise SessionExpiredError(detail="user disabled")
    ls.last_seen_at = now
    ls.request_count += 1
    db.commit()
    return user, ls


def logout(db: Session, ls: LoginSession, ip: str) -> None:
    ls.status = "logged_out"
    ls.logout_at = datetime.now(timezone.utc)
    db.commit()
    audit.record(Actions.LOGOUT, db=db, user_id=ls.user_id, username=ls.username, role=ls.role, ip=ip,
                 resource=f"login_session:{ls.id}",
                 details={"duration_minutes": round((ls.logout_at - _aware(ls.login_at)).total_seconds() / 60, 1)})
