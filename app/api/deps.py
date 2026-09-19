from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, Request
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.exceptions import PermissionDeniedError, SessionExpiredError
from app.core.logging import session_id_var, username_var
from app.db.models import LoginSession, User
from app.db.session import get_db
from app.services import audit, auth_service
from app.services.audit import Actions

settings = get_settings()
oauth2 = OAuth2PasswordBearer(tokenUrl=f"{settings.api_prefix}/auth/token", auto_error=False)


@dataclass
class CurrentUser:
    user: User
    login_session: LoginSession


def client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "")


async def get_current(token: str | None = Depends(oauth2), db: Session = Depends(get_db)) -> CurrentUser:
    # async on purpose: context vars set here (username / login session) propagate into the endpoint thread
    # and therefore into every log line of the request.
    if not token:
        raise SessionExpiredError("Please log in to continue.")
    user, ls = auth_service.validate_token(db, token)
    username_var.set(user.username)
    session_id_var.set(ls.id)
    return CurrentUser(user=user, login_session=ls)


def require_admin(request: Request, current: CurrentUser = Depends(get_current),
                  db: Session = Depends(get_db)) -> CurrentUser:
    if current.user.role != "admin":
        audit.record(Actions.ACCESS_DENIED, db=db, user_id=current.user.id, username=current.user.username,
                     role=current.user.role, outcome="blocked", resource=request.url.path, ip=client_ip(request))
        raise PermissionDeniedError("Only administrators can perform this action.")
    return current
