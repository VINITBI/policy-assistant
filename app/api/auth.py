from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.api.deps import CurrentUser, client_ip, get_current
from app.api.schemas import LoginRequest, TokenResponse, UserOut
from app.db.session import get_db
from app.services import auth_service

router = APIRouter(prefix="/auth", tags=["Authentication"])


def _token_response(token, ls, user) -> TokenResponse:
    return TokenResponse(access_token=token, expires_at=ls.expires_at, role=user.role, username=user.username,
                         full_name=user.full_name, login_session_id=ls.id)


@router.post("/login", response_model=TokenResponse, summary="Login (JSON) - used by the UI")
def login(body: LoginRequest, request: Request, db: Session = Depends(get_db)):
    token, ls, user = auth_service.login(db, username=body.username, password=body.password, ip=client_ip(request),
                                         user_agent=request.headers.get("user-agent", ""))
    return _token_response(token, ls, user)


@router.post("/token", response_model=TokenResponse, summary="Login (OAuth2 form) - used by Swagger 'Authorize'")
def token(request: Request, form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    tok, ls, user = auth_service.login(db, username=form.username, password=form.password, ip=client_ip(request),
                                       user_agent=request.headers.get("user-agent", "swagger"))
    return _token_response(tok, ls, user)


@router.post("/logout", summary="Logout and revoke the current login session")
def logout(request: Request, current: CurrentUser = Depends(get_current), db: Session = Depends(get_db)):
    auth_service.logout(db, current.login_session, client_ip(request))
    return {"message": "You have been logged out."}


@router.get("/me", response_model=UserOut)
def me(current: CurrentUser = Depends(get_current)):
    return current.user
