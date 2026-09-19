"""Admin: audit trail, login sessions, traces, dashboard stats, users, evaluations."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, Request
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import CurrentUser, client_ip, require_admin
from app.api.schemas import AuditOut, EvalRunOut, LoginSessionOut, TraceOut, UserCreate, UserOut
from app.core.exceptions import NotFoundError
from app.db.models import AuditLog, ChatMessage, EvaluationRun, LoginSession, Policy, RequestTrace, User
from app.db.session import get_db
from app.services import audit, auth_service
from app.services.audit import Actions

router = APIRouter(prefix="/admin", tags=["Admin - Audit & Observability"])


@router.get("/audit-logs", response_model=list[AuditOut])
def audit_logs(action: str | None = None, username: str | None = None, outcome: str | None = None,
               trace_id: str | None = None, limit: int = 200, _: CurrentUser = Depends(require_admin),
               db: Session = Depends(get_db)):
    q = select(AuditLog).order_by(AuditLog.ts.desc()).limit(min(limit, 2000))
    if action:
        q = q.where(AuditLog.action == action)
    if username:
        q = q.where(AuditLog.username == username)
    if outcome:
        q = q.where(AuditLog.outcome == outcome)
    if trace_id:
        q = q.where(AuditLog.trace_id == trace_id)
    return db.scalars(q).all()


@router.get("/login-sessions", response_model=list[LoginSessionOut])
def login_sessions(username: str | None = None, status: str | None = None, limit: int = 200,
                   _: CurrentUser = Depends(require_admin), db: Session = Depends(get_db)):
    q = select(LoginSession).order_by(LoginSession.login_at.desc()).limit(min(limit, 2000))
    if username:
        q = q.where(LoginSession.username == username)
    if status:
        q = q.where(LoginSession.status == status)
    return db.scalars(q).all()


@router.get("/traces", response_model=list[TraceOut])
def traces(name: str | None = None, limit: int = 50, _: CurrentUser = Depends(require_admin),
           db: Session = Depends(get_db)):
    q = select(RequestTrace).order_by(RequestTrace.created_at.desc()).limit(min(limit, 500))
    if name:
        q = q.where(RequestTrace.name == name)
    return db.scalars(q).all()


@router.get("/traces/{trace_id}", response_model=TraceOut)
def trace(trace_id: str, _: CurrentUser = Depends(require_admin), db: Session = Depends(get_db)):
    t = db.get(RequestTrace, trace_id)
    if t is None:
        raise NotFoundError("Trace not found.")
    return t


@router.get("/stats", summary="Dashboard KPIs")
def stats(_: CurrentUser = Depends(require_admin), db: Session = Depends(get_db)):
    since = datetime.now(timezone.utc) - timedelta(hours=24)
    a = ChatMessage
    answers = select(a).where(a.role == "assistant")
    by_type = dict(db.execute(select(a.answer_type, func.count()).where(a.role == "assistant")
                              .group_by(a.answer_type)).all())
    evals = [m.self_eval.get("overall") for m in db.scalars(answers.order_by(a.created_at.desc()).limit(500))
             if (m.self_eval or {}).get("overall") is not None]
    logins = dict(db.execute(select(AuditLog.action, func.count()).where(
        AuditLog.ts >= since, AuditLog.action.in_((Actions.LOGIN_SUCCESS, Actions.LOGIN_FAILED, Actions.LOGIN_LOCKED)))
        .group_by(AuditLog.action)).all())
    return {
        "users": db.scalar(select(func.count()).select_from(User)),
        "active_policies": db.scalar(select(func.count()).select_from(Policy).where(Policy.status == "active")),
        "indexed_chunks": db.scalar(select(func.coalesce(func.sum(Policy.chunk_count), 0)).where(Policy.status == "active")),
        "total_questions": sum(by_type.values()),
        "questions_24h": db.scalar(select(func.count()).select_from(a).where(a.role == "assistant", a.created_at >= since)),
        "answer_types": by_type,
        "avg_latency_ms": round(db.scalar(select(func.avg(a.latency_ms)).where(a.role == "assistant")) or 0, 1),
        "avg_self_eval": round(sum(evals) / len(evals), 3) if evals else None,
        "feedback_up": db.scalar(select(func.count()).select_from(a).where(a.feedback == 1)),
        "feedback_down": db.scalar(select(func.count()).select_from(a).where(a.feedback == -1)),
        "guardrail_blocks": db.scalar(select(func.count()).select_from(AuditLog).where(
            AuditLog.action == Actions.GUARDRAIL_BLOCK)),
        "logins_24h": logins,
        "active_login_sessions": db.scalar(select(func.count()).select_from(LoginSession).where(
            LoginSession.status == "active", LoginSession.expires_at > datetime.now(timezone.utc))),
    }


@router.get("/users", response_model=list[UserOut])
def users(_: CurrentUser = Depends(require_admin), db: Session = Depends(get_db)):
    return db.scalars(select(User).order_by(User.username)).all()


@router.post("/users", response_model=UserOut, status_code=201)
def create_user(body: UserCreate, request: Request, admin: CurrentUser = Depends(require_admin),
                db: Session = Depends(get_db)):
    user = auth_service.create_user(db, username=body.username, password=body.password, role=body.role,
                                    full_name=body.full_name)
    audit.record(Actions.USER_CREATE, db=db, user_id=admin.user.id, username=admin.user.username, role="admin",
                 ip=client_ip(request), resource=f"user:{user.username}", details={"role": user.role})
    return user


@router.post("/evaluations", response_model=EvalRunOut, status_code=202,
             summary="Run the DeepEval evaluation suite on the golden dataset (background)")
def run_evaluation(request: Request, background: BackgroundTasks, max_cases: int | None = None,
                   admin: CurrentUser = Depends(require_admin), db: Session = Depends(get_db)):
    from app.evaluation.runner import run_evaluation as _run  # heavy import only when needed

    run = EvaluationRun(triggered_by=admin.user.username)
    db.add(run)
    db.commit()
    audit.record(Actions.EVAL_RUN, db=db, user_id=admin.user.id, username=admin.user.username, role="admin",
                 ip=client_ip(request), resource=f"evaluation:{run.id}")
    background.add_task(_run, run.id, max_cases)
    return run


@router.get("/evaluations", response_model=list[EvalRunOut])
def evaluations(limit: int = 20, _: CurrentUser = Depends(require_admin), db: Session = Depends(get_db)):
    return db.scalars(select(EvaluationRun).order_by(EvaluationRun.started_at.desc()).limit(limit)).all()
