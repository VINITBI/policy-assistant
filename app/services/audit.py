"""Audit trail: every security-relevant action is written to the `audit_logs` table AND to logs/audit.log."""
from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.core.logging import get_logger, session_id_var, trace_id_var
from app.db.models import AuditLog
from app.db.session import db_scope

audit_log = get_logger("audit")


class Actions:
    LOGIN_SUCCESS = "LOGIN_SUCCESS"
    LOGIN_FAILED = "LOGIN_FAILED"
    LOGIN_LOCKED = "LOGIN_LOCKED"
    LOGOUT = "LOGOUT"
    SESSION_EXPIRED = "SESSION_EXPIRED"
    ACCESS_DENIED = "ACCESS_DENIED"
    POLICY_UPLOAD = "POLICY_UPLOAD"
    POLICY_INGESTED = "POLICY_INGESTED"
    POLICY_INGEST_FAILED = "POLICY_INGEST_FAILED"
    POLICY_DELETE = "POLICY_DELETE"
    POLICY_REINDEX = "POLICY_REINDEX"
    CHAT_QUERY = "CHAT_QUERY"
    CHAT_SESSION_DELETE = "CHAT_SESSION_DELETE"
    GUARDRAIL_BLOCK = "GUARDRAIL_BLOCK"
    FEEDBACK = "FEEDBACK"
    USER_CREATE = "USER_CREATE"
    EVAL_RUN = "EVAL_RUN"


def record(action: str, *, db: Session | None = None, user_id: str | None = None, username: str = "-",
           role: str = "-", resource: str = "", outcome: str = "success", details: dict[str, Any] | None = None,
           ip: str = "") -> None:
    """Write an audit event. Never raises - auditing must not break the user flow (failures are logged)."""
    entry = AuditLog(user_id=user_id, username=username, role=role, action=action, resource=resource,
                     outcome=outcome, details=details or {}, ip_address=ip, trace_id=trace_id_var.get(),
                     login_session_id=session_id_var.get())
    audit_log.info("%s %s by %s -> %s", action, resource, username, outcome,
                   extra={"action": action, "resource": resource, "outcome": outcome, "details": details or {},
                          "ip": ip, "actor": username})
    try:
        if db is not None:
            db.add(entry)
            db.commit()
        else:
            with db_scope() as s:
                s.add(entry)
    except Exception:  # noqa: BLE001
        audit_log.exception("failed to persist audit event %s", action)
