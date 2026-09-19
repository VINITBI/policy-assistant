from __future__ import annotations

from app.core.logging import get_logger
from app.core.observability import Trace
from app.db.models import RequestTrace
from app.db.session import db_scope

log = get_logger(__name__)


def save_trace(trace: Trace, *, username: str = "-", status: str = "ok") -> None:
    """Persist a pipeline trace so it can be inspected later by trace_id. Never raises."""
    try:
        with db_scope() as db:
            db.merge(RequestTrace(trace_id=trace.trace_id, name=trace.name, username=username,
                                  total_ms=trace.total_ms, status=status, spans=trace.to_dict()["spans"]))
    except Exception:  # noqa: BLE001
        log.exception("failed to persist trace %s", trace.trace_id)
