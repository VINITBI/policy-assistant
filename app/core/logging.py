"""Structured logging.

* logs/app.log    - every event as JSON (one object per line), rotated
* logs/error.log  - WARNING and above
* logs/audit.log  - security/audit events only (logger name "audit")
* console         - human readable

Every record automatically carries trace_id / username / session_id taken from
context variables that the request middleware and auth layer populate.
"""
from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

trace_id_var: ContextVar[str] = ContextVar("trace_id", default="-")
username_var: ContextVar[str] = ContextVar("username", default="-")
session_id_var: ContextVar[str] = ContextVar("session_id", default="-")

_RESERVED = set(vars(logging.makeLogRecord({})).keys()) | {"message", "asctime"}
_configured = False


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = trace_id_var.get()
        record.username = username_var.get()
        record.session_id = session_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "trace_id": getattr(record, "trace_id", "-"),
            "user": getattr(record, "username", "-"),
            "login_session": getattr(record, "session_id", "-"),
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and key not in payload and not key.startswith("_") and key not in (
                "trace_id", "username", "session_id"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__("%(asctime)s | %(levelname)-7s | %(trace_id)s | %(username)s | %(name)s | %(message)s",
                         datefmt="%H:%M:%S")


def setup_logging(log_dir: Path, level: str = "INFO") -> None:
    global _configured
    if _configured:
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    ctx = ContextFilter()

    def _file(name: str, lvl: int) -> RotatingFileHandler:
        h = RotatingFileHandler(log_dir / name, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
        h.setLevel(lvl)
        h.setFormatter(JsonFormatter())
        h.addFilter(ctx)
        return h

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(ConsoleFormatter())
    console.addFilter(ctx)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(console)
    root.addHandler(_file("app.log", logging.DEBUG))
    root.addHandler(_file("error.log", logging.WARNING))

    audit = logging.getLogger("audit")
    audit.setLevel(logging.INFO)
    audit.addHandler(_file("audit.log", logging.INFO))  # also propagates to app.log

    for noisy in ("httpx", "httpcore", "urllib3", "sentence_transformers", "transformers", "multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
