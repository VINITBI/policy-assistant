"""Observability: Prometheus metrics + lightweight per-request tracing.

A `Trace` collects timed spans for every pipeline stage. Spans are logged as they
finish, exported as Prometheus histograms, and the whole trace is persisted to the
`request_traces` table so an admin can inspect any request by its trace_id.
"""
from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator

from prometheus_client import Counter, Histogram

from app.core.logging import get_logger, trace_id_var

log = get_logger("trace")

# ---------------- Prometheus metrics ----------------
HTTP_REQUESTS = Counter("http_requests_total", "HTTP requests", ["method", "path", "status"])
HTTP_LATENCY = Histogram("http_request_duration_seconds", "HTTP latency", ["method", "path"])
STAGE_LATENCY = Histogram("pipeline_stage_duration_seconds", "Pipeline stage latency", ["stage"],
                          buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120))
RAG_QUERIES = Counter("rag_queries_total", "RAG queries by outcome", ["answer_type"])
GUARDRAIL_EVENTS = Counter("guardrail_events_total", "Guardrail blocks/warnings", ["stage", "rule", "action"])
LLM_CALLS = Counter("llm_calls_total", "LLM calls", ["model", "purpose", "status"])
LOGIN_EVENTS = Counter("login_events_total", "Login attempts", ["result"])
INGESTED_CHUNKS = Counter("ingested_chunks_total", "Chunks indexed")
FEEDBACK = Counter("answer_feedback_total", "User feedback on answers", ["rating"])


@dataclass
class Span:
    name: str
    start: float
    duration_ms: float = 0.0
    status: str = "ok"
    attrs: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def set(self, **attrs: Any) -> None:
        self.attrs.update(attrs)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "duration_ms": round(self.duration_ms, 2), "status": self.status,
                "attrs": self.attrs, "error": self.error}


@dataclass
class Trace:
    trace_id: str
    name: str
    spans: list[Span] = field(default_factory=list)
    started: float = field(default_factory=time.perf_counter)

    @property
    def total_ms(self) -> float:
        return round((time.perf_counter() - self.started) * 1000, 2)

    def to_dict(self) -> dict[str, Any]:
        return {"trace_id": self.trace_id, "name": self.name, "total_ms": self.total_ms,
                "spans": [s.to_dict() for s in self.spans]}


current_trace: ContextVar[Trace | None] = ContextVar("current_trace", default=None)


def new_trace_id() -> str:
    return uuid.uuid4().hex


def start_trace(name: str) -> Trace:
    tid = trace_id_var.get()
    if tid == "-":
        tid = new_trace_id()
        trace_id_var.set(tid)
    t = Trace(trace_id=tid, name=name)
    current_trace.set(t)
    return t


@contextmanager
def span(name: str, **attrs: Any) -> Iterator[Span]:
    """Time a pipeline stage. Usage: `with span("rerank", candidates=20) as s: ...; s.set(kept=5)`."""
    s = Span(name=name, start=time.perf_counter(), attrs=dict(attrs))
    try:
        yield s
    except Exception as exc:  # noqa: BLE001 - re-raised
        s.status = "error"
        s.error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        s.duration_ms = (time.perf_counter() - s.start) * 1000
        STAGE_LATENCY.labels(stage=name).observe(s.duration_ms / 1000)
        trace = current_trace.get()
        if trace is not None:
            trace.spans.append(s)
        log.info("span %s %s in %.1f ms", name, s.status, s.duration_ms,
                 extra={"span": name, "duration_ms": round(s.duration_ms, 2), "status": s.status,
                        "attrs": s.attrs, "error": s.error})
