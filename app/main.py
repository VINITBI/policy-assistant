"""FastAPI entry point.   uvicorn app.main:app --reload   ->   http://localhost:8000/docs"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api import admin, auth, chat, policies
from app.core.config import get_settings
from app.core.exceptions import AppError
from app.core.logging import get_logger, session_id_var, setup_logging, trace_id_var, username_var
from app.core.observability import HTTP_LATENCY, HTTP_REQUESTS, new_trace_id

settings = get_settings()
setup_logging(settings.log_dir, settings.log_level)
log = get_logger("app")

from app.db.session import SessionLocal, init_db  # noqa: E402  (after logging is configured)
from app.services.auth_service import bootstrap_users  # noqa: E402


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    with SessionLocal() as db:
        bootstrap_users(db)
    from app.llm.prompts import get_prompts

    get_prompts()  # fail fast on a broken prompts.yaml
    log.info("application started", extra={"env": settings.app_env, "llm": settings.llm_model,
                                           "embedding": settings.embedding_model, "qdrant": settings.qdrant_mode})
    yield
    log.info("application stopped")


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="Production-grade RAG chatbot over company policies. "
                "Click **Authorize** and log in (admin/Admin@123 or employee/Employee@123 by default).",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8501"], allow_methods=["*"], allow_headers=["*"],
                   expose_headers=["X-Trace-ID"])


@app.middleware("http")
async def trace_middleware(request: Request, call_next):
    trace_id_var.set(request.headers.get("x-trace-id") or new_trace_id())
    username_var.set("-")
    session_id_var.set("-")
    start = time.perf_counter()
    path = request.scope.get("route").path if request.scope.get("route") else request.url.path
    try:
        response = await call_next(request)
    except Exception:  # pragma: no cover - handled by exception handlers
        log.exception("unhandled error in request pipeline")
        response = _error(500, "INTERNAL_ERROR", AppError.user_message)
    elapsed = time.perf_counter() - start
    route = request.scope.get("route")
    path = route.path if route else path
    HTTP_REQUESTS.labels(request.method, path, response.status_code).inc()
    HTTP_LATENCY.labels(request.method, path).observe(elapsed)
    response.headers["X-Trace-ID"] = trace_id_var.get()
    if path not in ("/metrics", "/health"):
        log.info("%s %s -> %s (%.0f ms)", request.method, request.url.path, response.status_code, elapsed * 1000,
                 extra={"method": request.method, "path": request.url.path, "status": response.status_code,
                        "duration_ms": round(elapsed * 1000, 1)})
    return response


def _error(status: int, code: str, message: str, extra: dict | None = None) -> JSONResponse:
    body = {"error": {"code": code, "message": message, "trace_id": trace_id_var.get(), **(extra or {})}}
    return JSONResponse(status_code=status, content=body, headers={"X-Trace-ID": trace_id_var.get()})


@app.exception_handler(AppError)
async def app_error_handler(_: Request, exc: AppError):
    level = log.error if exc.status_code >= 500 else log.warning
    level("%s: %s", exc.code, exc.detail or exc.user_message, extra={"code": exc.code, "status": exc.status_code})
    return _error(exc.status_code, exc.code, exc.user_message)


@app.exception_handler(RequestValidationError)
async def validation_handler(_: Request, exc: RequestValidationError):
    fields = [".".join(str(p) for p in e["loc"][1:]) + f": {e['msg']}" for e in exc.errors()]
    log.warning("request validation failed", extra={"errors": fields})
    return _error(422, "INVALID_INPUT", "Some fields are missing or invalid: " + "; ".join(fields))


@app.exception_handler(StarletteHTTPException)
async def http_handler(_: Request, exc: StarletteHTTPException):
    return _error(exc.status_code, "HTTP_ERROR", str(exc.detail))


@app.exception_handler(Exception)
async def unhandled_handler(_: Request, exc: Exception):
    log.exception("unhandled exception: %s", exc)
    return _error(500, "INTERNAL_ERROR", AppError.user_message)


for r in (auth.router, policies.router, chat.router, admin.router):
    app.include_router(r, prefix=settings.api_prefix)


@app.get("/health", tags=["Ops"], summary="Liveness")
def health():
    return {"status": "ok"}


@app.get("/ready", tags=["Ops"], summary="Readiness: DB, vector DB, LLM")
def ready():
    from sqlalchemy import text

    from app.db.vector_store import get_vector_store
    from app.llm.ollama_client import get_llm

    checks = {}
    try:
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
        checks["database"] = {"ok": True}
    except Exception as exc:  # noqa: BLE001
        checks["database"] = {"ok": False, "error": str(exc)}
    try:
        checks["qdrant"] = {"ok": get_vector_store().health()}
    except Exception as exc:  # noqa: BLE001
        checks["qdrant"] = {"ok": False, "error": getattr(exc, "detail", str(exc))}
    checks["ollama"] = get_llm().health()
    ok = all(c.get("ok") for c in checks.values())
    return JSONResponse(status_code=200 if ok else 503, content={"status": "ready" if ok else "degraded",
                                                                 "checks": checks})


@app.get("/metrics", tags=["Ops"], summary="Prometheus metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
