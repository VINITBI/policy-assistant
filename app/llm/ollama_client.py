"""Ollama client (open-source LLMs served locally) with retries and model fallback.

primary model --(retry x N)--> fallback model --(retry x N)--> LLMUnavailable
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.core.exceptions import LLMUnavailable
from app.core.logging import get_logger
from app.core.observability import LLM_CALLS

log = get_logger(__name__)
settings = get_settings()


@dataclass
class LLMResult:
    text: str
    model: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0


class _Retryable(Exception):
    pass


class OllamaClient:
    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.http = httpx.Client(timeout=httpx.Timeout(settings.llm_timeout_seconds, connect=5.0))

    def health(self) -> dict[str, Any]:
        try:
            r = self.http.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            return {"ok": True, "models": models,
                    "primary_available": any(m.startswith(settings.llm_model) for m in models)}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    @retry(reraise=True, stop=stop_after_attempt(settings.llm_max_retries + 1),
           wait=wait_exponential(multiplier=0.5, max=4), retry=retry_if_exception_type(_Retryable))
    def _call(self, model: str, messages: list[dict], temperature: float, fmt: Any, max_tokens: int) -> LLMResult:
        body: dict[str, Any] = {"model": model, "messages": messages, "stream": False,
                                "options": {"temperature": temperature, "num_ctx": settings.llm_num_ctx,
                                            "num_predict": max_tokens, "top_p": 0.9, "repeat_penalty": 1.1}}
        if fmt is not None:
            body["format"] = fmt
        t0 = time.perf_counter()
        try:
            r = self.http.post(f"{self.base_url}/api/chat", json=body)
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
            raise _Retryable(f"{type(exc).__name__}: {exc}") from exc
        if r.status_code >= 500:
            raise _Retryable(f"HTTP {r.status_code}: {r.text[:200]}")
        if r.status_code == 404:
            raise LLMUnavailable(detail=f"model '{model}' not found - run: ollama pull {model}")
        r.raise_for_status()
        data = r.json()
        return LLMResult(text=data.get("message", {}).get("content", ""), model=model,
                         latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                         prompt_tokens=data.get("prompt_eval_count", 0), completion_tokens=data.get("eval_count", 0))

    def chat(self, messages: list[dict], *, purpose: str, temperature: float | None = None, json_format: Any = None,
             max_tokens: int = 1024, model: str | None = None, allow_fallback: bool = True) -> LLMResult:
        models = [model or settings.llm_model]
        if allow_fallback and settings.llm_fallback_model and settings.llm_fallback_model not in models:
            models.append(settings.llm_fallback_model)
        temp = settings.llm_temperature if temperature is None else temperature
        last_error = ""
        for m in models:
            try:
                res = self._call(m, messages, temp, json_format, max_tokens)
                LLM_CALLS.labels(model=m, purpose=purpose, status="ok").inc()
                log.info("llm call ok", extra={"model": m, "purpose": purpose, "latency_ms": res.latency_ms,
                                               "prompt_tokens": res.prompt_tokens,
                                               "completion_tokens": res.completion_tokens})
                return res
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                LLM_CALLS.labels(model=m, purpose=purpose, status="error").inc()
                log.warning("llm call failed, trying fallback" if m != models[-1] else "llm call failed",
                            extra={"model": m, "purpose": purpose, "error": last_error})
        raise LLMUnavailable(detail=last_error)


@lru_cache(maxsize=1)
def get_llm() -> OllamaClient:
    return OllamaClient()
