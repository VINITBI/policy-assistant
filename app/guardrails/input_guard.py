"""Input guardrails - run before any retrieval or LLM call.

1. validation (empty / length / gibberish / control chars)
2. per-user rate limiting
3. toxicity & harmful-intent filter
4. prompt-injection / jailbreak detection (regex heuristics + optional open-source
   DeBERTa classifier: protectai/deberta-v3-base-prompt-injection-v2)
5. PII & secrets masking (the query is cleaned, not rejected)
"""
from __future__ import annotations

import re
import threading
import time
import unicodedata
from collections import defaultdict, deque
from dataclasses import dataclass, field
from functools import lru_cache

from app.core.config import get_settings
from app.core.exceptions import GuardrailViolation, RateLimitError
from app.core.logging import get_logger
from app.core.observability import GUARDRAIL_EVENTS
from app.guardrails.patterns import HARMFUL_INTENT, INJECTION_PATTERNS, mask_pii, toxicity_hits

log = get_logger(__name__)
settings = get_settings()


@dataclass
class InputCheck:
    clean_query: str
    warnings: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)


class RateLimiter:
    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self.events: dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()

    def check(self, key: str) -> None:
        now = time.monotonic()
        with self.lock:
            q = self.events[key]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_minute:
                raise RateLimitError(detail=f"{key} exceeded {self.per_minute}/min")
            q.append(now)


class InjectionClassifier:
    """Lazy-loaded HF classifier; if it can't load, heuristics alone are used (logged once)."""

    def __init__(self) -> None:
        self._pipe = None
        self._failed = False
        self._lock = threading.Lock()

    def score(self, text: str) -> float | None:
        if not settings.injection_model_enabled or self._failed:
            return None
        with self._lock:
            if self._pipe is None:
                try:
                    from transformers import pipeline

                    self._pipe = pipeline("text-classification", model=settings.injection_model, truncation=True,
                                          max_length=512, device=-1)
                    log.info("prompt-injection classifier loaded", extra={"model": settings.injection_model})
                except Exception as exc:  # noqa: BLE001
                    self._failed = True
                    log.warning("prompt-injection classifier unavailable; using heuristics only",
                                extra={"error": str(exc)})
                    return None
        out = self._pipe(text)[0]
        return float(out["score"]) if out["label"].upper() == "INJECTION" else 1 - float(out["score"])


class InputGuard:
    def __init__(self) -> None:
        self.rate_limiter = RateLimiter(settings.rate_limit_per_minute)
        self.classifier = InjectionClassifier()

    def _block(self, rule: str, message: str, detail: str = "") -> None:
        GUARDRAIL_EVENTS.labels(stage="input", rule=rule, action="block").inc()
        log.warning("input guardrail blocked", extra={"rule": rule, "detail": detail})
        raise GuardrailViolation(message, rule=rule, detail=detail)

    def check(self, query: str, user_key: str) -> InputCheck:
        self.rate_limiter.check(user_key)

        q = unicodedata.normalize("NFKC", query or "")
        q = "".join(ch for ch in q if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
        q = re.sub(r"\s+", " ", q).strip()

        if len(q) < settings.min_query_chars:
            self._block("empty", "Please type a question about the policy.")
        if len(q) > settings.max_query_chars:
            self._block("too_long", f"Your question is too long. Please keep it under {settings.max_query_chars} "
                                    "characters.")
        alnum = sum(ch.isalnum() for ch in q)
        if alnum / max(len(q), 1) < 0.4 or not re.search(r"[A-Za-z]{2,}", q):
            self._block("gibberish", "That doesn't look like a question. Please rephrase it in plain words.")

        tox = toxicity_hits(q)
        if tox:
            self._block("toxicity", "Please keep the conversation respectful. I'm happy to help with policy questions.",
                        detail=",".join(tox))
        if HARMFUL_INTENT.search(q):
            self._block("harmful", "I can only help with questions about company policies.")

        heuristic = max((w for pat, w in INJECTION_PATTERNS if pat.search(q)), default=0.0)
        model_score = self.classifier.score(q) if heuristic < 1.0 else None
        if heuristic >= 1.0 or (model_score is not None and model_score >= settings.injection_threshold) or (
                heuristic >= 0.6 and (model_score or 0) >= 0.5):
            self._block("prompt_injection", "Your message looks like an attempt to change how the assistant works, "
                                            "so it was not processed. Please ask a question about the policy.",
                        detail=f"heuristic={heuristic} model={model_score}")

        result = InputCheck(clean_query=q)
        masked, kinds = mask_pii(q)
        if kinds:
            result.clean_query = masked
            result.flags.append("pii_masked:" + ",".join(kinds))
            result.warnings.append("Personal or sensitive details were removed from your question for your privacy.")
            GUARDRAIL_EVENTS.labels(stage="input", rule="pii", action="mask").inc()
        return result


@lru_cache(maxsize=1)
def get_input_guard() -> InputGuard:
    return InputGuard()
