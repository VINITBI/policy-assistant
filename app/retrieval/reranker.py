"""Cross-encoder re-ranking with BAAI/bge-reranker-v2-m3 (open-source, SOTA among open rerankers).
Falls back to fusion order if the model cannot be loaded or fails."""
from __future__ import annotations

import math
import threading
from functools import lru_cache

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.observability import span
from app.retrieval.retriever import Candidate

log = get_logger(__name__)
settings = get_settings()


class Reranker:
    def __init__(self) -> None:
        self._model = None
        self._failed = not settings.reranker_enabled
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None and not self._failed:
                try:
                    from sentence_transformers import CrossEncoder

                    self._model = CrossEncoder(settings.reranker_model, max_length=settings.reranker_max_length,
                                               device=settings.embedding_device)
                    log.info("reranker loaded", extra={"model": settings.reranker_model})
                except Exception as exc:  # noqa: BLE001
                    self._failed = True
                    log.error("reranker unavailable - falling back to fusion ranking", extra={"error": str(exc)})
        return self._model

    @property
    def available(self) -> bool:
        return self._load() is not None

    def rerank(self, query: str, candidates: list[Candidate], top_n: int) -> list[Candidate]:
        if not candidates:
            return []
        with span("rerank", candidates=len(candidates)) as s:
            model = self._load()
            if model is None:
                s.set(fallback="fusion_order")
                return candidates[:top_n]
            try:
                pairs = [(query, f"{c.payload.get('section_title', '')}\n{c.payload.get('text', '')}")
                         for c in candidates]
                scores = model.predict(pairs, batch_size=16, show_progress_bar=False)
                for c, sc in zip(candidates, scores):
                    sc = float(sc)
                    c.rerank_score = sc if 0.0 <= sc <= 1.0 else 1 / (1 + math.exp(-sc))  # ensure probability
                ranked = sorted(candidates, key=lambda c: c.rerank_score or 0.0, reverse=True)[:top_n]
                s.set(top_score=round(ranked[0].rerank_score or 0, 4), kept=len(ranked))
                return ranked
            except Exception as exc:  # noqa: BLE001
                log.error("rerank failed - using fusion order", extra={"error": str(exc)})
                s.set(fallback="fusion_order")
                return candidates[:top_n]


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    return Reranker()
