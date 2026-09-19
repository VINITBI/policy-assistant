"""Hybrid retrieval: for every query variant, dense (bge) + sparse (BM25) search fused by
Qdrant RRF; results across variants are fused again with weighted Reciprocal Rank Fusion."""
from __future__ import annotations

from dataclasses import dataclass, field

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.observability import span
from app.db.vector_store import get_vector_store
from app.ingestion.embeddings import get_dense_embedder, get_sparse_embedder
from app.retrieval.query_optimizer import OptimizedQuery

log = get_logger(__name__)
settings = get_settings()
RRF_K = 60


@dataclass
class Candidate:
    id: str
    payload: dict
    fusion_score: float = 0.0
    rerank_score: float | None = None
    hits: list[str] = field(default_factory=list)


class HybridRetriever:
    def retrieve(self, oq: OptimizedQuery, policy_id: str | None) -> list[Candidate]:
        dense, sparse, store = get_dense_embedder(), get_sparse_embedder(), get_vector_store()
        pool: dict[str, Candidate] = {}
        runs: list[tuple[str, list[float], dict, float]] = []
        with span("embed_query", variants=len(oq.variants)):
            for i, q in enumerate(oq.variants):
                runs.append((q, dense.embed_query(q), sparse.embed_query(q), 1.0 if i == 0 else 0.7))
            if oq.hyde_passage:  # HyDE passage: embedded as a document, used for dense matching
                runs.append(("hyde", dense.embed_documents([oq.hyde_passage])[0], {"indices": [], "values": []}, 0.5))

        with span("hybrid_search", policy_id=policy_id or "all") as s:
            for label, dvec, svec, weight in runs:
                points = store.hybrid_search(dvec, svec, policy_id, settings.retrieval_candidates_per_query)
                for rank, p in enumerate(points, start=1):
                    c = pool.setdefault(str(p.id), Candidate(id=str(p.id), payload=p.payload or {}))
                    c.fusion_score += weight / (RRF_K + rank)
                    c.hits.append(label[:40])
            ranked = sorted(pool.values(), key=lambda c: c.fusion_score, reverse=True)[: settings.fused_candidates]
            s.set(unique_candidates=len(pool), returned=len(ranked))
        return ranked
