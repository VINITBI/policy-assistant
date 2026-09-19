"""Qdrant vector store.

Collection design
-----------------
* named dense vector  "dense"  : cosine, HNSW (m=16, ef_construct=200), int8 scalar quantisation
                                 (4x less RAM, rescored with original vectors)
* named sparse vector "sparse" : BM25 term weights, IDF modifier applied by Qdrant
* payload indexes on policy_id / department / is_active / content_type for fast filtered search

Hybrid search = dense + sparse prefetch fused server-side with Reciprocal Rank Fusion.
"""
from __future__ import annotations

import threading
from contextlib import nullcontext
from functools import lru_cache
from typing import Any

from qdrant_client import QdrantClient, models

from app.core.config import get_settings
from app.core.exceptions import AppError, VectorStoreUnavailable
from app.core.logging import get_logger

log = get_logger(__name__)
settings = get_settings()

KEYWORD_INDEXES = ("policy_id", "policy_name", "department", "content_type", "doc_hash")


class VectorStore:
    def __init__(self) -> None:
        self.collection = settings.qdrant_collection
        try:
            if settings.qdrant_mode == "local":
                self.client = QdrantClient(path=str(settings.qdrant_local_path))
            else:
                self.client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=30)
        except Exception as exc:  # noqa: BLE001
            if "already accessed" in str(exc):
                raise VectorStoreUnavailable(
                    "The local policy database is already open in another program (usually the running API). "
                    "Stop the API before running scripts, or upload policies from the admin screen instead. "
                    "To allow both at once, run Qdrant in Docker and set QDRANT_MODE=server.",
                    detail=str(exc)) from exc
            raise VectorStoreUnavailable(detail=str(exc)) from exc
        self._lock = threading.RLock()
        # Embedded (local) Qdrant is not thread-safe: serialise every call. Server mode needs no lock.
        self._io = self._lock if settings.qdrant_mode == "local" else nullcontext()
        log.info("qdrant client ready", extra={"mode": settings.qdrant_mode, "collection": self.collection})

    # ---------- admin ----------
    def health(self) -> bool:
        try:
            with self._io:
                self.client.get_collections()
            return True
        except Exception:  # noqa: BLE001
            return False

    def ensure_collection(self, dim: int) -> None:
        with self._lock:
            try:
                if self.client.collection_exists(self.collection):
                    info = self.client.get_collection(self.collection)
                    vectors = info.config.params.vectors
                    existing = vectors["dense"].size if isinstance(vectors, dict) else vectors.size
                    if existing != dim:
                        raise AppError(
                            "The search index was built with a different embedding model. "
                            "An administrator must re-index all policies.",
                            detail=f"collection dim {existing} != model dim {dim}", code="EMBEDDING_DIM_MISMATCH")
                    return
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config={"dense": models.VectorParams(size=dim, distance=models.Distance.COSINE)},
                    sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)},
                    hnsw_config=models.HnswConfigDiff(m=settings.hnsw_m, ef_construct=settings.hnsw_ef_construct),
                    quantization_config=models.ScalarQuantization(
                        scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, quantile=0.99,
                                                               always_ram=True))
                    if settings.use_scalar_quantization else None,
                )
                if settings.qdrant_mode != "local":  # embedded Qdrant has no payload indexes (it scans instead)
                    for field in KEYWORD_INDEXES:
                        self.client.create_payload_index(self.collection, field, models.PayloadSchemaType.KEYWORD)
                    self.client.create_payload_index(self.collection, "is_active", models.PayloadSchemaType.BOOL)
                    self.client.create_payload_index(self.collection, "page_start", models.PayloadSchemaType.INTEGER)
                log.info("qdrant collection created", extra={"collection": self.collection, "dim": dim})
            except AppError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise VectorStoreUnavailable(detail=f"ensure_collection: {exc}") from exc

    # ---------- write ----------
    def upsert(self, points: list[dict[str, Any]], batch_size: int = 64) -> None:
        try:
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                with self._io:
                    self.client.upsert(self.collection, wait=True, points=[
                    models.PointStruct(id=p["id"], payload=p["payload"], vector={
                        "dense": p["dense"],
                        "sparse": models.SparseVector(indices=p["sparse"]["indices"], values=p["sparse"]["values"]),
                    }) for p in batch])
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreUnavailable(detail=f"upsert: {exc}") from exc

    def delete_policy(self, policy_id: str) -> None:
        try:
            with self._io:
                if self.client.collection_exists(self.collection):
                    self.client.delete(self.collection, points_selector=models.FilterSelector(filter=models.Filter(
                    must=[models.FieldCondition(key="policy_id", match=models.MatchValue(value=policy_id))])))
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreUnavailable(detail=f"delete: {exc}") from exc

    def count(self, policy_id: str | None = None) -> int:
        flt = self._filter(policy_id)
        with self._io:
            return self.client.count(self.collection, count_filter=flt, exact=True).count

    # ---------- read ----------
    @staticmethod
    def _filter(policy_id: str | None) -> models.Filter:
        must = [models.FieldCondition(key="is_active", match=models.MatchValue(value=True))]
        if policy_id:
            must.append(models.FieldCondition(key="policy_id", match=models.MatchValue(value=policy_id)))
        return models.Filter(must=must)

    def hybrid_search(self, dense: list[float], sparse: dict[str, list], policy_id: str | None,
                      limit: int) -> list[models.ScoredPoint]:
        flt = self._filter(policy_id)
        prefetch = [models.Prefetch(query=dense, using="dense", limit=limit, filter=flt,
                                    params=models.SearchParams(hnsw_ef=settings.search_hnsw_ef, quantization=
                                    models.QuantizationSearchParams(rescore=True, oversampling=2.0)))]
        if sparse.get("indices"):  # a query made only of stop-words has no sparse terms
            prefetch.append(models.Prefetch(query=models.SparseVector(indices=sparse["indices"],
                                                                      values=sparse["values"]),
                                            using="sparse", limit=limit, filter=flt))
        try:
            with self._io:
                res = self.client.query_points(
                self.collection,
                prefetch=prefetch,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit, with_payload=True,
            )
            return res.points
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreUnavailable(detail=f"hybrid_search: {exc}") from exc


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    return VectorStore()
