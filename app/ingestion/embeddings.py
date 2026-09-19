"""Embedding models (shared by the ingestion and retrieval pipelines).

* Dense  : BAAI/bge-base-en-v1.5 via sentence-transformers (768-d, L2-normalised -> cosine)
* Sparse : BM25 term weights computed locally; Qdrant applies IDF server-side (Modifier.IDF).
           This gives exact-keyword matching (policy codes, clause numbers, acronyms) that
           dense vectors miss.
"""
from __future__ import annotations

import math
import re
import threading
import zlib
from collections import Counter
from functools import lru_cache

import snowballstemmer

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)
settings = get_settings()

_STOP = set("""a an and are as at be but by for from has have he her his i if in into is it its of on or our she
that the their them there these they this to was we were what when where which who will with you your can
do does did not no so than then too very just also shall may must""".split())
_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-/][a-z0-9]+)*")


class DenseEmbedder:
    def __init__(self, model_name: str, device: str = "cpu", query_instruction: str = "") -> None:
        from sentence_transformers import SentenceTransformer  # heavy import, lazy

        log.info("loading embedding model", extra={"model": model_name, "device": device})
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = int(self.model.get_sentence_embedding_dimension())
        self.query_instruction = query_instruction
        self.max_seq_length = int(getattr(self.model, "max_seq_length", 512) or 512)

    def count_tokens(self, text: str) -> int:
        return len(self.model.tokenizer.encode(text, add_special_tokens=False))

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        vecs = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        vec = self.model.encode([self.query_instruction + text], normalize_embeddings=True, show_progress_bar=False)
        return vec[0].tolist()


class BM25SparseEmbedder:
    """BM25 TF component per document; IDF is applied by Qdrant (sparse vector modifier=IDF)."""

    def __init__(self, k1: float = 1.2, b: float = 0.75, avg_len: float = 250.0) -> None:
        self.k1, self.b, self.avg_len = k1, b, avg_len
        self.stemmer = snowballstemmer.stemmer("english")

    def tokenize(self, text: str) -> list[str]:
        toks = [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]
        return self.stemmer.stemWords(toks)

    @staticmethod
    def _hash(token: str) -> int:
        return zlib.crc32(token.encode("utf-8")) & 0x7FFFFFFF

    def _vector(self, weights: dict[int, float]) -> dict[str, list]:
        return {"indices": list(weights.keys()), "values": list(weights.values())}

    def embed_document(self, text: str) -> dict[str, list]:
        toks = self.tokenize(text)
        tf = Counter(toks)
        dl = len(toks) or 1
        weights: dict[int, float] = {}
        for tok, f in tf.items():
            w = f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avg_len))
            h = self._hash(tok)
            weights[h] = weights.get(h, 0.0) + w
        return self._vector(weights)

    def embed_query(self, text: str) -> dict[str, list]:
        return self._vector({self._hash(t): 1.0 for t in set(self.tokenize(text))})


_lock = threading.Lock()


@lru_cache(maxsize=1)
def _dense() -> DenseEmbedder:
    return DenseEmbedder(settings.embedding_model, settings.embedding_device, settings.embedding_query_instruction)


def get_dense_embedder() -> DenseEmbedder:
    with _lock:
        return _dense()


@lru_cache(maxsize=1)
def get_sparse_embedder() -> BM25SparseEmbedder:
    return BM25SparseEmbedder()


def safe_token_counter():
    """Tokenizer-accurate counter if the model is available, else a word-based approximation."""
    try:
        return get_dense_embedder().count_tokens
    except Exception:  # noqa: BLE001
        log.warning("embedding tokenizer unavailable, using approximate token counts")
        return lambda t: max(1, math.ceil(len(t.split()) * 1.3))
