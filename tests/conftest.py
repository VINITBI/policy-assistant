"""Test setup: isolated SQLite + embedded Qdrant, and lightweight fakes for the three ML models
(embedding, reranker, LLM) so the whole pipeline runs offline in seconds."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TMP = Path(tempfile.mkdtemp(prefix="policy_test_"))
os.environ.update({
    "DATABASE_URL": f"sqlite:///{TMP / 'test.db'}", "QDRANT_MODE": "local", "QDRANT_LOCAL_PATH": str(TMP / "qdrant"),
    "LOG_DIR": str(TMP / "logs"), "DATA_DIR": str(TMP), "POLICY_STORAGE_DIR": str(TMP / "policies"),
    "INBOX_DIR": str(TMP / "inbox"), "EVAL_REPORTS_DIR": str(TMP / "reports"), "INJECTION_MODEL_ENABLED": "false",
    "RATE_LIMIT_PER_MINUTE": "1000", "RERANK_MIN_SCORE": "0.15", "JWT_SECRET": "test-secret-test-secret-test-secret",
})

TOKEN = re.compile(r"[a-z0-9]+")


class FakeDense:
    model_name, dim = "fake-hash-embedder", 256

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for t in TOKEN.findall(text.lower()):
            v[int(hashlib.md5(t.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts, batch_size=32):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class FakeCrossEncoder:
    STOP = {"the", "a", "of", "to", "is", "i", "do", "how", "what", "many", "can", "my", "for", "in", "and", "get"}

    def predict(self, pairs, **_):
        out = []
        for q, d in pairs:
            qt = {t for t in TOKEN.findall(q.lower()) if t not in self.STOP}
            dt = set(TOKEN.findall(d.lower()))
            out.append(len(qt & dt) / max(len(qt), 1))
        return out


def fake_llm_call(self, model, messages, temperature, fmt, max_tokens):
    from app.llm.ollama_client import LLMResult

    system, user = messages[0]["content"], messages[-1]["content"]
    if "standalone search query" in system:
        q = user.split("Follow-up question:")[1].split("Standalone query:")[0].strip()
        text = q + " sick leave" if "that" in q.lower() else q
    elif "alternative phrasings" in system:
        text = "leave entitlement per year\nnumber of leave days allowed"
    elif "quality auditor" in system:
        text = json.dumps({"faithfulness": 5, "relevance": 5, "completeness": 4, "citation": 5, "clarity": 4,
                           "issues": ""})
    elif "<context>" in user:
        ctx = user.split("<context>")[1].split("</context>")[0]
        block = re.search(r"\[S1\][^\n]*\n(.+?)(?=\n\[S2\]|\Z)", ctx, re.S)
        question = set(TOKEN.findall(user.split("Question:")[-1].lower())) - FakeCrossEncoder.STOP
        sentences = re.split(r"(?<=[.])\s+", block.group(1).replace("\n", " ")) if block else []
        best = max(sentences, key=lambda s: len(question & set(TOKEN.findall(s.lower()))), default="")
        text = f"<think>internal</think>{best} [S1]" if best else "I couldn't find this in the selected policy."
    else:
        text = "ok"
    return LLMResult(text=text, model=model, latency_ms=1.0)


@pytest.fixture(scope="session", autouse=True)
def fake_models():
    from app.ingestion import embeddings
    from app.llm.ollama_client import OllamaClient
    from app.retrieval.reranker import Reranker

    mp = pytest.MonkeyPatch()
    fake = FakeDense()
    mp.setattr(embeddings, "_dense", lambda: fake)
    mp.setattr(Reranker, "_load", lambda self: FakeCrossEncoder())
    mp.setattr(OllamaClient, "_call", fake_llm_call)
    yield
    mp.undo()


@pytest.fixture(scope="session")
def sample_pdfs() -> Path:
    out = ROOT / "data" / "sample_policies"
    if not (out / "Leave_Policy.pdf").exists():
        import subprocess

        subprocess.run([sys.executable, str(ROOT / "scripts" / "create_sample_policies.py")], check=True)
    return out


@pytest.fixture(scope="session")
def client(fake_models):
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c
