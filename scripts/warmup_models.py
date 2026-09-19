"""Download / warm up every local model once (embedding, reranker, injection classifier) and check Ollama.

    python scripts/warmup_models.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402

s = get_settings()
setup_logging(s.log_dir, "INFO")

from app.guardrails.input_guard import get_input_guard  # noqa: E402
from app.ingestion.embeddings import get_dense_embedder  # noqa: E402
from app.llm.ollama_client import get_llm  # noqa: E402
from app.retrieval.reranker import get_reranker  # noqa: E402

emb = get_dense_embedder()
print(f"[ok] embedding  {s.embedding_model}  dim={emb.dim}")
print(f"[{'ok' if get_reranker().available else 'FAIL'}] reranker   {s.reranker_model}")
score = get_input_guard().classifier.score("What is the leave policy?")
print(f"[{'ok' if score is not None else 'off'}] injection  {s.injection_model}")
h = get_llm().health()
print(f"[{'ok' if h.get('ok') else 'FAIL'}] ollama     {h}")
for m in (s.llm_model, s.llm_fallback_model, s.eval_judge_model):
    if h.get("ok") and not any(x.startswith(m) for x in h["models"]):
        print(f"      missing model -> run:  ollama pull {m}")
