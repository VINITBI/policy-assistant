# Policy Assistant: RAG chatbot over company policies

> **New here? Open `START_HERE.md`.** Two double-clicks (`setup.bat`, then `run.bat`) get you running.

Employees ask questions about company policy PDFs and get grounded answers with citations. Admins upload and manage the policies. Everything is open source and runs on your own machine: FastAPI, Streamlit, Qdrant, bge embeddings and reranker, Ollama LLMs and DeepEval.

```
               ┌──────────────────────── INGESTION PIPELINE (admin) ────────────────────────┐
  PDF upload → │ validate → de-dup(sha256) → PyMuPDF layout parse → clean (headers/footers,  │
  / inbox      │ hyphenation, unicode) → heading hierarchy + tables → parent/child chunking  │
               │ → contextual headers → bge-base (768-d dense) + BM25 (sparse) → Qdrant      │
               └──────────────────────────────────────────────────────────────────────────────┘
               ┌──────────────────────── RETRIEVAL PIPELINE (user) ─────────────────────────┐
  question  →  │ INPUT GUARDRAILS → chat memory → condense follow-up → acronym expansion →   │
               │ multi-query (+HyDE opt.) → hybrid search (dense+sparse, RRF in Qdrant) →    │
               │ cross-variant RRF → bge-reranker-v2-m3 → relevance gate → small-to-big      │
               │ context → Ollama LLM (primary→fallback→extractive) → SANITIZE → OUTPUT      │
               │ GUARDRAILS (leakage, toxicity, PII, groundedness, numbers, citations) →     │
               │ confidence → persist + audit + trace → background LLM self-evaluation       │
               └──────────────────────────────────────────────────────────────────────────────┘
  Cross-cutting: JWT auth + login-session table · audit trail (DB + audit.log) · JSON logs with
  trace_id/user/session · per-stage spans stored per request · Prometheus /metrics · DeepEval suite
```

## Repository layout

```
policy-assistant/
├── app/
│   ├── main.py                 FastAPI app, middleware (trace-id, metrics), error handlers, /health /ready /metrics
│   ├── core/                   config · logging (JSON) · exceptions (user-friendly) · security (JWT/bcrypt) · observability
│   ├── db/                     models.py (SQLite tables) · session.py · vector_store.py (Qdrant)
│   ├── api/                    auth.py · policies.py · chat.py · admin.py · schemas.py · deps.py (RBAC)
│   ├── ingestion/              pdf_parser.py · chunker.py · embeddings.py · pipeline.py
│   ├── retrieval/              query_optimizer.py · retriever.py · reranker.py · pipeline.py
│   ├── llm/                    ollama_client.py (retry + fallback) · prompts.py (YAML loader)
│   ├── guardrails/             input_guard.py · output_guard.py (sanitiser) · patterns.py
│   ├── services/               auth_service · chat_service (sessions/memory/self-eval) · audit · trace_store
│   └── evaluation/             judge.py (DeepEval ↔ Ollama) · runner.py (metrics suite)
├── ui/                         app.py (Streamlit) · api_client.py
├── prompts/prompts.yaml        ALL prompts + user-facing messages (versioned)
├── eval/golden_dataset.yaml    golden Q&A + guardrail test cases
├── scripts/                    start_all · ingest_folder · run_evaluation · generate_golden_dataset · warmup_models · create_sample_policies
├── tests/                      offline test-suite (18 tests; models stubbed)
├── data/                       sample_policies/ · inbox/ (bulk load) · policies/ (stored uploads) · app.db
├── logs/                       app.log · error.log · audit.log (JSON, rotated)
├── docker-compose.yml          Qdrant (+ optional Ollama)
├── requirements.txt · .env.example
```

## Setup (Windows / macOS / Linux)

Prerequisites: Python 3.10+, Docker Desktop (for Qdrant) and [Ollama](https://ollama.com). If you'd rather not run Docker, set `QDRANT_MODE=local` in `.env`. Qdrant then runs embedded on disk and no server is needed.

```bash
cd policy-assistant
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                    # Windows: copy .env.example .env   -> then edit JWT_SECRET & passwords

docker compose up -d qdrant             # vector DB on :6333 (dashboard: http://localhost:6333/dashboard)
ollama pull llama3.1:8b                 # answer model
ollama pull qwen2.5:3b                  # fallback model
ollama pull qwen2.5:7b                  # judge for self-eval & DeepEval

python scripts/warmup_models.py         # downloads bge embedding/reranker + injection classifier once, checks Ollama
python scripts/start_all.py             # API :8000 + UI :8501
```

* UI: http://localhost:8501
* Swagger: http://localhost:8000/docs. Click **Authorize** and enter `admin` / `Admin@123` (or `employee` / `Employee@123`).

These two accounts are created on first start from `.env`. Change their passwords before you share the app. Admins can add more users from the UI (Users page) or with `POST /api/v1/admin/users`.

### Load your policies

You can load them in three ways:

1. **UI:** log in as admin, open **Upload policy**, pick the PDF and fill in name, department, version, effective date and tags.
2. **Bulk from your folder:** `python scripts/ingest_folder.py --folder "C:\Users\me\Documents\Policies" --department HR`
3. **Inbox:** copy the PDFs into `data/inbox/`, then click **Ingest all PDFs in inbox** (or call `POST /admin/policies/ingest-inbox`).

To try the app before using real policies: `python scripts/create_sample_policies.py`, then `python scripts/ingest_folder.py --folder data/sample_policies`. The golden dataset is written for these samples.

## How each requirement is met

| # | Requirement | Implementation |
|---|---|---|
| 1 | Two pipelines | `app/ingestion/pipeline.py` (background job and CLI) and `app/retrieval/pipeline.py` are fully separate. |
| 2 | Chunking, embeddings, Qdrant, indexing, metadata | **Chunking:** the parser detects headings from font size, bold text, numbering and ALL-CAPS lines. Chunks never cross a section. Parent chunks are about 1200 tokens and child chunks about 350 tokens with 60 tokens of overlap, split on sentence boundaries. Tables are kept as Markdown, and the header row repeats when a table is split. Every child gets a "Policy > Section" header before embedding. **Embeddings:** `BAAI/bge-base-en-v1.5` (768-d, normalized, with a query instruction) plus BM25 sparse vectors. **Qdrant:** named dense and sparse vectors, HNSW (m=16, ef_construct=200), int8 scalar quantization with rescoring, IDF modifier, and payload indexes on policy_id, department, is_active, content_type and doc_hash. **Metadata per chunk:** policy id, name, version, department, effective date, tags, file, hash, section path, page range, parent id and text, token count, ingested by/at, embedding model. |
| 3 | Query optimization, retrieval, reranking, LLM, prompting, sanitization | **Query optimization:** follow-up questions are rewritten into standalone ones, acronyms are expanded (WFH, LOP, PTO…), multi-query is on and HyDE is optional. **Retrieval:** hybrid dense + sparse search with RRF inside Qdrant, then weighted RRF across query variants. **Reranking:** `bge-reranker-v2-m3` cross-encoder with a relevance threshold. **Context:** each matched child is swapped for its parent section, within a token budget. **LLM:** Ollama `llama3.1:8b`, falling back to `qwen2.5:3b`. **Prompt:** role prompt, grounding rules, XML delimiters, a citation contract, a defense against injected instructions, and a one-shot example. **Sanitizer:** strips `<think>` blocks, HTML/scripts, role markers and invalid `[S#]` citations, and caps length. |
| 4 | Guardrails with friendly messages | **Input:** length, gibberish, rate limit, toxicity, harmful intent, prompt injection (regex plus the open-source DeBERTa classifier), and PII/secret masking. **Output:** prompt leakage, toxicity, PII redaction, groundedness score, unverified numbers, missing citations. A blocked message appears in the chat with a polite explanation. |
| 5 & 14 | Evaluation with DeepEval | Retrieval: contextual precision, recall and relevancy (DeepEval), plus hit-rate@k, MRR and keyword recall. Generation: answer relevancy, faithfulness, hallucination. LLM quality: G-Eval correctness, citation quality and tone, plus bias and toxicity. **Prompt self-evaluation:** DeepEval `PromptAlignmentMetric` against the instructions in `prompts.yaml`, plus the runtime LLM self-grading score. Also scored: refusal accuracy and guardrail accuracy. All judging uses local Ollama, with no API keys. |
| 6, 11 | Detailed logging | JSON logs in `logs/app.log`, `error.log` and `audit.log`. Every line carries `trace_id`, `user` and `login_session`, plus stage timings, model names and token counts. Files rotate at 10 MB × 5. |
| 7 | FastAPI + Swagger | `/docs`, with OAuth2 "Authorize" through `/auth/token`. |
| 8, 9 | Role-based UI with login | Streamlit. Admins get Upload, Manage, Ask, Audit, Login sessions, Observability, Evaluation and Users pages. Users get a policy selector and chat box only. Upload endpoints return 403 for users, and the attempt is audited. |
| 10, 22 | Login tracking | `login_sessions` table: login, last seen, logout or expiry time, IP, user agent and request count. The JWT `jti` is the session id, so logout revokes it on the server. Failed logins and lockouts (5 attempts → 15 minutes) are audited. |
| 12 | Observability, traceability, auditability | Per-request traces with the duration of every pipeline stage are stored in `request_traces` and viewable in the Observability page. The `X-Trace-ID` header appears on every response and error. Prometheus exposes `/metrics`. The `audit_logs` table records every security-relevant action and can be downloaded as CSV. |
| 13 | PDF policies | PyMuPDF parser. Rejects encrypted, corrupt, image-only and oversized files with a clear message. |
| 15 | Guardrails | Custom and open source, with no paid services. |
| 16 | Local only | All components are open source and run offline once the models are downloaded. DeepEval telemetry is turned off. |
| 17, 18 | Simple Python repo | See the layout above. |
| 19 | Prompts in YAML | `prompts/prompts.yaml`, versioned. The prompt version is stored with every answer. |
| 20 | Error handling and fallback | Typed errors with a user-safe message, code and trace id. Fallbacks: LLM primary → fallback model → extractive answer; reranker → fusion order; query rewrite → original query; self-eval and audit never break a request; a failed ingestion rolls back its partial vectors. |
| 21, 23 | Chat memory and history | `chat_sessions` and `chat_messages`. The last 6 turns feed the query rewrite and the prompt. The sidebar lists earlier conversations, which reopen with one click. Deletion is soft, so the audit trail survives. |

Also included:
- Uploading a new version of a policy with the same name replaces the old version.
- Duplicate files are detected by their hash.
- Thumbs up/down feedback.
- Confidence badges.
- Sources with section and page numbers.
- A relevance gate that answers "not found in policy" without calling the LLM.
- A script that bootstraps a synthetic golden dataset.
- A CI quality gate: `run_evaluation.py --fail-under 0.7`.

## API overview

| Method | Path | Role |
|---|---|---|
| POST | `/api/v1/auth/login` · `/auth/token` · `/auth/logout` · GET `/auth/me` | any |
| GET | `/api/v1/policies` | user/admin |
| POST | `/api/v1/chat/query` `{question, policy_id?, session_id?}` | user/admin |
| GET/DELETE | `/api/v1/chat/sessions` · `/chat/sessions/{id}/messages` | owner |
| POST | `/api/v1/chat/messages/{id}/feedback` | owner |
| POST/GET/DELETE | `/api/v1/admin/policies` · `/{id}/reindex` · `/ingest-inbox` · `/admin/ingestion-jobs` | admin |
| GET | `/api/v1/admin/audit-logs` · `/login-sessions` · `/traces[/{id}]` · `/stats` · `/users` | admin |
| POST/GET | `/api/v1/admin/evaluations` | admin |
| GET | `/health` · `/ready` (checks DB, Qdrant and Ollama) · `/metrics` | public |

Every error has the same shape: `{"error": {"code": "GUARDRAIL_PROMPT_INJECTION", "message": "<friendly text>", "trace_id": "..."}}`.

## Evaluation

```bash
python scripts/run_evaluation.py --max-cases 3        # quick smoke test
python scripts/run_evaluation.py                      # full suite -> eval/reports/eval_<ts>.json + .csv
python scripts/generate_golden_dataset.py --per-policy 5   # draft Q&A from your own policies (review before use)
```

You can also run the suite from the admin UI (**Evaluation** page), which keeps the history of runs. Each report records the model, prompt version, chunking settings and policy versions, so you can trace every score to the exact setup that produced it.

## Tuning tips

- **Slow CPU:** set `RERANKER_MODEL=BAAI/bge-reranker-base`, `MULTI_QUERY_ENABLED=false` and `LLM_MODEL=qwen2.5:3b`. Every LLM call costs a few seconds locally.
- **GPU:** set `EMBEDDING_DEVICE=cuda`. Ollama uses the GPU automatically.
- **Changing the embedding model:** the app refuses to mix vector sizes (`EMBEDDING_DIM_MISMATCH`). Delete the collection, or change `QDRANT_COLLECTION`, then re-index.
- **Too many "not found" answers:** lower `RERANK_MIN_SCORE`. **Answers going beyond the policy:** raise `GROUNDEDNESS_MIN`.
- **Scanned PDFs:** run OCR first (for example `ocrmypdf in.pdf out.pdf`). The app rejects PDFs that contain only images.

## Tests

```bash
pytest -q        # 18 tests, runs offline in seconds: embedded Qdrant, temp SQLite, stub models
```

The tests cover parsing, chunking, sparse vectors, input and output guardrails, the sanitizer, and the full API flow: role checks, upload and ingestion, chat with memory, blocked prompts, the relevance gate, history, feedback, audit, traces, logout revocation, lockout, and the evaluation runner.
