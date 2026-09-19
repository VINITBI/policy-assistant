"""Central application settings, loaded from environment variables / .env file."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", env_file_encoding="utf-8", extra="ignore")

    # ---- App ----
    app_name: str = "Policy Assistant"
    app_env: str = "local"
    api_prefix: str = "/api/v1"
    log_level: str = "INFO"
    log_dir: Path = BASE_DIR / "logs"
    data_dir: Path = BASE_DIR / "data"
    policy_storage_dir: Path = BASE_DIR / "data" / "policies"
    inbox_dir: Path = BASE_DIR / "data" / "inbox"
    prompts_file: Path = BASE_DIR / "prompts" / "prompts.yaml"
    database_url: str = f"sqlite:///{BASE_DIR / 'data' / 'app.db'}"

    # ---- Security / auth ----
    jwt_secret: str = "CHANGE-ME-use-a-long-random-string"
    jwt_algorithm: str = "HS256"
    access_token_minutes: int = 60
    max_failed_logins: int = 5
    lockout_minutes: int = 15
    bootstrap_admin_username: str = "admin"
    bootstrap_admin_password: str = "Admin@123"
    bootstrap_user_username: str = "employee"
    bootstrap_user_password: str = "Employee@123"
    rate_limit_per_minute: int = 20

    # ---- Upload limits ----
    max_upload_mb: int = 25
    max_pdf_pages: int = 500

    # ---- Qdrant ----
    qdrant_mode: str = "server"  # "server" (docker) | "local" (embedded, on-disk)
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_local_path: Path = BASE_DIR / "data" / "qdrant_local"
    qdrant_collection: str = "policy_chunks"
    hnsw_m: int = 16
    hnsw_ef_construct: int = 200
    search_hnsw_ef: int = 128
    use_scalar_quantization: bool = True

    # ---- Embeddings / reranker (open-source, HuggingFace) ----
    embedding_model: str = "BAAI/bge-base-en-v1.5"  # 768-dim
    embedding_query_instruction: str = "Represent this sentence for searching relevant passages: "
    embedding_batch_size: int = 32
    embedding_device: str = "cpu"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_enabled: bool = True
    reranker_max_length: int = 512

    # ---- Chunking ----
    child_chunk_tokens: int = 350
    child_chunk_overlap_tokens: int = 60
    parent_chunk_tokens: int = 1200
    min_chunk_tokens: int = 25

    # ---- Retrieval ----
    retrieval_candidates_per_query: int = 20  # per retriever (dense / sparse) per query variant
    fused_candidates: int = 20  # handed to the reranker
    rerank_top_n: int = 5
    rerank_min_score: float = 0.15  # below this -> "not in policy"
    context_max_tokens: int = 3000
    history_turns: int = 6
    query_rewrite_enabled: bool = True
    multi_query_enabled: bool = True
    multi_query_count: int = 2
    hyde_enabled: bool = False

    # ---- LLM (Ollama) ----
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.1:8b"
    llm_fallback_model: str = "qwen2.5:3b"
    llm_temperature: float = 0.1
    llm_num_ctx: int = 8192
    llm_timeout_seconds: int = 120
    llm_max_retries: int = 2

    # ---- Guardrails ----
    max_query_chars: int = 1000
    min_query_chars: int = 3
    injection_model_enabled: bool = True
    injection_model: str = "protectai/deberta-v3-base-prompt-injection-v2"
    injection_threshold: float = 0.90
    groundedness_min: float = 0.35  # lexical support ratio; below -> answer withheld
    groundedness_warn: float = 0.55

    # ---- Self-evaluation / evaluation ----
    self_eval_enabled: bool = True
    eval_judge_model: str = "qwen2.5:7b"
    eval_dataset: Path = BASE_DIR / "eval" / "golden_dataset.yaml"
    eval_reports_dir: Path = BASE_DIR / "eval" / "reports"
    eval_threshold: float = 0.6

    # ---- UI ----
    api_base_url: str = Field(default="http://localhost:8000")

    def ensure_dirs(self) -> None:
        for d in (self.log_dir, self.data_dir, self.policy_storage_dir, self.inbox_dir, self.eval_reports_dir):
            Path(d).mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
