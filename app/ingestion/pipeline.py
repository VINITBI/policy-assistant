"""Ingestion pipeline:  validate -> store -> parse -> clean -> chunk -> embed (dense+sparse) -> index.

Runs as a background job; progress and failures are tracked in `ingestion_jobs`,
every stage is traced, and the outcome is written to the audit trail.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.exceptions import AppError, DuplicateError, IngestionError, NotFoundError, ValidationError
from app.core.logging import get_logger, trace_id_var, username_var
from app.core.observability import INGESTED_CHUNKS, new_trace_id, span, start_trace
from app.db.models import IngestionJob, Policy
from app.db.session import db_scope
from app.db.vector_store import get_vector_store
from app.ingestion.chunker import HierarchicalChunker
from app.ingestion.embeddings import get_dense_embedder, get_sparse_embedder
from app.ingestion.pdf_parser import parse_pdf
from app.services import audit
from app.services.audit import Actions
from app.services.trace_store import save_trace

log = get_logger(__name__)
settings = get_settings()


def _safe_filename(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).name)[:120]
    return stem if stem.lower().endswith(".pdf") else stem + ".pdf"


def validate_upload(filename: str, content: bytes) -> None:
    if not filename or not filename.lower().endswith(".pdf"):
        raise ValidationError("Only PDF files can be uploaded.")
    if not content:
        raise ValidationError("The uploaded file is empty.")
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise ValidationError(f"The file is larger than the {settings.max_upload_mb} MB limit.")
    if not content[:1024].lstrip().startswith(b"%PDF"):
        raise ValidationError("The file does not look like a valid PDF.")


def register_policy(db: Session, *, filename: str, content: bytes, name: str, department: str, version: str,
                    effective_date: str, description: str, tags: list[str], uploaded_by: str) -> tuple[Policy, IngestionJob]:
    """Validate, de-duplicate, store the file and create the policy + job rows."""
    validate_upload(filename, content)
    name = (name or Path(filename).stem.replace("_", " ")).strip()[:200]
    file_hash = hashlib.sha256(content).hexdigest()
    dup = db.scalar(select(Policy).where(Policy.file_hash == file_hash, Policy.status.in_(("active", "processing"))))
    if dup:
        raise DuplicateError(f"This exact document is already uploaded as '{dup.name}' (v{dup.version}).")

    policy = Policy(name=name, department=department or "General", version=version or "1.0",
                    effective_date=effective_date or "", description=description or "",
                    tags=[t.strip() for t in tags if t.strip()], file_name=_safe_filename(filename),
                    file_path="", file_hash=file_hash, file_size=len(content), uploaded_by=uploaded_by)
    db.add(policy)
    db.flush()
    target_dir = settings.policy_storage_dir / policy.id
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / policy.file_name
    path.write_bytes(content)
    policy.file_path = str(path)
    job = IngestionJob(policy_id=policy.id, triggered_by=uploaded_by, trace_id=trace_id_var.get())
    db.add(job)
    db.commit()
    log.info("policy registered", extra={"policy_id": policy.id, "policy": name, "bytes": len(content)})
    return policy, job


def run_ingestion(policy_id: str, job_id: str, triggered_by: str) -> None:
    """Background job. Never raises: failures are recorded on the job/policy and audited."""
    trace_id_var.set(new_trace_id())
    username_var.set(triggered_by)
    trace = start_trace("ingestion")
    store = None
    stats: dict = {}
    with db_scope() as db:
        policy = db.get(Policy, policy_id)
        job = db.get(IngestionJob, job_id)
        if policy is None or job is None:
            log.error("ingestion job references missing rows", extra={"policy_id": policy_id, "job_id": job_id})
            return
        job.status, job.started_at, job.trace_id = "running", datetime.now(timezone.utc), trace.trace_id
        policy.status = "processing"
        db.commit()
        try:
            store = get_vector_store()
            with span("parse_pdf", file=policy.file_name) as s:
                doc = parse_pdf(Path(policy.file_path), settings.max_pdf_pages)
                if doc.char_count < 100:
                    raise IngestionError("Too little text could be extracted. The PDF may be scanned.")
                s.set(pages=doc.page_count, blocks=len(doc.blocks))
            stats.update(pages=doc.page_count, **doc.metadata)

            dense = get_dense_embedder()
            sparse = get_sparse_embedder()
            with span("chunk") as s:
                chunker = HierarchicalChunker(dense.count_tokens, settings.child_chunk_tokens,
                                              settings.child_chunk_overlap_tokens, settings.parent_chunk_tokens,
                                              settings.min_chunk_tokens)
                chunks = chunker.chunk(doc, policy.id, policy.name)
                if not chunks:
                    raise IngestionError("No content could be extracted from this PDF.")
                s.set(chunks=len(chunks), parents=len({c.parent_id for c in chunks}))
            stats.update(chunks=len(chunks), parents=len({c.parent_id for c in chunks}),
                         avg_chunk_tokens=round(sum(c.token_count for c in chunks) / len(chunks), 1))

            with span("embed", chunks=len(chunks), model=dense.model_name):
                texts = [c.embed_text for c in chunks]
                dense_vecs = dense.embed_documents(texts, settings.embedding_batch_size)
                sparse_vecs = [sparse.embed_document(t) for t in texts]

            now = datetime.now(timezone.utc).isoformat()
            points = [{
                "id": c.chunk_id, "dense": dv, "sparse": sv,
                "payload": {
                    "policy_id": policy.id, "policy_name": policy.name, "department": policy.department,
                    "version": policy.version, "effective_date": policy.effective_date, "tags": policy.tags,
                    "file_name": policy.file_name, "doc_hash": policy.file_hash, "doc_title": doc.title,
                    "chunk_id": c.chunk_id, "chunk_index": c.chunk_index, "parent_id": c.parent_id,
                    "parent_text": c.parent_text, "text": c.text, "section_title": c.section_title,
                    "section_path": c.section_path, "page_start": c.page_start, "page_end": c.page_end,
                    "token_count": c.token_count, "content_type": c.content_type, "is_active": True,
                    "ingested_at": now, "ingested_by": triggered_by, "embedding_model": dense.model_name,
                }} for c, dv, sv in zip(chunks, dense_vecs, sparse_vecs)]

            with span("index", points=len(points)):
                store.ensure_collection(dense.dim)
                store.delete_policy(policy.id)  # idempotent re-index
                store.upsert(points)

            superseded = _supersede_old_versions(db, policy)
            policy.status, policy.error = "active", ""
            policy.page_count, policy.chunk_count = doc.page_count, len(chunks)
            policy.indexed_at = datetime.now(timezone.utc)
            job.status, job.stats = "succeeded", {**stats, "superseded": superseded}
            INGESTED_CHUNKS.inc(len(chunks))
            audit.record(Actions.POLICY_INGESTED, db=db, username=triggered_by, resource=f"policy:{policy.id}",
                         details={"name": policy.name, **job.stats})
            log.info("ingestion succeeded", extra={"policy_id": policy.id, **stats})
        except Exception as exc:  # noqa: BLE001
            user_msg = exc.user_message if isinstance(exc, AppError) else "Unexpected error while processing the PDF."
            log.exception("ingestion failed", extra={"policy_id": policy.id})
            try:
                if store is not None:
                    store.delete_policy(policy.id)  # roll back partial index
            except Exception:  # noqa: BLE001
                log.warning("rollback of partial index failed", extra={"policy_id": policy.id})
            policy.status, policy.error = "failed", user_msg
            job.status, job.error, job.stats = "failed", f"{user_msg} ({type(exc).__name__}: {exc})"[:2000], stats
            audit.record(Actions.POLICY_INGEST_FAILED, db=db, username=triggered_by, outcome="failure",
                         resource=f"policy:{policy.id}", details={"error": user_msg})
        finally:
            job.finished_at = datetime.now(timezone.utc)
            db.commit()
            save_trace(trace, username=triggered_by, status="ok" if job.status == "succeeded" else "error")


def _supersede_old_versions(db: Session, policy: Policy) -> list[str]:
    """A newly indexed document with the same policy name replaces older active versions."""
    olds = db.scalars(select(Policy).where(func.lower(Policy.name) == policy.name.lower(),
                                           Policy.id != policy.id, Policy.status == "active")).all()
    for old in olds:
        get_vector_store().delete_policy(old.id)
        old.status = "superseded"
        log.info("policy version superseded", extra={"old_policy_id": old.id, "new_policy_id": policy.id})
    return [o.id for o in olds]


def delete_policy(db: Session, policy_id: str) -> Policy:
    policy = db.get(Policy, policy_id)
    if policy is None or policy.status == "deleted":
        raise NotFoundError("Policy not found.")
    get_vector_store().delete_policy(policy_id)
    policy.status = "deleted"
    db.commit()
    return policy


def prepare_reindex(db: Session, policy_id: str, triggered_by: str) -> IngestionJob:
    policy = db.get(Policy, policy_id)
    if policy is None or policy.status == "deleted":
        raise NotFoundError("Policy not found.")
    if not Path(policy.file_path).exists():
        raise IngestionError("The original PDF is missing from storage; please upload it again.")
    job = IngestionJob(policy_id=policy.id, triggered_by=triggered_by, trace_id=trace_id_var.get())
    policy.status = "processing"
    db.add(job)
    db.commit()
    return job


def register_inbox(db: Session, triggered_by: str) -> tuple[list[tuple[str, str]], list[dict]]:
    """Register every PDF found in data/inbox (bulk load of policies already on the local machine)."""
    queued, skipped = [], []
    for pdf in sorted(settings.inbox_dir.glob("*.pdf")):
        try:
            policy, job = register_policy(db, filename=pdf.name, content=pdf.read_bytes(),
                                          name=pdf.stem.replace("_", " ").replace("-", " ").title(), department="General",
                                          version="1.0", effective_date="", description="", tags=[],
                                          uploaded_by=triggered_by)
            queued.append((policy.id, job.id))
        except AppError as exc:
            db.rollback()
            skipped.append({"file": pdf.name, "reason": exc.user_message})
    return queued, skipped
