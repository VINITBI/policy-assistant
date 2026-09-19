"""Chat orchestration: sessions & memory, guardrails, RAG, persistence, audit, self-evaluation."""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.exceptions import AppError, GuardrailViolation, NotFoundError, ValidationError
from app.core.logging import get_logger, trace_id_var
from app.core.observability import FEEDBACK, RAG_QUERIES, start_trace
from app.db.models import ChatMessage, ChatSession, Policy, User
from app.db.session import db_scope
from app.guardrails.input_guard import get_input_guard
from app.llm.ollama_client import get_llm
from app.llm.prompts import get_prompts
from app.retrieval.pipeline import RAGPipeline
from app.services import audit
from app.services.audit import Actions
from app.services.trace_store import save_trace

log = get_logger(__name__)
settings = get_settings()
_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


# ---------------- sessions ----------------
def _get_session(db: Session, user: User, session_id: str) -> ChatSession:
    cs = db.get(ChatSession, session_id)
    if cs is None or cs.is_deleted or cs.user_id != user.id:
        raise NotFoundError("This conversation was not found.")
    return cs


def list_sessions(db: Session, user: User, limit: int = 50) -> list[ChatSession]:
    return list(db.scalars(select(ChatSession).where(ChatSession.user_id == user.id, ChatSession.is_deleted.is_(False))
                           .order_by(ChatSession.updated_at.desc()).limit(limit)))


def get_messages(db: Session, user: User, session_id: str) -> tuple[ChatSession, list[ChatMessage]]:
    cs = _get_session(db, user, session_id)
    return cs, list(cs.messages)


def delete_session(db: Session, user: User, session_id: str, ip: str) -> None:
    cs = _get_session(db, user, session_id)
    cs.is_deleted = True  # soft delete: kept for audit
    db.commit()
    audit.record(Actions.CHAT_SESSION_DELETE, db=db, user_id=user.id, username=user.username, role=user.role, ip=ip,
                 resource=f"chat_session:{session_id}")


def _history(cs: ChatSession) -> str:
    """Conversation memory: last N turns (assistant turns truncated) for condensation + answer prompt."""
    msgs = [m for m in cs.messages if m.answer_type != "blocked"][-settings.history_turns * 2:]
    lines = []
    for m in msgs:
        text = m.content if m.role == "user" else re.sub(r"\[S\d+\]", "", m.content)[:500]
        lines.append(f"{'User' if m.role == 'user' else 'Assistant'}: {text}")
    return "\n".join(lines)


# ---------------- main entry ----------------
def ask(db: Session, *, user: User, question: str, policy_id: str | None, session_id: str | None,
        ip: str = "") -> dict:
    t0 = time.perf_counter()
    trace = start_trace("chat_query")
    prompts = get_prompts()
    policy_id = None if policy_id in (None, "", "all") else policy_id

    active = db.scalars(select(Policy).where(Policy.status == "active")).all()
    if not active:
        raise AppError(prompts.message("no_policies"), code="NO_POLICIES", status_code=409)
    policy = None
    if policy_id:
        policy = db.get(Policy, policy_id)
        if policy is None or policy.status != "active":
            raise ValidationError("The selected policy is not available. Please choose another policy.")

    if session_id:
        cs = _get_session(db, user, session_id)
    else:
        cs = ChatSession(user_id=user.id, policy_id=policy_id, title=question.strip()[:60] or "New conversation")
        db.add(cs)
        db.flush()

    status = "ok"
    try:
        # ---- input guardrails ----
        try:
            check = get_input_guard().check(question, user_key=user.id)
        except GuardrailViolation as gv:
            RAG_QUERIES.labels(answer_type="blocked").inc()
            audit.record(Actions.GUARDRAIL_BLOCK, db=db, user_id=user.id, username=user.username, role=user.role,
                         ip=ip, outcome="blocked", resource=f"chat_session:{cs.id}",
                         details={"rule": gv.rule, "question": question[:200]})
            return _persist(db, cs, question, None, policy_id, blocked_msg=gv.user_message, flags=[gv.rule],
                            latency=(time.perf_counter() - t0) * 1000)

        # ---- RAG ----
        result = get_pipeline().run(check.clean_query, policy_id, _history(cs))
        result.flags = check.flags + result.flags
        result.warnings = check.warnings + result.warnings
        RAG_QUERIES.labels(answer_type=result.answer_type).inc()
        latency = (time.perf_counter() - t0) * 1000
        response = _persist(db, cs, check.clean_query, result, policy_id, latency=latency)
        audit.record(Actions.CHAT_QUERY, db=db, user_id=user.id, username=user.username, role=user.role, ip=ip,
                     resource=f"chat_session:{cs.id}",
                     details={"policy": policy.name if policy else "all", "answer_type": result.answer_type,
                              "confidence": result.confidence, "latency_ms": round(latency),
                              "question": check.clean_query[:200], "message_id": response["message_id"]})
        log.info("chat answered", extra={"answer_type": result.answer_type, "confidence": result.confidence,
                                         "latency_ms": round(latency), "sources": len(result.sources)})
        response["_eval"] = {"question": result.standalone_query, "context": result.context,
                             "answer": result.answer, "run": result.answer_type == "answered"}
        return response
    except Exception:
        status = "error"
        raise
    finally:
        save_trace(trace, username=user.username, status=status)


def _persist(db: Session, cs: ChatSession, question: str, result, policy_id: str | None, *, latency: float,
             blocked_msg: str | None = None, flags: list[str] | None = None) -> dict:
    tid = trace_id_var.get()
    db.add(ChatMessage(session_id=cs.id, role="user", content=question, policy_id=policy_id, trace_id=tid))
    if result is None:
        a = ChatMessage(session_id=cs.id, role="assistant", content=blocked_msg or "", policy_id=policy_id,
                        answer_type="blocked", confidence="", guardrail_flags=flags or [], latency_ms=latency,
                        trace_id=tid, prompt_version=get_prompts().version)
    else:
        a = ChatMessage(session_id=cs.id, role="assistant", content=result.answer, policy_id=policy_id,
                        standalone_query=result.standalone_query, sources=result.sources,
                        answer_type=result.answer_type, confidence=result.confidence,
                        guardrail_flags=result.flags, model=result.model, prompt_version=result.prompt_version,
                        latency_ms=latency, trace_id=tid)
    db.add(a)
    cs.updated_at = datetime.now(timezone.utc)
    db.commit()
    return {"session_id": cs.id, "message_id": a.id, "answer": a.content, "answer_type": a.answer_type,
            "confidence": a.confidence, "sources": a.sources or [],
            "warnings": (result.warnings if result else []), "standalone_query": a.standalone_query,
            "trace_id": tid, "latency_ms": round(latency, 1)}


# ---------------- self-evaluation (background) ----------------
SELF_EVAL_KEYS = ("faithfulness", "relevance", "completeness", "citation", "clarity")


def compute_self_eval(question: str, context: str, answer: str) -> dict:
    """Prompt-based self-evaluation: the LLM grades an answer on 5 criteria (1-5) -> overall 0..1."""
    res = get_llm().chat(get_prompts().messages("self_eval", question=question, context=context[:8000], answer=answer),
                         purpose="self_eval", temperature=0.0, json_format="json", max_tokens=256,
                         model=settings.eval_judge_model)
    data = json.loads(re.search(r"\{.*\}", res.text, re.S).group(0))
    scores: dict = {k: max(1, min(5, int(data.get(k, 1)))) for k in SELF_EVAL_KEYS}
    scores["overall"] = round(sum(scores[k] for k in SELF_EVAL_KEYS) / (5 * len(SELF_EVAL_KEYS)), 3)
    scores["issues"] = str(data.get("issues", ""))[:500]
    scores["judge"] = res.model
    return scores


def self_evaluate(message_id: str, question: str, context: str, answer: str) -> None:
    """Background task: self-evaluate an answer and store the scores on the message. Never raises."""
    if not settings.self_eval_enabled:
        return
    try:
        scores = compute_self_eval(question, context, answer)
        with db_scope() as db:
            msg = db.get(ChatMessage, message_id)
            if msg:
                msg.self_eval = scores
        log.info("self-evaluation stored", extra={"message_id": message_id, "overall": scores["overall"]})
    except Exception as exc:  # noqa: BLE001
        log.warning("self-evaluation failed", extra={"message_id": message_id, "error": str(exc)})


def feedback(db: Session, user: User, message_id: str, rating: int, comment: str, ip: str) -> None:
    msg = db.get(ChatMessage, message_id)
    if msg is None or msg.role != "assistant" or msg.session.user_id != user.id:
        raise NotFoundError("Message not found.")
    msg.feedback, msg.feedback_comment = (1 if rating > 0 else -1), comment[:1000]
    db.commit()
    FEEDBACK.labels(rating="up" if rating > 0 else "down").inc()
    audit.record(Actions.FEEDBACK, db=db, user_id=user.id, username=user.username, role=user.role, ip=ip,
                 resource=f"message:{message_id}", details={"rating": msg.feedback})
