"""Retrieval-augmented generation pipeline (stateless - persistence lives in chat_service).

  optimise query -> hybrid retrieve -> cross-encoder rerank -> relevance gate
  -> small-to-big context build -> LLM generate (primary -> fallback -> extractive)
  -> sanitise + output guardrails -> confidence
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.core.config import get_settings
from app.core.exceptions import LLMUnavailable
from app.core.logging import get_logger
from app.core.observability import span
from app.guardrails.output_guard import OutputGuard, sanitize
from app.ingestion.embeddings import get_dense_embedder
from app.llm.ollama_client import get_llm
from app.llm.prompts import get_prompts
from app.retrieval.query_optimizer import QueryOptimizer
from app.retrieval.reranker import get_reranker
from app.retrieval.retriever import Candidate, HybridRetriever

log = get_logger(__name__)
settings = get_settings()


@dataclass
class RAGResult:
    answer: str
    answer_type: str  # answered | no_context | fallback_extractive | withheld
    standalone_query: str
    sources: list[dict] = field(default_factory=list)
    context: str = ""
    retrieval_context: list[str] = field(default_factory=list)  # for evaluation
    confidence: str = "low"
    top_score: float | None = None
    groundedness: float | None = None
    flags: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    model: str = ""
    prompt_version: str = ""


class RAGPipeline:
    def __init__(self) -> None:
        self.optimizer = QueryOptimizer()
        self.retriever = HybridRetriever()
        self.output_guard = OutputGuard()

    # ---------------- context ----------------
    def build_context(self, ranked: list[Candidate]) -> tuple[str, list[dict], list[str]]:
        """Small-to-big: each unique parent section becomes one numbered source [S#]."""
        count = get_dense_embedder().count_tokens
        budget = settings.context_max_tokens
        blocks, sources, contexts, seen_parents = [], [], [], set()
        for c in ranked:
            p = c.payload
            if p.get("parent_id") in seen_parents:
                continue
            body = p.get("parent_text") or p.get("text", "")
            tokens = count(body)
            if tokens > budget:  # parent too big for the remaining budget -> fall back to the child chunk
                body, tokens = p.get("text", ""), count(p.get("text", ""))
                if tokens > budget:
                    continue
            budget -= tokens
            seen_parents.add(p.get("parent_id"))
            ref = f"S{len(sources) + 1}"
            pages = f"p.{p.get('page_start')}" + (f"-{p.get('page_end')}" if p.get("page_end") != p.get("page_start") else "")
            section = " > ".join(p.get("section_path") or []) or p.get("section_title", "")
            blocks.append(f"[{ref}] ({p.get('policy_name')} > {section}, {pages})\n{body}")
            contexts.append(body)
            sources.append({"ref": ref, "policy_id": p.get("policy_id"), "policy_name": p.get("policy_name"),
                            "version": p.get("version"), "section": section, "pages": pages,
                            "score": round(c.rerank_score, 4) if c.rerank_score is not None else None,
                            "snippet": p.get("text", "")[:400], "chunk_id": p.get("chunk_id")})
        return "\n\n".join(blocks), sources, contexts

    @staticmethod
    def _confidence(top: float | None, grounded: float, cited: bool) -> str:
        top = 0.5 if top is None else top
        if top >= 0.6 and grounded >= 0.7 and cited:
            return "high"
        if top >= 0.3 and grounded >= 0.5:
            return "medium"
        return "low"

    # ---------------- main ----------------
    def run(self, question: str, policy_id: str | None, history: str) -> RAGResult:
        prompts = get_prompts()
        with span("query_optimization"):
            oq = self.optimizer.optimize(question, history)
        result = RAGResult(answer="", answer_type="no_context", standalone_query=oq.standalone,
                           prompt_version=prompts.version)

        candidates = self.retriever.retrieve(oq, policy_id)
        reranker = get_reranker()
        ranked = reranker.rerank(oq.standalone, candidates, settings.rerank_top_n)
        if ranked and ranked[0].rerank_score is not None:
            result.top_score = round(ranked[0].rerank_score, 4)
            ranked = [c for c in ranked if (c.rerank_score or 0) >= settings.rerank_min_score]

        if not ranked:
            log.info("relevance gate: nothing relevant found", extra={"top_score": result.top_score})
            result.answer = prompts.message("no_context")
            return result

        with span("build_context") as s:
            context, sources, contexts = self.build_context(ranked)
            s.set(sources=len(sources))
        result.sources, result.context, result.retrieval_context = sources, context, contexts

        try:
            with span("generate", model=settings.llm_model) as s:
                llm = get_llm().chat(prompts.messages("answer", history=history or "(none)", context=context,
                                                      question=oq.standalone), purpose="answer", max_tokens=800)
                s.set(model=llm.model, completion_tokens=llm.completion_tokens)
            result.model = llm.model
            if llm.model != settings.llm_model:
                result.flags.append(f"fallback_model:{llm.model}")
        except LLMUnavailable:
            log.error("LLM unavailable - returning extractive fallback answer")
            parts = [prompts.message("llm_down_extractive")]
            for src in sources[:3]:
                parts.append(f"\n**{src['section']}** ({src['pages']}) [{src['ref']}]\n> {src['snippet']}")
            result.answer = sanitize("\n".join(parts), len(sources))
            result.answer_type, result.confidence = "fallback_extractive", "low"
            result.flags.append("llm_unavailable")
            return result

        with span("output_guardrails") as s:
            check = self.output_guard.check(llm.text, context, len(sources))
            s.set(flags=check.flags, groundedness=check.groundedness)
        result.answer, result.groundedness = check.answer, check.groundedness
        result.flags += check.flags
        result.warnings += check.warnings
        if check.replaced:
            result.answer_type = "withheld"
        elif prompts.message("no_context").lower()[:40] in check.answer.lower():
            result.answer_type = "no_context"
        else:
            result.answer_type = "answered"
            cited = {f"S{i}" for i in check.cited}
            if cited:  # show the cited sources first
                result.sources.sort(key=lambda x: x["ref"] not in cited)
        result.confidence = "low" if result.answer_type != "answered" else self._confidence(
            result.top_score, check.groundedness, bool(check.cited))
        return result
