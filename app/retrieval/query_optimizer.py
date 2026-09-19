"""Query optimisation:
1. normalisation + HR acronym expansion (deterministic, free)
2. conversational condensation -> standalone query (uses chat history; LLM)
3. multi-query expansion -> N paraphrases for better recall (LLM)
4. optional HyDE hypothetical passage for dense retrieval (LLM)
Every LLM step falls back to the original query on failure.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.observability import span
from app.llm.ollama_client import get_llm
from app.llm.prompts import get_prompts

log = get_logger(__name__)
settings = get_settings()

ACRONYMS = {
    "wfh": "work from home", "pto": "paid time off", "lop": "loss of pay", "lwp": "leave without pay",
    "el": "earned leave", "cl": "casual leave", "sl": "sick leave", "ml": "maternity leave",
    "hr": "human resources", "posh": "prevention of sexual harassment", "coc": "code of conduct",
    "reimb": "reimbursement", "t&e": "travel and expense", "ctc": "cost to company", "nda": "non-disclosure agreement",
    "byod": "bring your own device", "ooo": "out of office", "pip": "performance improvement plan",
}


@dataclass
class OptimizedQuery:
    original: str
    standalone: str
    variants: list[str] = field(default_factory=list)  # includes standalone first
    hyde_passage: str | None = None


def normalize(q: str) -> str:
    q = re.sub(r"\s+", " ", q).strip()
    extra = [full for abbr, full in ACRONYMS.items() if re.search(rf"(?<![\w&]){re.escape(abbr)}(?![\w&])", q.lower())
             and full not in q.lower()]
    return f"{q} ({'; '.join(extra)})" if extra else q


def _one_line(text: str) -> str:
    text = re.sub(r"(?is)<think>.*?</think>", "", text).strip().strip('"').strip()
    return text.splitlines()[0].strip() if text else ""


class QueryOptimizer:
    def condense(self, question: str, history: str) -> str:
        if not settings.query_rewrite_enabled or not history.strip():
            return question
        with span("query_condense") as s:
            try:
                res = get_llm().chat(get_prompts().messages("condense", history=history, question=question),
                                     purpose="condense", temperature=0.0, max_tokens=96)
                out = _one_line(res.text)
                if 3 <= len(out) <= max(300, len(question) * 4):
                    s.set(rewritten=out)
                    return out
            except Exception as exc:  # noqa: BLE001
                s.set(fallback=True)
                log.warning("query condensation failed; using original", extra={"error": str(exc)})
        return question

    def expand(self, query: str) -> list[str]:
        if not settings.multi_query_enabled or settings.multi_query_count <= 0:
            return []
        with span("multi_query") as s:
            try:
                res = get_llm().chat(get_prompts().messages("multi_query", n=settings.multi_query_count, question=query),
                                     purpose="multi_query", temperature=0.3, max_tokens=160)
                lines = [re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", l).strip().strip('"')
                         for l in re.sub(r"(?is)<think>.*?</think>", "", res.text).splitlines()]
                seen, out = {query.lower()}, []
                for l in lines:
                    if 5 <= len(l) <= 300 and l.lower() not in seen:
                        seen.add(l.lower())
                        out.append(l)
                s.set(variants=out[: settings.multi_query_count])
                return out[: settings.multi_query_count]
            except Exception as exc:  # noqa: BLE001
                s.set(fallback=True)
                log.warning("multi-query expansion failed; continuing with single query", extra={"error": str(exc)})
                return []

    def hyde(self, query: str) -> str | None:
        if not settings.hyde_enabled:
            return None
        with span("hyde"):
            try:
                return get_llm().chat(get_prompts().messages("hyde", question=query), purpose="hyde",
                                      temperature=0.2, max_tokens=200).text.strip() or None
            except Exception as exc:  # noqa: BLE001
                log.warning("HyDE failed", extra={"error": str(exc)})
                return None

    def optimize(self, question: str, history: str) -> OptimizedQuery:
        standalone = normalize(self.condense(question, history))
        variants = [standalone] + self.expand(standalone)
        return OptimizedQuery(original=question, standalone=standalone, variants=variants,
                              hyde_passage=self.hyde(standalone))
