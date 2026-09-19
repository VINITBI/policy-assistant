"""Response sanitisation + output guardrails.

Sanitisation (always):
  strip reasoning tags (<think>), HTML/script, role markers and prompt echoes, invalid citation tags,
  normalise whitespace, cap length.
Guardrails:
  * system-prompt leakage          -> answer replaced
  * toxicity                        -> answer replaced
  * high-risk PII not in the source -> redacted
  * groundedness (lexical support of each sentence by the context) -> withheld / warning
  * numbers not present in the context -> warning (numeric hallucination)
  * missing citations               -> warning
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass, field

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.observability import GUARDRAIL_EVENTS
from app.guardrails.patterns import HIGH_RISK_PII, LEAKAGE_MARKERS, find_pii, mask_pii, toxicity_hits
from app.ingestion.embeddings import get_sparse_embedder
from app.llm.prompts import get_prompts

log = get_logger(__name__)
settings = get_settings()

CITATION = re.compile(r"\[S(\d+)\]")
SENTENCES = re.compile(r"(?<=[.!?])\s+|\n+")
NUMBER = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?%?")


@dataclass
class OutputCheck:
    answer: str
    groundedness: float = 1.0
    cited: list[int] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    replaced: bool = False


def sanitize(text: str, max_sources: int, max_chars: int = 6000) -> str:
    text = re.sub(r"(?is)<think>.*?</think>", "", text or "")
    text = re.sub(r"(?is)<(script|style|iframe)[^>]*>.*?</\1>", "", text)
    text = re.sub(r"(?is)</?(context|history|system|user|assistant)[^>]*>", "", text)
    text = re.sub(r"<[^>\n]{1,200}>", "", text)  # remaining html tags
    text = html.unescape(text)
    text = re.sub(r"(?im)^\s*(answer|assistant|response)\s*:\s*", "", text)
    text = re.sub(r"(?i)\b(as an ai (language )?model|i am an ai)[^.]*\.\s*", "", text)
    # keep only citations that point at a real source
    text = CITATION.sub(lambda m: m.group(0) if 1 <= int(m.group(1)) <= max_sources else "", text)
    text = re.sub(r"(\[S\d+\])(\s*\1)+", r"\1", text)  # de-duplicate repeated tags
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + " ..."
    return text


class OutputGuard:
    def __init__(self) -> None:
        self.tok = get_sparse_embedder().tokenize

    def _event(self, rule: str, action: str) -> None:
        GUARDRAIL_EVENTS.labels(stage="output", rule=rule, action=action).inc()

    def groundedness(self, answer: str, context: str) -> tuple[float, list[str]]:
        ctx_tokens = set(self.tok(context))
        no_ctx = get_prompts().message("no_context").lower()[:40]
        scores, unsupported = [], []
        for sent in SENTENCES.split(CITATION.sub("", answer)):
            toks = self.tok(sent)
            if len(toks) < 4 or no_ctx in sent.lower():
                continue
            support = sum(t in ctx_tokens for t in toks) / len(toks)
            scores.append(support)
            if support < 0.5:
                unsupported.append(sent.strip()[:120])
        return (round(sum(scores) / len(scores), 3) if scores else 1.0), unsupported

    def check(self, raw_answer: str, context: str, n_sources: int) -> OutputCheck:
        prompts = get_prompts()
        answer = sanitize(raw_answer, n_sources)
        res = OutputCheck(answer=answer)

        if not answer:
            self._event("empty", "replace")
            res.answer, res.replaced = prompts.message("unsafe_output"), True
            res.flags.append("empty_answer")
            return res

        if LEAKAGE_MARKERS.search(answer):
            self._event("prompt_leakage", "replace")
            log.warning("output guardrail: prompt leakage detected")
            res.answer, res.replaced = prompts.message("unsafe_output"), True
            res.flags.append("prompt_leakage")
            return res

        if toxicity_hits(answer):
            self._event("toxicity", "replace")
            res.answer, res.replaced = prompts.message("unsafe_output"), True
            res.flags.append("toxicity")
            return res

        source_values = {v for _, v in find_pii(context)}
        redacted, kinds = mask_pii(answer, only=HIGH_RISK_PII)
        # contact details that literally appear in the policy are allowed; anything else is redacted
        redacted, other = mask_pii(redacted, keep_values=source_values)
        if kinds or other:
            self._event("pii", "redact")
            res.answer = redacted
            res.flags.append("pii_redacted:" + ",".join(sorted(set(kinds + other))))

        if prompts.message("no_context").lower()[:40] in res.answer.lower():
            return res  # model declined - nothing to ground

        score, unsupported = self.groundedness(res.answer, context)
        res.groundedness = score
        if score < settings.groundedness_min:
            self._event("groundedness", "replace")
            log.warning("output guardrail: low groundedness", extra={"score": score, "unsupported": unsupported})
            res.flags.append(f"ungrounded:{score}")
            res.answer, res.replaced = prompts.message("unsafe_output"), True
            return res
        if score < settings.groundedness_warn:
            self._event("groundedness", "warn")
            res.flags.append(f"low_grounding:{score}")
            res.warnings.append(prompts.message("low_grounding_warning"))

        ctx_numbers = set(NUMBER.findall(context))
        bad_numbers = [n for n in NUMBER.findall(CITATION.sub("", res.answer)) if n not in ctx_numbers]
        if bad_numbers:
            self._event("unverified_numbers", "warn")
            res.flags.append("unverified_numbers:" + ",".join(bad_numbers[:5]))
            if prompts.message("low_grounding_warning") not in res.warnings:
                res.warnings.append(prompts.message("low_grounding_warning"))

        res.cited = sorted({int(n) for n in CITATION.findall(res.answer)})
        if not res.cited:
            self._event("no_citation", "warn")
            res.flags.append("no_citations")
        return res
