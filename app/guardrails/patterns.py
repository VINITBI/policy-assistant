"""Custom, dependency-free guardrail rules (regex + word lists)."""
from __future__ import annotations

import re

PII_PATTERNS: dict[str, re.Pattern] = {
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "US_SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "AADHAAR": re.compile(r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b"),
    "PAN": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    "UK_NINO": re.compile(r"\b[A-CEGHJ-PR-TW-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b"),
    "PHONE": re.compile(r"(?<!\w)(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3,5}\)?[\s-]?)\d{3,4}[\s-]?\d{3,4}(?!\w)"),
    "IBAN": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"),
    "SECRET": re.compile(r"(?i)\b(password|passwd|pwd|api[_-]?key|secret|token)\s*[:=]\s*\S+"),
}
# Types we always redact in answers, even if present in the source text.
HIGH_RISK_PII = {"CREDIT_CARD", "US_SSN", "AADHAAR", "PAN", "UK_NINO", "IBAN", "SECRET"}

INJECTION_PATTERNS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"(?i)\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|all|earlier|system)\b"
                r".{0,20}\b(instructions?|rules?|prompts?|guidelines?|context)\b"), 1.0),
    (re.compile(r"(?i)\b(reveal|show|print|repeat|display|output|leak)\b.{0,30}\b(system|hidden|initial|original)\b"
                r".{0,15}\b(prompt|instructions?|message|rules?)\b"), 1.0),
    (re.compile(r"(?i)\byou are (now|no longer)\b|\bfrom now on,? you\b|\bpretend (to be|you are)\b"), 0.7),
    (re.compile(r"(?i)\b(jailbreak|DAN mode|developer mode|do anything now|god mode)\b"), 1.0),
    (re.compile(r"(?i)\bact as (an?|the)\b.{0,30}\b(unrestricted|unfiltered|evil|hacker|without (any )?rules)\b"), 1.0),
    (re.compile(r"(<\|im_start\|>|<\|system\|>|\[/?INST\]|<<SYS>>|###\s*(system|instruction))", re.IGNORECASE), 1.0),
    (re.compile(r"(?i)\b(bypass|disable|turn off)\b.{0,20}\b(guardrails?|filters?|safety|restrictions?)\b"), 1.0),
    (re.compile(r"(?i)\bnew instructions?\s*:"), 0.6),
]

TOXIC_TERMS = {
    "idiot", "stupid", "moron", "retard", "dumbass", "bastard", "bitch", "fuck", "fucking", "shit", "asshole",
    "cunt", "dick", "wanker", "slut", "whore", "kill yourself", "kys", "nigger", "faggot",
}
HARMFUL_INTENT = re.compile(
    r"(?i)\b(how (to|do i|can i) (hack|steal|poison|harm|hurt|kill)|make (a )?(bomb|weapon|explosive)|"
    r"commit (fraud|suicide)|launder money|evade (tax|taxes))\b")

LEAKAGE_MARKERS = re.compile(
    r"(?i)(my (system )?(prompt|instructions) (is|are|say)|you are \"?policy assistant\"?, the official|"
    r"rules - follow all of them|<\/?(context|history)>|^\s*(system|assistant|user)\s*:)", re.MULTILINE)


def find_pii(text: str) -> list[tuple[str, str]]:
    found = []
    for kind, pat in PII_PATTERNS.items():
        for m in pat.finditer(text):
            val = m.group(0)
            if kind == "PHONE" and sum(c.isdigit() for c in val) < 10:
                continue
            if kind == "CREDIT_CARD" and not _luhn(val):
                continue
            found.append((kind, val))
    return found


def mask_pii(text: str, only: set[str] | None = None, keep_values: set[str] | None = None) -> tuple[str, list[str]]:
    kinds: list[str] = []
    for kind, val in find_pii(text):
        if only and kind not in only:
            continue
        if keep_values and val in keep_values:
            continue
        text = text.replace(val, f"[{kind} REDACTED]")
        kinds.append(kind)
    return text, sorted(set(kinds))


def toxicity_hits(text: str) -> list[str]:
    low = f" {text.lower()} "
    return [t for t in TOXIC_TERMS if re.search(rf"(?<![a-z]){re.escape(t)}(?![a-z])", low)]


def _luhn(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()]
    if len(digits) < 13:
        return False
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        checksum += d
    return checksum % 10 == 0
