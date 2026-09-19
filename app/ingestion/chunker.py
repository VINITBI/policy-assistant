"""Structure-aware, hierarchical (parent/child) chunking.

1. The document is split into *sections* along detected headings (never crossing a section boundary).
2. Each section is split into *parent* chunks (~1200 tokens) on paragraph/sentence boundaries.
3. Each parent is split into *child* chunks (~350 tokens, ~60 token overlap) - these are embedded.
4. Every child is prefixed with a contextual header (policy + section path) before embedding.

At query time the small child gives precise matching; its parent gives the LLM enough
surrounding context ("small-to-big" retrieval). Tables are kept whole where possible and,
if split, the header row is repeated in every piece.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Callable

from app.ingestion.pdf_parser import Block, ParsedDocument

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
TokenCounter = Callable[[str], int]


@dataclass
class Unit:
    text: str
    page: int
    tokens: int
    kind: str = "text"


@dataclass
class Chunk:
    chunk_id: str
    chunk_index: int
    text: str
    embed_text: str
    parent_id: str
    parent_text: str
    section_title: str
    section_path: list[str]
    page_start: int
    page_end: int
    token_count: int
    content_type: str = "text"
    extra: dict = field(default_factory=dict)


def approx_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))


class HierarchicalChunker:
    def __init__(self, count_tokens: TokenCounter | None = None, child_tokens: int = 350, overlap_tokens: int = 60,
                 parent_tokens: int = 1200, min_tokens: int = 40) -> None:
        self.count = count_tokens or approx_tokens
        self.child_tokens = child_tokens
        self.overlap_tokens = overlap_tokens
        self.parent_tokens = max(parent_tokens, child_tokens)
        self.min_tokens = min_tokens

    # ---------- public ----------
    def chunk(self, doc: ParsedDocument, policy_id: str, policy_name: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        for path, units in self._sections(doc.blocks):
            for parent_units in self._pack(units, self.parent_tokens, overlap=0):
                parent_text = "\n".join(u.text for u in parent_units)
                parent_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{policy_id}:parent:{len(chunks)}:{parent_text[:64]}"))
                for child_units in self._pack(parent_units, self.child_tokens, overlap=self.overlap_tokens):
                    text = "\n".join(u.text for u in child_units).strip()
                    if not text:
                        continue
                    header = f"Policy: {policy_name}\nSection: {' > '.join(path) or 'General'}\n\n"
                    idx = len(chunks)
                    chunks.append(Chunk(
                        chunk_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{policy_id}:child:{idx}")),
                        chunk_index=idx, text=text, embed_text=header + text, parent_id=parent_id,
                        parent_text=parent_text, section_title=path[-1] if path else "General",
                        section_path=path, page_start=min(u.page for u in child_units),
                        page_end=max(u.page for u in child_units), token_count=self.count(text),
                        content_type="table" if all(u.kind == "table" for u in child_units) else "text"))
        return chunks

    # ---------- internals ----------
    def _sections(self, blocks: list[Block]) -> list[tuple[list[str], list[Unit]]]:
        """Group body blocks by section path; tiny sections are carried into the next one."""
        sections: list[tuple[list[str], list[Unit]]] = []
        current_path: list[str] | None = None
        current: list[Unit] = []
        carry: list[Unit] = []

        def close() -> None:
            nonlocal current, carry
            if current_path is None and not current:
                return
            units = carry + current
            carry = []
            if not units:
                return
            if sum(u.tokens for u in units) < self.min_tokens:
                title = " > ".join(current_path or [])
                carry = ([Unit(title, units[0].page, self.count(title))] if title else []) + units
                return
            sections.append((list(current_path or []), units))

        for b in blocks:
            if b.kind == "heading":
                close()
                current_path, current = b.section_path, []
                continue
            if current_path is None:
                current_path = b.section_path
            current.extend(self._units(b))
        close()
        if carry:  # trailing tiny content
            if sections:
                sections[-1][1].extend(carry)
            else:
                sections.append((list(current_path or []), carry))
        return sections

    def _units(self, block: Block) -> list[Unit]:
        """Split a block into units no larger than the child size."""
        tokens = self.count(block.text)
        if tokens <= self.child_tokens:
            return [Unit(block.text, block.page, tokens, block.kind)]
        if block.kind == "table":
            return self._split_table(block)
        units: list[Unit] = []
        for sent in SENTENCE_SPLIT.split(block.text):
            st = self.count(sent)
            if st <= self.child_tokens:
                units.append(Unit(sent, block.page, st))
            else:  # pathological long sentence -> word windows
                words = sent.split()
                step = max(1, int(self.child_tokens / 1.3))
                for i in range(0, len(words), step):
                    piece = " ".join(words[i:i + step])
                    units.append(Unit(piece, block.page, self.count(piece)))
        return units

    def _split_table(self, block: Block) -> list[Unit]:
        rows = block.text.splitlines()
        header, body = rows[:2], rows[2:]
        units, cur = [], list(header)
        for row in body:
            if self.count("\n".join(cur + [row])) > self.child_tokens and len(cur) > len(header):
                units.append(Unit("\n".join(cur), block.page, self.count("\n".join(cur)), "table"))
                cur = list(header)
            cur.append(row)
        if len(cur) > len(header):
            units.append(Unit("\n".join(cur), block.page, self.count("\n".join(cur)), "table"))
        return units

    def _pack(self, units: list[Unit], limit: int, overlap: int) -> list[list[Unit]]:
        """Greedy packing with unit-level (sentence/paragraph) overlap."""
        groups: list[list[Unit]] = []
        cur: list[Unit] = []
        cur_tokens = 0
        for u in units:
            if cur and cur_tokens + u.tokens > limit:
                groups.append(cur)
                tail: list[Unit] = []
                t = 0
                for prev in reversed(cur):
                    if t + prev.tokens > overlap or prev.kind == "table":
                        break
                    tail.insert(0, prev)
                    t += prev.tokens
                cur, cur_tokens = tail, t
            cur.append(u)
            cur_tokens += u.tokens
        if cur:
            new_tokens = sum(u.tokens for u in cur)
            if groups and new_tokens < self.min_tokens and sum(x.tokens for x in groups[-1]) + new_tokens <= limit * 1.2:
                groups[-1].extend(x for x in cur if x not in groups[-1])
            else:
                groups.append(cur)
        return groups
