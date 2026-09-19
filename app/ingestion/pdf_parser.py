"""Layout-aware PDF parsing with PyMuPDF.

* detects headings from font size / bold / numbering / ALL-CAPS -> section hierarchy
* extracts tables as Markdown (kept intact, not shredded into lines)
* removes running headers, footers and page numbers
* repairs hyphenation and normalises unicode
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

try:
    import pymupdf as fitz  # PyMuPDF >= 1.24
except ImportError:  # pragma: no cover
    import fitz

from app.core.exceptions import IngestionError
from app.core.logging import get_logger

log = get_logger(__name__)

NUMBERED_HEADING = re.compile(
    r"^((\d+(\.\d+)*)\.?|[IVXLC]+\.|[A-Z]\.|(section|article|chapter|part|clause|annexure|appendix)\s+[\w.\-]+)[:.)]?\s+\S",
    re.IGNORECASE)
PAGE_NUMBER = re.compile(r"^(page\s*)?\d+(\s*(of|/)\s*\d+)?$", re.IGNORECASE)
LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl", "­": "",
             "’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-",
             "•": "-", "●": "-", "▪": "-", "": "-"}


@dataclass
class Block:
    text: str
    page: int
    kind: str = "text"  # text | heading | table
    level: int = 0
    section_path: list[str] = field(default_factory=list)


@dataclass
class ParsedDocument:
    file_name: str
    page_count: int
    title: str
    blocks: list[Block]
    metadata: dict

    @property
    def char_count(self) -> int:
        return sum(len(b.text) for b in self.blocks)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    for src, dst in LIGATURES.items():
        text = text.replace(src, dst)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _signature(text: str) -> str:
    return re.sub(r"\d+", "#", text.lower()).strip()


def _heading_level(text: str, size: float, bold: bool, body_size: float, size_levels: dict[float, int]) -> int:
    """Return heading level (1..4) or 0 if the line is body text."""
    words = text.split()
    if not (2 < len(text) < 150) or len(words) > 15 or text.endswith((",", ";")):
        return 0
    ends_sentence = text.endswith(".") and len(words) > 3
    m = NUMBERED_HEADING.match(text)
    if m and len(words) <= 12 and not ends_sentence and (bold or size >= body_size + 0.5 or len(words) <= 8):
        num = m.group(2)
        return min(num.count(".") + 1, 4) if num else 1
    if size >= body_size + 1.5 and not ends_sentence:
        return size_levels.get(round(size), 1)
    if bold and len(words) <= 12 and not ends_sentence and size >= body_size - 0.5:
        return min(len(size_levels) + 1, 4)
    letters = [c for c in text if c.isalpha()]
    if len(letters) >= 4 and text.isupper() and len(words) <= 10 and not ends_sentence:
        return min(len(size_levels) + 1, 4)
    return 0


def parse_pdf(path: Path, max_pages: int = 500) -> ParsedDocument:
    try:
        doc = fitz.open(path)
    except Exception as exc:  # noqa: BLE001
        raise IngestionError("The file could not be opened as a PDF. Please check it is not corrupted.",
                             detail=str(exc)) from exc
    with doc:
        if doc.needs_pass:
            raise IngestionError("The PDF is password-protected. Please upload an unprotected copy.")
        if doc.page_count == 0:
            raise IngestionError("The PDF has no pages.")
        if doc.page_count > max_pages:
            raise IngestionError(f"The PDF has {doc.page_count} pages; the limit is {max_pages}.")

        raw_lines: list[dict] = []  # every text line with style info
        tables: list[Block] = []
        size_chars: Counter[float] = Counter()
        page_count = doc.page_count

        for pno, page in enumerate(doc, start=1):
            height = page.rect.height or 1
            table_rects = []
            try:  # table extraction (PyMuPDF >= 1.23)
                for t in page.find_tables().tables:
                    md = t.to_markdown().strip()
                    if md and md.count("\n") >= 1:
                        table_rects.append(fitz.Rect(t.bbox))
                        tables.append(Block(text=normalize_text_table(md), page=pno, kind="table"))
                        tables[-1].__dict__["y"] = t.bbox[1]
            except Exception:  # noqa: BLE001
                log.debug("table detection unavailable on page %s", pno)

            data = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)
            for b_idx, block in enumerate(data.get("blocks", [])):
                for line in block.get("lines", []):
                    spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
                    if not spans:
                        continue
                    bbox = fitz.Rect(line["bbox"])
                    if any(bbox.intersects(r) for r in table_rects):
                        continue
                    text = normalize_text("".join(s["text"] for s in spans))
                    if not text:
                        continue
                    size = max(s["size"] for s in spans)
                    bold = all((s.get("flags", 0) & 16) or "bold" in s.get("font", "").lower() for s in spans)
                    for s in spans:
                        size_chars[round(s["size"] * 2) / 2] += len(s["text"])
                    raw_lines.append({"text": text, "page": pno, "size": size, "bold": bold, "block": (pno, b_idx),
                                      "y": bbox.y0, "edge": bbox.y0 < height * 0.08 or bbox.y1 > height * 0.92})

    if not raw_lines and not tables:
        raise IngestionError("No readable text was found. The PDF looks scanned (image-only); "
                             "please upload a text-based PDF.")

    # --- running header / footer removal ---
    edge_sigs = Counter(_signature(l["text"]) for l in raw_lines if l["edge"])
    repeated = {sig for sig, n in edge_sigs.items() if page_count >= 3 and n >= max(2, page_count * 0.5)}
    lines = [l for l in raw_lines
             if not (l["edge"] and (_signature(l["text"]) in repeated or PAGE_NUMBER.match(l["text"])))]
    removed = len(raw_lines) - len(lines)

    body_size = size_chars.most_common(1)[0][0] if size_chars else 11.0
    big_sizes = sorted({round(l["size"]) for l in lines if l["size"] >= body_size + 1.5}, reverse=True)
    size_levels = {s: min(i + 1, 3) for i, s in enumerate(big_sizes)}

    # --- assemble paragraphs / headings, interleave tables by position ---
    items: list[tuple[int, float, Block]] = []
    buf: list[str] = []
    buf_meta: tuple | None = None

    def flush() -> None:
        nonlocal buf, buf_meta
        if buf and buf_meta:
            text = _join_lines(buf)
            items.append((buf_meta[0], buf_meta[1], Block(text=text, page=buf_meta[0])))
        buf, buf_meta = [], None

    for l in lines:
        level = _heading_level(l["text"], l["size"], l["bold"], body_size, size_levels)
        if level:
            flush()
            items.append((l["page"], l["y"], Block(text=l["text"].rstrip(":"), page=l["page"], kind="heading",
                                                   level=level)))
            continue
        if buf_meta and buf_meta[2] != l["block"]:
            flush()
        if not buf_meta:
            buf_meta = (l["page"], l["y"], l["block"])
        buf.append(l["text"])
    flush()
    for t in tables:
        items.append((t.page, t.__dict__.pop("y", 0.0), t))
    items.sort(key=lambda x: (x[0], x[1]))

    # --- section hierarchy ---
    stack: list[tuple[int, str]] = []
    blocks: list[Block] = []
    for _, _, blk in items:
        if blk.kind == "heading":
            while stack and stack[-1][0] >= blk.level:
                stack.pop()
            stack.append((blk.level, blk.text))
        blk.section_path = [t for _, t in stack]
        blocks.append(blk)

    headings = [b for b in blocks if b.kind == "heading"]
    title = headings[0].text if headings else Path(path).stem.replace("_", " ")
    log.info("pdf parsed", extra={"file": Path(path).name, "pages": page_count, "blocks": len(blocks),
                                  "headings": len(headings), "tables": len(tables), "header_footer_removed": removed,
                                  "body_font": body_size})
    return ParsedDocument(file_name=Path(path).name, page_count=page_count, title=title, blocks=blocks,
                          metadata={"headings": len(headings), "tables": len(tables), "body_font_size": body_size})


def normalize_text_table(md: str) -> str:
    return "\n".join(normalize_text(line) for line in md.splitlines() if line.strip())


def _join_lines(lines: list[str]) -> str:
    out = ""
    for line in lines:
        if not out:
            out = line
        elif out.endswith("-") and line[:1].islower():
            out = out[:-1] + line  # de-hyphenate
        elif line.startswith("- "):
            out += "\n" + line  # keep bullet structure
        else:
            out += " " + line
    return out.strip()
