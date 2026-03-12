"""
Step 2: Structure-aware chunking of extracted pages.

Strategy:
  - Detect legal document structure via regex (Part, Chapter, Article, Section)
  - Chunk at Article/Section level (natural legal boundaries)
  - Short docs (amendments, court orders): single chunk per document
  - Track page numbers for each chunk (1-based PDF pages)

Output: index/chunks.jsonl - one line per chunk:
  {"chunk_id": "docid_001", "doc_id": "sha256", "page_numbers": [1,2],
   "section_path": "Part 3 > Chapter 2 > Article 15", "doc_title": "...", "text": "..."}
"""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from tqdm import tqdm

from src.config import PAGES_JSONL, CHUNKS_JSONL, MAX_CHUNK_TOKENS, MIN_CHUNK_TOKENS
from src.embeddings import get_embedding_client

logger = logging.getLogger(__name__)
_MAX_APPROX_TO_SKIP_EXACT_MULTIPLIER = 4

# --- Structure Detection Patterns ---

# Matches: PART 1, PART I, PART IV, Part One, etc.
RE_PART = re.compile(
    r"^(?:PART|Part)\s+(?:[IVXLC]+|\d+|One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten)",
    re.MULTILINE,
)

# Matches: CHAPTER 1, Chapter 2, etc.
RE_CHAPTER = re.compile(
    r"^(?:CHAPTER|Chapter)\s+(?:\d+|[IVXLC]+)",
    re.MULTILINE,
)

# Matches: Article 1, ARTICLE 15, etc. (primary chunk boundary)
RE_ARTICLE = re.compile(
    r"^(?:Article|ARTICLE)\s+(\d+[A-Z]?)",
    re.MULTILINE,
)

# Matches: Section 1, SECTION 15, etc. (alternative chunk boundary)
RE_SECTION = re.compile(
    r"^(?:Section|SECTION)\s+(\d+[A-Z]?)",
    re.MULTILINE,
)

# Matches: Regulation 1, REGULATION 15, Rule 1, etc.
RE_REGULATION = re.compile(
    r"^(?:Regulation|REGULATION|Rule|RULE)\s+(\d+[A-Z]?)",
    re.MULTILINE,
)

# Combined boundary patterns (ordered by specificity)
BOUNDARY_PATTERNS = [
    ("article", RE_ARTICLE),
    ("section", RE_SECTION),
    ("regulation", RE_REGULATION),
]


def _approx_tokens(text: str) -> int:
    """Fast heuristic used only to avoid unnecessary remote token-count calls."""
    return int(len(text.split()) * 1.3)


@lru_cache(maxsize=50_000)
def _count_tokens(text: str) -> int:
    """Exact token count using the configured embedding model tokenizer."""
    text = text.strip()
    if not text:
        return 0
    return get_embedding_client().count_tokens(text)


def _count_tokens_capped(text: str, max_tokens: int | None = None) -> int:
    """
    Use exact model tokenization near the budget boundary and fast-reject texts
    that are obviously too large.
    """
    text = text.strip()
    if not text:
        return 0

    approx_tokens = _approx_tokens(text)
    if max_tokens is not None and approx_tokens > max_tokens * _MAX_APPROX_TO_SKIP_EXACT_MULTIPLIER:
        return max_tokens + 1
    return _count_tokens(text)


def _fits_token_budget(text: str, max_tokens: int) -> bool:
    return _count_tokens_capped(text, max_tokens) <= max_tokens


def _detect_doc_title(pages_text: str) -> str:
    """Try to extract document title from the first page text."""
    lines = pages_text[:2000].split("\n")
    title_candidates = []
    for line in lines[:20]:
        line = line.strip()
        if not line:
            continue
        if len(line) < 5 or len(line) > 200:
            continue
        if any(kw in line for kw in ["Law", "Regulation", "Rules", "Order", "Act", "Amendment"]):
            title_candidates.append(line)
            break
        if line.isupper() and len(line) > 10:
            title_candidates.append(line)
            break
    return title_candidates[0] if title_candidates else ""


def _find_boundaries(text: str) -> list[tuple[int, str, str]]:
    """Find structural boundaries as (char_position, boundary_type, label)."""
    boundaries = []
    for btype, pattern in BOUNDARY_PATTERNS:
        for match in pattern.finditer(text):
            boundaries.append((match.start(), btype, match.group(0).strip()))
    boundaries.sort(key=lambda item: item[0])
    return boundaries


def _build_section_path(current_part: str, current_chapter: str, boundary_label: str) -> str:
    """Build hierarchical section path."""
    parts = []
    if current_part:
        parts.append(current_part)
    if current_chapter:
        parts.append(current_chapter)
    parts.append(boundary_label)
    return " > ".join(parts)


def _split_by_characters(text: str, max_tokens: int) -> list[str]:
    """Last-resort splitter for long strings without usable whitespace boundaries."""
    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        lo = start + 1
        hi = len(text)
        best_end = start

        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = text[start:mid].strip()
            if candidate and _fits_token_budget(candidate, max_tokens):
                best_end = mid
                lo = mid + 1
            else:
                hi = mid - 1

        if best_end <= start:
            raise ValueError("Could not split text into chunks within the token budget")

        chunk = text[start:best_end].strip()
        if chunk:
            chunks.append(chunk)
        start = best_end
        while start < len(text) and text[start].isspace():
            start += 1

    return chunks


def _split_units_to_fit(
    units: list[str],
    joiner: str,
    max_tokens: int,
    fallback_splitter,
) -> list[str]:
    """Pack the largest possible exact-token subspans using binary search."""
    units = [unit.strip() for unit in units if unit and unit.strip()]
    if not units:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(units):
        lo = start + 1
        hi = len(units)
        best_end = start

        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = joiner.join(units[start:mid]).strip()
            if candidate and _fits_token_budget(candidate, max_tokens):
                best_end = mid
                lo = mid + 1
            else:
                hi = mid - 1

        if best_end > start:
            chunks.append(joiner.join(units[start:best_end]).strip())
            start = best_end
            continue

        split_parts = fallback_splitter(units[start], max_tokens)
        if not split_parts:
            raise ValueError("Fallback splitter returned no text parts")
        chunks.extend(split_parts)
        start += 1

    return [chunk for chunk in chunks if chunk]


def _split_by_words(text: str, max_tokens: int) -> list[str]:
    """Split text by exact embedding-token budget using word boundaries."""
    words = text.split()
    if not words:
        return []
    return _split_units_to_fit(words, " ", max_tokens, _split_by_characters)


def _split_by_lines(text: str, max_tokens: int) -> list[str]:
    """Split text by lines before falling back to word windows."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) <= 1:
        return _split_by_words(text, max_tokens)
    return _split_units_to_fit(lines, "\n", max_tokens, _split_by_words)


def _split_by_paragraphs(text: str, max_tokens: int) -> list[str]:
    """Split a long text into paragraph-level chunks."""
    paragraphs = [para.strip() for para in re.split(r"\n\s*\n", text) if para.strip()]
    if len(paragraphs) <= 1:
        return _split_by_lines(text, max_tokens)
    return _split_units_to_fit(paragraphs, "\n\n", max_tokens, _split_by_lines)


def _split_text_to_fit(text: str, max_tokens: int) -> list[str]:
    """Guarantee that every returned chunk stays within the configured token budget."""
    text = text.strip()
    if not text:
        return []
    if _fits_token_budget(text, max_tokens):
        return [text]
    return _split_by_paragraphs(text, max_tokens)


def _assign_pages_to_split_parts(
    source_text: str,
    parts: list[str],
    page_offsets: list[tuple[int, int, int]],
    *,
    base_start: int = 0,
) -> list[list[int]]:
    """Map ordered split parts back to the pages they overlap."""
    assigned_pages: list[list[int]] = []
    search_start = 0

    for part in parts:
        if not part:
            assigned_pages.append([])
            continue

        match_start = source_text.find(part, search_start)
        if match_start == -1:
            probe = part[: min(120, len(part))]
            if probe:
                match_start = source_text.find(probe, search_start)

        if match_start == -1:
            assigned_pages.append([])
            continue

        match_end = match_start + len(part)
        assigned_pages.append(
            _get_pages_for_span(
                base_start + match_start,
                base_start + match_end,
                page_offsets,
            )
        )
        search_start = match_end

    return assigned_pages


def chunk_document(doc_id: str, pages: list[dict]) -> list[dict]:
    """
    Chunk a single document into structure-aware segments.

    Args:
        doc_id: Document identifier (SHA hash)
        pages: List of {page_num, text} dicts, sorted by page_num

    Returns:
        List of chunk dicts with chunk_id, doc_id, page_numbers, section_path, doc_title, text
    """
    if not pages:
        return []

    full_text = ""
    page_char_offsets = []
    for page in sorted(pages, key=lambda item: item["page_num"]):
        start = len(full_text)
        full_text += page["text"] + "\n\n"
        end = len(full_text)
        page_char_offsets.append((start, end, page["page_num"]))

    doc_title = _detect_doc_title(full_text)
    total_tokens = _count_tokens_capped(full_text, MAX_CHUNK_TOKENS)

    if total_tokens < MAX_CHUNK_TOKENS:
        all_page_nums = [page["page_num"] for page in pages]
        return [{
            "chunk_id": f"{doc_id}_000",
            "doc_id": doc_id,
            "page_numbers": all_page_nums,
            "section_path": doc_title or "Full document",
            "doc_title": doc_title,
            "text": full_text.strip(),
        }]

    boundaries = _find_boundaries(full_text)
    current_part = ""
    current_chapter = ""

    if not boundaries:
        return _chunk_by_pages(doc_id, pages, doc_title)

    chunks = []
    chunk_idx = 0

    if boundaries[0][0] > 100:
        pre_text = full_text[:boundaries[0][0]].strip()
        if _count_tokens_capped(pre_text, MIN_CHUNK_TOKENS) >= MIN_CHUNK_TOKENS:
            pre_pages = _get_pages_for_span(0, boundaries[0][0], page_char_offsets)
            pre_parts = _split_text_to_fit(pre_text, MAX_CHUNK_TOKENS)
            pre_part_pages = _assign_pages_to_split_parts(
                pre_text,
                pre_parts,
                page_char_offsets,
            )
            for offset, (sub_text, sub_pages) in enumerate(
                zip(pre_parts, pre_part_pages),
                start=1,
            ):
                section_path = doc_title or "Preamble"
                if len(pre_parts) > 1:
                    section_path = f"{section_path} (part {offset})"
                chunks.append({
                    "chunk_id": f"{doc_id}_{chunk_idx:03d}",
                    "doc_id": doc_id,
                    "page_numbers": sub_pages or pre_pages,
                    "section_path": section_path,
                    "doc_title": doc_title,
                    "text": sub_text,
                })
                chunk_idx += 1

    for i, (pos, _btype, label) in enumerate(boundaries):
        for match in RE_PART.finditer(full_text[:pos]):
            current_part = match.group(0).strip()
        for match in RE_CHAPTER.finditer(full_text[:pos]):
            current_chapter = match.group(0).strip()

        end_pos = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(full_text)
        segment_text = full_text[pos:end_pos].strip()
        if not segment_text:
            continue

        section_path = _build_section_path(current_part, current_chapter, label)
        segment_tokens = _count_tokens_capped(segment_text, MAX_CHUNK_TOKENS)
        segment_pages = _get_pages_for_span(pos, end_pos, page_char_offsets)

        if segment_tokens > MAX_CHUNK_TOKENS:
            sub_texts = _split_text_to_fit(segment_text, MAX_CHUNK_TOKENS)
            sub_pages_by_text = _assign_pages_to_split_parts(
                segment_text,
                sub_texts,
                page_char_offsets,
                base_start=pos,
            )
            for j, (sub_text, sub_pages) in enumerate(
                zip(sub_texts, sub_pages_by_text),
                start=1,
            ):
                part_section_path = section_path
                if len(sub_texts) > 1:
                    part_section_path = f"{section_path} (part {j})"
                chunks.append({
                    "chunk_id": f"{doc_id}_{chunk_idx:03d}",
                    "doc_id": doc_id,
                    "page_numbers": sub_pages or segment_pages,
                    "section_path": part_section_path,
                    "doc_title": doc_title,
                    "text": sub_text,
                })
                chunk_idx += 1
        elif segment_tokens < MIN_CHUNK_TOKENS:
            merged_text = None
            if chunks:
                candidate_text = chunks[-1]["text"] + "\n\n" + segment_text
                if _count_tokens_capped(candidate_text, MAX_CHUNK_TOKENS) <= MAX_CHUNK_TOKENS:
                    merged_text = candidate_text

            if merged_text is not None:
                prev = chunks[-1]
                prev["text"] = merged_text
                prev["page_numbers"] = sorted(set(prev["page_numbers"] + segment_pages))
                prev["section_path"] += f" + {label}"
            else:
                chunks.append({
                    "chunk_id": f"{doc_id}_{chunk_idx:03d}",
                    "doc_id": doc_id,
                    "page_numbers": segment_pages,
                    "section_path": section_path,
                    "doc_title": doc_title,
                    "text": segment_text,
                })
                chunk_idx += 1
        else:
            chunks.append({
                "chunk_id": f"{doc_id}_{chunk_idx:03d}",
                "doc_id": doc_id,
                "page_numbers": segment_pages,
                "section_path": section_path,
                "doc_title": doc_title,
                "text": segment_text,
            })
            chunk_idx += 1

    return chunks


def _chunk_by_pages(doc_id: str, pages: list[dict], doc_title: str) -> list[dict]:
    """Fallback: chunk by pages when no structural boundaries are found."""
    chunks = []
    current_parts: list[str] = []
    current_pages: list[int] = []
    current_tokens = 0
    chunk_idx = 0

    for page in sorted(pages, key=lambda item: item["page_num"]):
        page_num = page["page_num"]
        for segment in _split_text_to_fit(page["text"], MAX_CHUNK_TOKENS):
            segment_tokens = _count_tokens(segment)
            if current_parts and current_tokens + segment_tokens > MAX_CHUNK_TOKENS:
                chunks.append({
                    "chunk_id": f"{doc_id}_{chunk_idx:03d}",
                    "doc_id": doc_id,
                    "page_numbers": current_pages,
                    "section_path": doc_title or f"Pages {current_pages[0]}-{current_pages[-1]}",
                    "doc_title": doc_title,
                    "text": "\n\n".join(current_parts).strip(),
                })
                chunk_idx += 1
                current_parts = []
                current_pages = []
                current_tokens = 0

            current_parts.append(segment)
            current_tokens += segment_tokens
            if page_num not in current_pages:
                current_pages.append(page_num)

    if current_parts:
        chunks.append({
            "chunk_id": f"{doc_id}_{chunk_idx:03d}",
            "doc_id": doc_id,
            "page_numbers": current_pages,
            "section_path": doc_title or f"Pages {current_pages[0]}-{current_pages[-1]}",
            "doc_title": doc_title,
            "text": "\n\n".join(current_parts).strip(),
        })

    return chunks


def _get_pages_for_span(
    start_char: int, end_char: int, page_offsets: list[tuple[int, int, int]]
) -> list[int]:
    """Get page numbers that overlap with a character span."""
    pages = []
    for p_start, p_end, p_num in page_offsets:
        if p_start < end_char and p_end > start_char:
            pages.append(p_num)
    return sorted(set(pages)) if pages else [page_offsets[0][2]]


def _get_pages_for_text(
    sub_text: str, full_text: str, page_offsets: list[tuple[int, int, int]]
) -> list[int]:
    """Find which pages a sub-text comes from by searching in full text."""
    pos = full_text.find(sub_text[:100])
    if pos == -1:
        return []
    return _get_pages_for_span(pos, pos + len(sub_text), page_offsets)


def chunk_all_documents(
    pages_path: Path | str | None = None,
    output_path: Path | str | None = None,
) -> Path:
    """
    Chunk all documents from pages.jsonl into chunks.jsonl.
    Returns the output path.
    """
    pages_path = Path(pages_path) if pages_path else PAGES_JSONL
    output_path = Path(output_path) if output_path else CHUNKS_JSONL

    output_path.parent.mkdir(parents=True, exist_ok=True)

    docs: dict[str, list[dict]] = defaultdict(list)
    with open(pages_path, "r", encoding="utf-8") as handle:
        for line in handle:
            page = json.loads(line)
            docs[page["doc_id"]].append(page)

    logger.info("Chunking %s documents with MAX_CHUNK_TOKENS=%s...", len(docs), MAX_CHUNK_TOKENS)

    total_chunks = 0
    with open(output_path, "w", encoding="utf-8") as handle:
        for doc_id in tqdm(sorted(docs.keys()), desc="Chunking documents"):
            chunks = chunk_document(doc_id, docs[doc_id])
            for chunk in chunks:
                handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                total_chunks += 1

    logger.info("Created %s chunks from %s documents. Saved to %s", total_chunks, len(docs), output_path)
    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    chunk_all_documents()