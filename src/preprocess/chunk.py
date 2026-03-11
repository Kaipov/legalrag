"""
Step 2: Structure-aware chunking of extracted pages.

Strategy:
  - Detect legal document structure via regex (Part, Chapter, Article, Section)
  - Chunk at Article/Section level (natural legal boundaries)
  - Short docs (amendments, court orders): single chunk per document
  - Track page numbers for each chunk (1-based PDF pages)

Output: index/chunks.jsonl — one line per chunk:
  {"chunk_id": "docid_001", "doc_id": "sha256", "page_numbers": [1,2],
   "section_path": "Part 3 > Chapter 2 > Article 15", "doc_title": "...", "text": "..."}
"""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterator

from tqdm import tqdm

from src.config import PAGES_JSONL, CHUNKS_JSONL, MAX_CHUNK_TOKENS, MIN_CHUNK_TOKENS

logger = logging.getLogger(__name__)

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
    """Rough token count estimation (words * 1.3)."""
    return int(len(text.split()) * 1.3)


def _detect_doc_title(pages_text: str) -> str:
    """Try to extract document title from the first page text."""
    lines = pages_text[:2000].split("\n")
    # Look for title-like lines (uppercase or prominent)
    title_candidates = []
    for line in lines[:20]:
        line = line.strip()
        if not line:
            continue
        # Skip very short or very long lines
        if len(line) < 5 or len(line) > 200:
            continue
        # Title heuristics: contains "Law", "Regulation", "Rules", "Order", "Act"
        if any(kw in line for kw in ["Law", "Regulation", "Rules", "Order", "Act", "Amendment"]):
            title_candidates.append(line)
            break
        # Uppercase line might be title
        if line.isupper() and len(line) > 10:
            title_candidates.append(line)
            break
    return title_candidates[0] if title_candidates else ""


def _find_boundaries(text: str) -> list[tuple[int, str, str]]:
    """
    Find structural boundaries in text.
    Returns list of (char_position, boundary_type, label).
    """
    boundaries = []
    for btype, pattern in BOUNDARY_PATTERNS:
        for m in pattern.finditer(text):
            label = m.group(0).strip()
            boundaries.append((m.start(), btype, label))

    # Sort by position
    boundaries.sort(key=lambda x: x[0])
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


def _split_by_paragraphs(text: str, max_tokens: int) -> list[str]:
    """Split a long text into paragraph-level chunks."""
    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []
    current = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = _approx_tokens(para)
        if current_tokens + para_tokens > max_tokens and current:
            chunks.append("\n\n".join(current))
            current = [para]
            current_tokens = para_tokens
        else:
            current.append(para)
            current_tokens += para_tokens

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def chunk_document(
    doc_id: str,
    pages: list[dict],
) -> list[dict]:
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

    # Combine all pages into one text, tracking page boundaries
    full_text = ""
    page_char_offsets = []  # (start_char, end_char, page_num)

    for page in sorted(pages, key=lambda p: p["page_num"]):
        start = len(full_text)
        full_text += page["text"] + "\n\n"
        end = len(full_text)
        page_char_offsets.append((start, end, page["page_num"]))

    doc_title = _detect_doc_title(full_text)
    total_tokens = _approx_tokens(full_text)

    # For short documents (amendments, court orders, etc.): single chunk
    if total_tokens < MAX_CHUNK_TOKENS or len(pages) <= 2:
        all_page_nums = [p["page_num"] for p in pages]
        return [{
            "chunk_id": f"{doc_id}_000",
            "doc_id": doc_id,
            "page_numbers": all_page_nums,
            "section_path": doc_title or "Full document",
            "doc_title": doc_title,
            "text": full_text.strip(),
        }]

    # Find structural boundaries
    boundaries = _find_boundaries(full_text)

    # Track Part and Chapter context
    current_part = ""
    current_chapter = ""

    for m in RE_PART.finditer(full_text):
        pass  # We'll update below
    for m in RE_CHAPTER.finditer(full_text):
        pass  # We'll update below

    # If no boundaries found, fall back to page-level chunking
    if not boundaries:
        return _chunk_by_pages(doc_id, pages, doc_title)

    # Split text at boundaries
    chunks = []
    chunk_idx = 0

    # Add implicit first chunk (text before first boundary) if substantial
    if boundaries[0][0] > 100:
        pre_text = full_text[:boundaries[0][0]].strip()
        if _approx_tokens(pre_text) >= MIN_CHUNK_TOKENS:
            pre_pages = _get_pages_for_span(0, boundaries[0][0], page_char_offsets)
            chunks.append({
                "chunk_id": f"{doc_id}_{chunk_idx:03d}",
                "doc_id": doc_id,
                "page_numbers": pre_pages,
                "section_path": doc_title or "Preamble",
                "doc_title": doc_title,
                "text": pre_text,
            })
            chunk_idx += 1

    # Process each boundary segment
    for i, (pos, btype, label) in enumerate(boundaries):
        # Update Part/Chapter context by scanning what came before
        for m in RE_PART.finditer(full_text[:pos]):
            current_part = m.group(0).strip()
        for m in RE_CHAPTER.finditer(full_text[:pos]):
            current_chapter = m.group(0).strip()

        # Get text from this boundary to the next
        end_pos = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(full_text)
        segment_text = full_text[pos:end_pos].strip()

        if not segment_text:
            continue

        section_path = _build_section_path(current_part, current_chapter, label)
        segment_tokens = _approx_tokens(segment_text)
        segment_pages = _get_pages_for_span(pos, end_pos, page_char_offsets)

        # If segment is too long, split by paragraphs
        if segment_tokens > MAX_CHUNK_TOKENS:
            sub_texts = _split_by_paragraphs(segment_text, MAX_CHUNK_TOKENS)
            for j, sub_text in enumerate(sub_texts):
                sub_pages = _get_pages_for_text(sub_text, full_text, page_char_offsets)
                chunks.append({
                    "chunk_id": f"{doc_id}_{chunk_idx:03d}",
                    "doc_id": doc_id,
                    "page_numbers": sub_pages or segment_pages,
                    "section_path": f"{section_path} (part {j+1})" if len(sub_texts) > 1 else section_path,
                    "doc_title": doc_title,
                    "text": sub_text.strip(),
                })
                chunk_idx += 1
        elif segment_tokens < MIN_CHUNK_TOKENS:
            # Too short — will be merged with next or appended to previous
            if chunks:
                # Merge with previous chunk
                prev = chunks[-1]
                prev["text"] += "\n\n" + segment_text
                prev["page_numbers"] = sorted(set(prev["page_numbers"] + segment_pages))
                prev["section_path"] += f" + {label}"
            else:
                # Nothing to merge with, create anyway
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
    """Fallback: chunk by pages when no structural boundaries found."""
    chunks = []
    current_text = ""
    current_pages = []
    chunk_idx = 0

    for page in sorted(pages, key=lambda p: p["page_num"]):
        if _approx_tokens(current_text + page["text"]) > MAX_CHUNK_TOKENS and current_text:
            chunks.append({
                "chunk_id": f"{doc_id}_{chunk_idx:03d}",
                "doc_id": doc_id,
                "page_numbers": current_pages,
                "section_path": doc_title or f"Pages {current_pages[0]}-{current_pages[-1]}",
                "doc_title": doc_title,
                "text": current_text.strip(),
            })
            chunk_idx += 1
            current_text = page["text"] + "\n\n"
            current_pages = [page["page_num"]]
        else:
            current_text += page["text"] + "\n\n"
            current_pages.append(page["page_num"])

    if current_text.strip():
        chunks.append({
            "chunk_id": f"{doc_id}_{chunk_idx:03d}",
            "doc_id": doc_id,
            "page_numbers": current_pages,
            "section_path": doc_title or f"Pages {current_pages[0]}-{current_pages[-1]}",
            "doc_title": doc_title,
            "text": current_text.strip(),
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
    # Find the position of sub_text in full_text
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

    # Load all pages, grouped by doc_id
    docs: dict[str, list[dict]] = defaultdict(list)
    with open(pages_path, "r", encoding="utf-8") as f:
        for line in f:
            page = json.loads(line)
            docs[page["doc_id"]].append(page)

    logger.info(f"Chunking {len(docs)} documents...")

    total_chunks = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for doc_id in tqdm(sorted(docs.keys()), desc="Chunking documents"):
            pages = docs[doc_id]
            chunks = chunk_document(doc_id, pages)
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                total_chunks += 1

    logger.info(f"Created {total_chunks} chunks from {len(docs)} documents. Saved to {output_path}")
    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    chunk_all_documents()
