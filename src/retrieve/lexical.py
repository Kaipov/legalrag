"""
Shared lexical normalization for BM25 indexing and search.

Keeps lexical retrieval cheap at query time while making it more aware of
legal structure and citation formats.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "of", "in", "to", "for",
    "with", "on", "at", "by", "from", "as", "into", "through", "during",
    "and", "or", "but", "not", "no", "nor", "if", "then", "than",
    "that", "this", "these", "those", "it", "its",
}

_CASE_CITATION_RE = re.compile(
    r"\b(?P<prefix>CFI|SCT|ENF|CA|ARB|TCD|DEC)\s*[- ]?\s*"
    r"(?P<number>\d{3})\s*(?:[-/]\s*)?(?P<year>\d{4})(?:\s*/\s*(?P<suffix>\d+))?\b",
    re.IGNORECASE,
)
_LAW_NUMBER_RE = re.compile(
    r"\b(?:(?:DIFC|DFSA)\s+)?(?P<kind>Law|Regulation|Rules?)\s+"
    r"(?:No\.?|Number)\s*(?P<number>\d+)\s+of\s+(?P<year>\d{4})\b",
    re.IGNORECASE,
)
_STRUCTURAL_REFERENCE_RE = re.compile(
    r"\b(?P<label>Article|Section|Regulation|Rule|Schedule|Appendix|Chapter|Part)\s+"
    r"(?P<number>\d+[A-Z]?)(?P<tail>(?:\([^)]+\))*)",
    re.IGNORECASE,
)
_NUMBERED_HEADING_RE = re.compile(r"^\s*(?P<number>\d+[A-Z]?)\.\s+(?P<title>.+)$")
_STATUTORY_TITLE_RE = re.compile(r"\b(LAW|REGULATION|RULES?)\b", re.IGNORECASE)

BM25_TITLE_WEIGHT = 3
BM25_SECTION_WEIGHT = 2


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").split())


def _iter_base_tokens(text: str) -> Iterable[str]:
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        if len(token) > 1 and token not in _STOPWORDS:
            yield token


def _expand_case_citation_aliases(text: str) -> list[str]:
    aliases: list[str] = []
    for match in _CASE_CITATION_RE.finditer(text):
        prefix = match.group("prefix").lower()
        number = match.group("number")
        year = match.group("year")
        suffix = match.group("suffix")

        aliases.append(f"{prefix}{number}")
        aliases.append(f"{prefix}{number}{year}")
        if suffix:
            aliases.append(f"{prefix}{number}{year}_{suffix}")
    return aliases


def _expand_law_number_aliases(text: str) -> list[str]:
    aliases: list[str] = []
    for match in _LAW_NUMBER_RE.finditer(text):
        kind = match.group("kind").lower().rstrip("s")
        number = match.group("number")
        year = match.group("year")
        aliases.extend(
            [
                f"{kind}{number}",
                f"{kind}{number}{year}",
                f"{kind}no{number}",
                f"{kind}no{number}{year}",
            ]
        )
    return aliases


def _expand_structural_reference_aliases(text: str) -> list[str]:
    aliases: list[str] = []
    for match in _STRUCTURAL_REFERENCE_RE.finditer(text):
        label = match.group("label").lower()
        number = match.group("number").lower()
        tail = match.group("tail") or ""
        aliases.append(f"{label}{number}")

        parts = [part.lower() for part in re.findall(r"\(([^)]+)\)", tail)]
        if not parts:
            continue

        current = f"{label}{number}"
        for part in parts:
            normalized = re.sub(r"[^a-z0-9]+", "", part)
            if not normalized:
                continue
            current = f"{current}_{normalized}"
            aliases.append(current)
    return aliases


def _expand_numbered_heading_aliases(text: str, *, doc_title: str = "") -> list[str]:
    if not _STATUTORY_TITLE_RE.search(doc_title or ""):
        return []

    aliases: list[str] = []
    for segment in re.split(r"\s*>\s*", text):
        match = _NUMBERED_HEADING_RE.match(segment)
        if not match:
            continue
        aliases.append(f"article{match.group('number').lower()}")
    return aliases


def tokenize_legal_text(text: str, *, doc_title: str = "") -> list[str]:
    """
    Tokenize legal text for lexical retrieval.

    In addition to plain lowercase tokens, emit normalized aliases for common
    legal citation formats so that variant spellings still match lexically.
    """
    text = _normalize_text(text)
    if not text:
        return []

    tokens = list(_iter_base_tokens(text))
    tokens.extend(_expand_case_citation_aliases(text))
    tokens.extend(_expand_law_number_aliases(text))
    tokens.extend(_expand_structural_reference_aliases(text))
    tokens.extend(_expand_numbered_heading_aliases(text, doc_title=doc_title))
    return tokens


def build_bm25_document_tokens(chunk: dict) -> list[str]:
    """
    Build a weighted lexical representation from structured chunk fields.

    We emulate a lightweight BM25F setup by repeating title and section tokens
    so that exact field matches survive RRF fusion more often.
    """
    doc_title = _normalize_text(chunk.get("doc_title") or "")
    section_path = _normalize_text(chunk.get("section_path") or "")
    text = _normalize_text(chunk.get("text") or "")

    title_tokens = tokenize_legal_text(doc_title, doc_title=doc_title)
    section_tokens = tokenize_legal_text(section_path, doc_title=doc_title)
    body_tokens = tokenize_legal_text(text, doc_title=doc_title)

    weighted_tokens: list[str] = []
    weighted_tokens.extend(title_tokens * BM25_TITLE_WEIGHT)
    weighted_tokens.extend(section_tokens * BM25_SECTION_WEIGHT)
    weighted_tokens.extend(body_tokens)
    return weighted_tokens or ["emptychunk"]


def summarize_token_counts(tokens: list[str]) -> Counter[str]:
    """Small helper for tests and debugging."""
    return Counter(tokens)
