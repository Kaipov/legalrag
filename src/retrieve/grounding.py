"""
Grounding: collect and filter page references from retrieved chunks.

Grounding uses F-beta (beta=2.5): recall ~6x more important than precision.
Strategy: start from the chunks used for generation, then prune each chunk
back down to the pages most likely to contain the supporting evidence.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from src.config import PAGES_JSONL

logger = logging.getLogger(__name__)

# Default reranker score threshold for including a chunk's pages
DEFAULT_SCORE_THRESHOLD = -1.0  # bge-reranker outputs logits, can be negative
MAX_PAGES_PER_CHUNK = 2
MAX_PAGES_PER_WIDE_CHUNK = 3
_PAGE_TEXTS: dict[str, dict[int, str]] | None = None
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "of", "in", "to", "for",
    "with", "on", "at", "by", "from", "as", "into", "through", "during",
    "and", "or", "but", "not", "no", "nor", "if", "then", "than",
    "that", "this", "these", "those", "it", "its",
}


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 1 and token not in _STOPWORDS
    }


def _load_page_texts() -> dict[str, dict[int, str]]:
    global _PAGE_TEXTS
    if _PAGE_TEXTS is not None:
        return _PAGE_TEXTS

    page_texts: dict[str, dict[int, str]] = {}
    path = Path(PAGES_JSONL)
    if not path.exists():
        logger.warning("Grounding page index %s not found; falling back to chunk spans", path)
        _PAGE_TEXTS = page_texts
        return _PAGE_TEXTS

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            page = json.loads(line)
            doc_id = str(page.get("doc_id") or "").strip()
            page_num = int(page.get("page_num") or 0)
            if not doc_id or page_num <= 0:
                continue
            page_texts.setdefault(doc_id, {})[page_num] = str(page.get("text") or "")

    _PAGE_TEXTS = page_texts
    return _PAGE_TEXTS


def _score_page(
    page_text: str,
    query_terms: set[str],
    answer_terms: set[str],
    chunk_terms: set[str],
) -> tuple[float, int, int, int]:
    page_terms = _tokenize(page_text)
    chunk_overlap = len(page_terms & chunk_terms)
    query_overlap = len(page_terms & query_terms)
    answer_overlap = len(page_terms & answer_terms)
    score = (2.0 * chunk_overlap) + (1.5 * query_overlap) + answer_overlap
    return score, chunk_overlap, query_overlap, answer_overlap


def _select_pages_for_chunk(
    chunk: dict[str, Any],
    query_terms: set[str],
    answer_terms: set[str],
    page_texts_by_doc: dict[str, dict[int, str]],
) -> list[int]:
    candidate_pages = sorted(
        {
            int(page)
            for page in chunk.get("page_numbers", [])
            if isinstance(page, int) and page > 0
        }
    )
    if len(candidate_pages) <= 3:
        return candidate_pages

    doc_id = str(chunk.get("doc_id") or "").strip()
    doc_pages = page_texts_by_doc.get(doc_id, {})
    if not doc_pages:
        limit = MAX_PAGES_PER_WIDE_CHUNK if len(candidate_pages) > 10 else MAX_PAGES_PER_CHUNK
        return candidate_pages[:limit]

    chunk_terms = _tokenize(str(chunk.get("text") or ""))
    scored_pages: list[tuple[float, int, int, int, int]] = []
    for page_num in candidate_pages:
        score, chunk_overlap, query_overlap, answer_overlap = _score_page(
            doc_pages.get(page_num, ""),
            query_terms,
            answer_terms,
            chunk_terms,
        )
        scored_pages.append((score, chunk_overlap, query_overlap, answer_overlap, page_num))

    scored_pages.sort(key=lambda item: (-item[0], -item[1], -item[2], -item[3], item[4]))
    limit = MAX_PAGES_PER_WIDE_CHUNK if len(candidate_pages) > 10 else MAX_PAGES_PER_CHUNK

    if all(item[0] == 0 and item[1] == 0 and item[2] == 0 and item[3] == 0 for item in scored_pages):
        return candidate_pages[:limit]

    return sorted(item[4] for item in scored_pages[:limit])


def collect_grounding_pages(
    reranked_chunks: list[tuple[dict, float]],
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    is_null: bool = False,
    question_text: str = "",
    answer_text: str = "",
    page_texts_by_doc: dict[str, dict[int, str]] | None = None,
) -> list[dict[str, Any]]:
    """
    Collect page references for grounding from the chunks used for generation.

    Args:
        reranked_chunks: List of (chunk_dict, score) tuples.
        score_threshold: Minimum score to include a chunk's pages.
        is_null: If True, return empty list (null answer = no sources).
        question_text: Original user question for page-level lexical matching.
        answer_text: Model answer text for a light grounding bias.
        page_texts_by_doc: Optional injected page-text map for tests.

    Returns:
        List of {"doc_id": str, "page_numbers": [int]} for submission.
        Merged by doc_id, pages sorted.
    """
    if is_null:
        return []

    pages_by_doc: dict[str, set[int]] = {}
    query_terms = _tokenize(question_text)
    answer_terms = _tokenize(answer_text)
    page_texts_by_doc = page_texts_by_doc or _load_page_texts()

    for chunk, score in reranked_chunks:
        if score < score_threshold:
            continue

        doc_id = str(chunk.get("doc_id") or "").strip()
        if not doc_id:
            continue

        selected_pages = _select_pages_for_chunk(
            chunk,
            query_terms,
            answer_terms,
            page_texts_by_doc,
        )
        if not selected_pages:
            continue

        pages_by_doc.setdefault(doc_id, set()).update(selected_pages)

    result = []
    for doc_id in sorted(pages_by_doc.keys()):
        pages = sorted(pages_by_doc[doc_id])
        if pages:
            result.append({
                "doc_id": doc_id,
                "page_numbers": pages,
            })

    return result


def compute_fbeta(
    predicted_pages: set[tuple[str, int]],
    golden_pages: set[tuple[str, int]],
    beta: float = 2.5,
) -> float:
    """
    Compute F-beta score for grounding evaluation.

    Args:
        predicted_pages: Set of (doc_id, page_num) tuples we predicted
        golden_pages: Set of (doc_id, page_num) tuples that are correct
        beta: Beta parameter (2.5 means recall ~6x more important)

    Returns:
        F-beta score (0.0 to 1.0)
    """
    if not predicted_pages and not golden_pages:
        return 1.0
    if not predicted_pages or not golden_pages:
        return 0.0

    intersection = predicted_pages & golden_pages
    precision = len(intersection) / len(predicted_pages) if predicted_pages else 0
    recall = len(intersection) / len(golden_pages) if golden_pages else 0

    if precision + recall == 0:
        return 0.0

    beta_sq = beta ** 2
    f_beta = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
    return f_beta