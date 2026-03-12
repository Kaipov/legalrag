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
from src.retrieve.grounding_policy import GroundingIntent

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


def _page_focus_bonus(page_num: int, doc_last_page: int, intent: GroundingIntent | None) -> float:
    if intent is None or intent.page_focus == "any":
        return 0.0

    if intent.page_focus == "first":
        if page_num <= 1:
            return 4.0
        if page_num == 2:
            return 2.5
        if page_num == 3:
            return 1.0
        return 0.0

    if intent.page_focus == "last":
        distance_from_end = max(0, doc_last_page - page_num)
        if distance_from_end == 0:
            return 4.0
        if distance_from_end == 1:
            return 2.5
        if distance_from_end == 2:
            return 1.0
    return 0.0


def _intent_text_bonus(page_text: str, intent: GroundingIntent | None) -> float:
    if intent is None or intent.kind == "generic":
        return 0.0

    text = page_text.lower()
    bonus = 0.0
    phrase_hits = 0
    for phrase in intent.keyphrases:
        if phrase and phrase in text:
            phrase_hits += 1
    bonus += min(3, phrase_hits) * 1.25

    case_hits = 0
    upper_text = text.upper()
    for case_id in intent.case_ids:
        if case_id and case_id in upper_text:
            case_hits += 1
    bonus += min(2, case_hits) * 0.8
    return bonus


def _score_page(
    page_text: str,
    query_terms: set[str],
    answer_terms: set[str],
    chunk_terms: set[str],
    page_num: int,
    doc_last_page: int,
    intent: GroundingIntent | None,
) -> tuple[float, int, int, int]:
    page_terms = _tokenize(page_text)
    chunk_overlap = len(page_terms & chunk_terms)
    query_overlap = len(page_terms & query_terms)
    answer_overlap = len(page_terms & answer_terms)
    score = (2.0 * chunk_overlap) + (1.5 * query_overlap) + answer_overlap
    score += _page_focus_bonus(page_num, doc_last_page, intent)
    score += _intent_text_bonus(page_text, intent)
    return score, chunk_overlap, query_overlap, answer_overlap


def _intent_page_limit(candidate_pages: list[int], intent: GroundingIntent | None) -> int:
    default_limit = MAX_PAGES_PER_WIDE_CHUNK if len(candidate_pages) > 10 else MAX_PAGES_PER_CHUNK
    if intent is None or intent.max_pages_per_chunk is None:
        return default_limit
    return max(1, min(default_limit, intent.max_pages_per_chunk))


def _rank_pages_by_intent(
    candidate_pages: list[int],
    doc_last_page: int,
    intent: GroundingIntent | None,
) -> list[int]:
    if intent is None or intent.page_focus == "any":
        return list(candidate_pages)

    ranked = sorted(
        candidate_pages,
        key=lambda page_num: (-_page_focus_bonus(page_num, doc_last_page, intent), page_num),
    )
    return ranked


def _select_pages_for_chunk(
    chunk: dict[str, Any],
    query_terms: set[str],
    answer_terms: set[str],
    page_texts_by_doc: dict[str, dict[int, str]],
    intent: GroundingIntent | None = None,
) -> list[int]:
    candidate_pages = sorted(
        {
            int(page)
            for page in chunk.get("page_numbers", [])
            if isinstance(page, int) and page > 0
        }
    )
    if not candidate_pages:
        return []

    limit = _intent_page_limit(candidate_pages, intent)
    if len(candidate_pages) <= limit and (intent is None or intent.page_focus == "any"):
        return candidate_pages

    doc_id = str(chunk.get("doc_id") or "").strip()
    doc_pages = page_texts_by_doc.get(doc_id, {})
    doc_last_page = max(doc_pages.keys(), default=max(candidate_pages))
    if not doc_pages:
        ranked_pages = _rank_pages_by_intent(candidate_pages, doc_last_page, intent)
        return sorted(ranked_pages[:limit])

    chunk_terms = _tokenize(str(chunk.get("text") or ""))
    scored_pages: list[tuple[float, int, int, int, int]] = []
    for page_num in candidate_pages:
        score, chunk_overlap, query_overlap, answer_overlap = _score_page(
            doc_pages.get(page_num, ""),
            query_terms,
            answer_terms,
            chunk_terms,
            page_num,
            doc_last_page,
            intent,
        )
        scored_pages.append((score, chunk_overlap, query_overlap, answer_overlap, page_num))

    scored_pages.sort(key=lambda item: (-item[0], -item[1], -item[2], -item[3], item[4]))

    if all(item[0] == 0 and item[1] == 0 and item[2] == 0 and item[3] == 0 for item in scored_pages):
        ranked_pages = _rank_pages_by_intent(candidate_pages, doc_last_page, intent)
        return sorted(ranked_pages[:limit])

    return sorted(item[4] for item in scored_pages[:limit])


def collect_grounding_pages(
    reranked_chunks: list[tuple[dict, float]],
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    is_null: bool = False,
    question_text: str = "",
    answer_text: str = "",
    page_texts_by_doc: dict[str, dict[int, str]] | None = None,
    intent: GroundingIntent | None = None,
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
        intent: Detected grounding intent used for page-local selection priors.

    Returns:
        List of {"doc_id": str, "page_numbers": [int]} for submission.
        Merged by doc_id, pages sorted.
    """
    if is_null:
        return []

    pages_by_doc: dict[str, set[int]] = {}
    query_terms = _tokenize(question_text)
    answer_terms = _tokenize(answer_text)
    if page_texts_by_doc is None:
        page_texts_by_doc = _load_page_texts()

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
            intent=intent,
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