"""
Grounding: collect and filter page references from retrieved chunks.

Grounding uses F-beta (beta=2.5): recall ~6x more important than precision.
Strategy: include pages from all chunks above a relevance threshold.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Default reranker score threshold for including a chunk's pages
DEFAULT_SCORE_THRESHOLD = -1.0  # bge-reranker outputs logits, can be negative


def collect_grounding_pages(
    reranked_chunks: list[tuple[dict, float]],
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    is_null: bool = False,
) -> list[dict[str, Any]]:
    """
    Collect page references for grounding from reranked chunks.

    Args:
        reranked_chunks: List of (chunk_dict, reranker_score) from retriever
        score_threshold: Minimum reranker score to include a chunk's pages
        is_null: If True, return empty list (null answer = no sources)

    Returns:
        List of {"doc_id": str, "page_numbers": [int]} for submission.
        Merged by doc_id, pages sorted.
    """
    if is_null:
        return []

    # Collect pages from chunks above threshold
    pages_by_doc: dict[str, set[int]] = {}

    for chunk, score in reranked_chunks:
        if score < score_threshold:
            continue

        doc_id = chunk["doc_id"]
        page_nums = chunk.get("page_numbers", [])

        if doc_id not in pages_by_doc:
            pages_by_doc[doc_id] = set()
        pages_by_doc[doc_id].update(page_nums)

    # Build result in submission format
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
