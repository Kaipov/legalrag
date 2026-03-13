"""
Grounding: collect and filter page references from retrieved chunks.

Grounding uses F-beta (beta=2.5): recall ~6x more important than precision.
Strategy: start from the chunks used for generation, then prune each chunk
back down to the pages most likely to contain the supporting evidence.
If page-level indices are available, run a second page retrieval pass limited
to the documents already present in generation chunks.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.config import PAGES_JSONL
from src.retrieve.grounding_policy import GroundingIntent
from src.retrieve.lexical import tokenize_legal_text

logger = logging.getLogger(__name__)

# Default reranker score threshold for including a chunk's pages
DEFAULT_SCORE_THRESHOLD = -1.0  # bge-reranker outputs logits, can be negative
MAX_PAGES_PER_CHUNK = 2
MAX_PAGES_PER_WIDE_CHUNK = 3
_PAGE_TEXTS: dict[str, dict[int, str]] | None = None
_PAGE_RETRIEVER: Any = None



def _tokenize(text: str) -> set[str]:
    return set(tokenize_legal_text(text))



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



def _get_page_retriever():
    global _PAGE_RETRIEVER
    if _PAGE_RETRIEVER is False:
        return None
    if _PAGE_RETRIEVER is not None:
        return _PAGE_RETRIEVER

    try:
        from src.retrieve.page_search import PageRetriever
    except Exception as exc:  # pragma: no cover - defensive import fallback
        logger.warning("Could not import page retriever; skipping page grounding rescue: %s", exc)
        _PAGE_RETRIEVER = False
        return None

    try:
        _PAGE_RETRIEVER = PageRetriever()
    except FileNotFoundError:
        logger.info("Page indices not found; grounding will use chunk-local page selection only")
        _PAGE_RETRIEVER = False
        return None
    return _PAGE_RETRIEVER



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



def _max_pages_per_doc_limit(intent: GroundingIntent | None) -> int:
    if intent is None or intent.max_pages_per_doc is None:
        return 2
    return max(1, int(intent.max_pages_per_doc))



def _is_structured_answer_type(answer_type: str) -> bool:
    return str(answer_type or "").lower() in {"number", "date", "boolean", "name", "names"}



def _max_docs_limit(intent: GroundingIntent | None, answer_type: str) -> int:
    if intent is None or intent.kind == "generic":
        return 2 if _is_structured_answer_type(answer_type) else 3
    if intent.is_compare:
        if intent.case_ids:
            return max(2, min(3, len(intent.case_ids)))
        return 2
    if intent.kind == "last_page":
        return max(1, min(2, len(intent.case_ids) or 1))
    if intent.is_page_local:
        return max(1, min(2, len(intent.case_ids) or 1))
    return 2



def _max_total_pages_limit(intent: GroundingIntent | None, answer_type: str, doc_count: int) -> int:
    if intent is None or intent.kind == "generic":
        return 2 if _is_structured_answer_type(answer_type) else 3
    if intent.is_compare:
        return max(2, min(3, doc_count))
    if intent.kind == "last_page":
        return 2
    if intent.is_page_local:
        return max(1, doc_count)
    return max(1, min(2, doc_count))



def _score_selected_pages_for_doc(
    doc_id: str,
    candidate_pages: list[int],
    chunk_terms: set[str],
    query_terms: set[str],
    answer_terms: set[str],
    page_texts_by_doc: dict[str, dict[int, str]],
    intent: GroundingIntent | None,
) -> list[tuple[float, int, int, int, int]]:
    doc_pages = page_texts_by_doc.get(doc_id, {})
    doc_last_page = max(doc_pages.keys(), default=max(candidate_pages))
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
    return scored_pages



def _select_top_pages_for_doc(
    doc_id: str,
    candidate_pages: list[int],
    chunk_terms: set[str],
    query_terms: set[str],
    answer_terms: set[str],
    page_texts_by_doc: dict[str, dict[int, str]],
    intent: GroundingIntent | None,
) -> list[int]:
    limit = _max_pages_per_doc_limit(intent)
    if len(candidate_pages) <= limit:
        return sorted(candidate_pages)

    scored_pages = _score_selected_pages_for_doc(
        doc_id,
        candidate_pages,
        chunk_terms,
        query_terms,
        answer_terms,
        page_texts_by_doc,
        intent,
    )
    if all(item[0] == 0 and item[1] == 0 and item[2] == 0 and item[3] == 0 for item in scored_pages):
        doc_pages = page_texts_by_doc.get(doc_id, {})
        doc_last_page = max(doc_pages.keys(), default=max(candidate_pages))
        ranked_pages = _rank_pages_by_intent(candidate_pages, doc_last_page, intent)
        return sorted(ranked_pages[:limit])

    return sorted(item[4] for item in scored_pages[:limit])



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
    scored_pages = _score_selected_pages_for_doc(
        doc_id,
        candidate_pages,
        chunk_terms,
        query_terms,
        answer_terms,
        page_texts_by_doc,
        intent,
    )

    if all(item[0] == 0 and item[1] == 0 and item[2] == 0 and item[3] == 0 for item in scored_pages):
        ranked_pages = _rank_pages_by_intent(candidate_pages, doc_last_page, intent)
        return sorted(ranked_pages[:limit])

    return sorted(item[4] for item in scored_pages[:limit])



def _normalize_allowed_doc_ids(
    allowed_doc_ids: set[str] | None,
    reranked_chunks: list[tuple[dict, float]],
) -> set[str]:
    if allowed_doc_ids is not None:
        return {str(doc_id).strip() for doc_id in allowed_doc_ids if str(doc_id).strip()}

    normalized: set[str] = set()
    for chunk, _score in reranked_chunks:
        doc_id = str(chunk.get("doc_id") or "").strip()
        if doc_id:
            normalized.add(doc_id)
    return normalized



def _is_null_like_answer(answer_text: str) -> bool:
    normalized = " ".join(str(answer_text or "").strip().lower().split())
    return normalized in {
        "",
        "null",
        "this question cannot be answered based on the available difc documents.",
    }



def _build_page_retrieval_query(
    question_text: str,
    answer_text: str,
    intent: GroundingIntent | None,
) -> str:
    parts: list[str] = []
    question_text = str(question_text or "").strip()
    if question_text:
        parts.append(question_text)

    answer_text = str(answer_text or "").strip()
    if answer_text and not _is_null_like_answer(answer_text):
        parts.append(answer_text)

    if intent is not None:
        for phrase in intent.keyphrases[:2]:
            phrase = str(phrase or "").strip()
            if phrase:
                parts.append(phrase)

    deduped_parts: list[str] = []
    seen_parts: set[str] = set()
    for part in parts:
        lowered = part.lower()
        if lowered in seen_parts:
            continue
        seen_parts.add(lowered)
        deduped_parts.append(part)
    return "\n".join(deduped_parts)



def _retrieve_additional_pages(
    question_text: str,
    answer_text: str,
    allowed_doc_ids: set[str],
    intent: GroundingIntent | None,
) -> list[tuple[dict[str, Any], float]]:
    if not allowed_doc_ids:
        return []

    page_retriever = _get_page_retriever()
    if page_retriever is None:
        return []

    retrieval_query = _build_page_retrieval_query(question_text, answer_text, intent)
    if not retrieval_query:
        return []

    try:
        return page_retriever.search(
            retrieval_query,
            allowed_doc_ids=allowed_doc_ids,
        )
    except Exception as exc:  # pragma: no cover - defensive runtime fallback
        logger.warning("Page retrieval grounding rescue failed: %s", exc)
        return []



def _finalize_grounding_results(
    pages_by_doc: dict[str, set[int]],
    doc_chunk_terms_by_doc: dict[str, set[str]],
    query_terms: set[str],
    answer_terms: set[str],
    page_texts_by_doc: dict[str, dict[int, str]],
    intent: GroundingIntent | None,
    answer_type: str,
) -> list[dict[str, Any]]:
    doc_entries: list[dict[str, Any]] = []
    for doc_id in sorted(pages_by_doc.keys()):
        selected_pages = _select_top_pages_for_doc(
            doc_id,
            sorted(pages_by_doc[doc_id]),
            doc_chunk_terms_by_doc.get(doc_id, set()),
            query_terms,
            answer_terms,
            page_texts_by_doc,
            intent,
        )
        if not selected_pages:
            continue

        scored_pages = _score_selected_pages_for_doc(
            doc_id,
            selected_pages,
            doc_chunk_terms_by_doc.get(doc_id, set()),
            query_terms,
            answer_terms,
            page_texts_by_doc,
            intent,
        )
        best_score = scored_pages[0] if scored_pages else (0.0, 0, 0, 0, selected_pages[0])
        doc_entries.append(
            {
                "doc_id": doc_id,
                "pages": sorted(selected_pages),
                "scored_pages": scored_pages,
                "doc_score": best_score[0],
                "chunk_overlap": best_score[1],
                "query_overlap": best_score[2],
                "answer_overlap": best_score[3],
            }
        )

    if not doc_entries:
        return []

    doc_entries.sort(
        key=lambda entry: (
            -entry["doc_score"],
            -entry["chunk_overlap"],
            -entry["query_overlap"],
            -entry["answer_overlap"],
            entry["doc_id"],
        )
    )

    kept_entries = doc_entries[: _max_docs_limit(intent, answer_type)]
    if not kept_entries:
        return []

    total_page_limit = _max_total_pages_limit(intent, answer_type, len(kept_entries))
    current_total = sum(len(entry["pages"]) for entry in kept_entries)
    if current_total <= total_page_limit:
        return [{"doc_id": entry["doc_id"], "page_numbers": entry["pages"]} for entry in kept_entries]

    selected_by_doc: dict[str, list[int]] = {entry["doc_id"]: [] for entry in kept_entries}
    remaining_candidates: list[tuple[float, int, int, int, str, int]] = []

    if len(kept_entries) <= total_page_limit:
        for entry in kept_entries:
            if not entry["scored_pages"]:
                continue
            first_page = int(entry["scored_pages"][0][4])
            selected_by_doc[entry["doc_id"]].append(first_page)
            for score, chunk_overlap, query_overlap, answer_overlap, page_num in entry["scored_pages"][1:]:
                remaining_candidates.append((score, chunk_overlap, query_overlap, answer_overlap, entry["doc_id"], int(page_num)))
    else:
        for entry in kept_entries:
            for score, chunk_overlap, query_overlap, answer_overlap, page_num in entry["scored_pages"]:
                remaining_candidates.append((score, chunk_overlap, query_overlap, answer_overlap, entry["doc_id"], int(page_num)))

    remaining_candidates.sort(key=lambda item: (-item[0], -item[1], -item[2], -item[3], item[4], item[5]))
    current_total = sum(len(pages) for pages in selected_by_doc.values())

    for _score, _chunk_overlap, _query_overlap, _answer_overlap, doc_id, page_num in remaining_candidates:
        if current_total >= total_page_limit:
            break
        if page_num in selected_by_doc[doc_id]:
            continue
        selected_by_doc[doc_id].append(page_num)
        current_total += 1

    result = []
    for entry in kept_entries:
        pages = sorted(selected_by_doc[entry["doc_id"]])
        if not pages:
            continue
        result.append({"doc_id": entry["doc_id"], "page_numbers": pages})
    return result



def collect_grounding_pages(
    reranked_chunks: list[tuple[dict, float]],
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    is_null: bool = False,
    question_text: str = "",
    answer_text: str = "",
    page_texts_by_doc: dict[str, dict[int, str]] | None = None,
    intent: GroundingIntent | None = None,
    allowed_doc_ids: set[str] | None = None,
    answer_type: str = "",
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
        allowed_doc_ids: Optional doc restriction for page retrieval rescue.
        answer_type: Answer type used to apply stricter minimal-proof caps.

    Returns:
        List of {"doc_id": str, "page_numbers": [int]} for submission.
        Merged by doc_id, pages sorted.
    """
    if is_null:
        return []

    pages_by_doc: dict[str, set[int]] = {}
    doc_chunk_terms_by_doc: dict[str, set[str]] = {}
    query_terms = _tokenize(question_text)
    answer_terms = _tokenize(answer_text)
    if page_texts_by_doc is None:
        page_texts_by_doc = _load_page_texts()

    normalized_allowed_doc_ids = _normalize_allowed_doc_ids(allowed_doc_ids, reranked_chunks)

    for chunk, score in reranked_chunks:
        if score < score_threshold:
            continue

        doc_id = str(chunk.get("doc_id") or "").strip()
        if not doc_id:
            continue

        doc_chunk_terms_by_doc.setdefault(doc_id, set()).update(_tokenize(str(chunk.get("text") or "")))

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

    for page_ref, _score in _retrieve_additional_pages(
        question_text,
        answer_text,
        normalized_allowed_doc_ids,
        intent,
    ):
        doc_id = str(page_ref.get("doc_id") or "").strip()
        page_num = int(page_ref.get("page_num") or 0)
        if not doc_id or page_num <= 0:
            continue
        if normalized_allowed_doc_ids and doc_id not in normalized_allowed_doc_ids:
            continue
        if doc_id in page_texts_by_doc and page_num not in page_texts_by_doc.get(doc_id, {}):
            continue
        pages_by_doc.setdefault(doc_id, set()).add(page_num)
        doc_chunk_terms_by_doc.setdefault(doc_id, set())

    return _finalize_grounding_results(
        pages_by_doc,
        doc_chunk_terms_by_doc,
        query_terms,
        answer_terms,
        page_texts_by_doc,
        intent,
        answer_type,
    )



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
