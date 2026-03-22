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

from src.config import PAGE_METADATA_JSONL, PAGES_JSONL
from src.retrieve.grounding_policy import GroundingIntent
from src.retrieve.grounding_utils import (
    QuestionAnchors,
    classify_article_page_match,
    extract_question_anchors,
    match_target_law_doc,
    score_grounding_page,
)
from src.retrieve.lexical import tokenize_legal_text

logger = logging.getLogger(__name__)

# Default reranker score threshold for including a chunk's pages
DEFAULT_SCORE_THRESHOLD = -1.0  # bge-reranker outputs logits, can be negative
MAX_PAGES_PER_CHUNK = 2
MAX_PAGES_PER_WIDE_CHUNK = 3
_PAGE_TEXTS: dict[str, dict[int, str]] | None = None
_PAGE_RECORDS_BY_DOC: dict[str, dict[int, dict[str, Any]]] | None = None
_FIRST_PAGE_RECORDS: list[dict[str, Any]] | None = None
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



def _load_page_records() -> tuple[dict[str, dict[int, dict[str, Any]]], list[dict[str, Any]]]:
    global _PAGE_RECORDS_BY_DOC
    global _FIRST_PAGE_RECORDS

    if _PAGE_RECORDS_BY_DOC is not None and _FIRST_PAGE_RECORDS is not None:
        return _PAGE_RECORDS_BY_DOC, _FIRST_PAGE_RECORDS

    records_by_doc: dict[str, dict[int, dict[str, Any]]] = {}
    first_page_records: list[dict[str, Any]] = []
    path = Path(PAGE_METADATA_JSONL)
    if not path.exists():
        logger.warning("Grounding page metadata %s not found; falling back to page texts", path)
        _PAGE_RECORDS_BY_DOC = records_by_doc
        _FIRST_PAGE_RECORDS = first_page_records
        return _PAGE_RECORDS_BY_DOC, _FIRST_PAGE_RECORDS

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            doc_id = str(record.get("doc_id") or "").strip()
            page_num = int(record.get("page_num") or 0)
            if not doc_id or page_num <= 0:
                continue
            records_by_doc.setdefault(doc_id, {})[page_num] = record
            if bool(record.get("is_first_page")) or page_num == 1:
                first_page_records.append(record)

    first_page_records = sorted(
        first_page_records,
        key=lambda item: (str(item.get("doc_id") or ""), int(item.get("page_num") or 0)),
    )
    _PAGE_RECORDS_BY_DOC = records_by_doc
    _FIRST_PAGE_RECORDS = first_page_records
    return _PAGE_RECORDS_BY_DOC, _FIRST_PAGE_RECORDS



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
    page_record: dict[str, Any],
    query_terms: set[str],
    answer_terms: set[str],
    chunk_terms: set[str],
    page_num: int,
    doc_last_page: int,
    intent: GroundingIntent | None,
    anchors: QuestionAnchors,
    cited_page_keys: set[tuple[str, int]],
    target_law_doc_id: str | None,
) -> tuple[float, int, int, int, int]:
    page_text = str(page_record.get("text") or "")
    page_terms = _tokenize(" ".join(part for part in (str(page_record.get("doc_title") or ""), str(page_record.get("section_path") or ""), page_text) if part))
    chunk_overlap = len(page_terms & chunk_terms)
    query_overlap = len(page_terms & query_terms)
    answer_overlap = len(page_terms & answer_terms)
    support_signals = _page_support_signal_count(
        page_record,
        anchors,
        query_terms,
        answer_terms,
        cited_page_keys,
        target_law_doc_id=target_law_doc_id,
    )
    score = score_grounding_page(
        page_record,
        anchors,
        " ".join(sorted(answer_terms)),
        intent,
        cited_page_keys,
        target_law_doc_id=target_law_doc_id,
    )
    score += (1.75 * chunk_overlap) + (1.35 * query_overlap) + (1.1 * answer_overlap)
    score += _page_focus_bonus(page_num, doc_last_page, intent)
    score += _intent_text_bonus(page_text, intent)
    return score, support_signals, chunk_overlap, query_overlap, answer_overlap



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
    if intent is not None and intent.max_docs is not None:
        return max(1, int(intent.max_docs))
    if intent is None or intent.kind == "generic":
        return 1 if _is_structured_answer_type(answer_type) else 2
    if intent.is_compare:
        if intent.case_ids:
            return max(2, min(3, len(intent.case_ids)))
        return 2
    if intent.kind == "article_ref":
        return 1
    if intent.kind == "last_page":
        return max(1, min(2, len(intent.case_ids) or 1))
    if intent.is_page_local:
        return max(1, min(2, len(intent.case_ids) or 1))
    return 2



def _max_total_pages_limit(intent: GroundingIntent | None, answer_type: str, doc_count: int) -> int:
    if intent is not None and intent.max_total_pages is not None:
        return max(1, int(intent.max_total_pages))
    if intent is None or intent.kind == "generic":
        return 2 if _is_structured_answer_type(answer_type) else 4
    if intent.is_compare:
        return max(2, min(3, doc_count))
    if intent.kind == "article_ref":
        return 2
    if intent.kind == "last_page":
        return 2
    if intent.is_page_local:
        return max(1, doc_count)
    return max(1, min(2, doc_count))



def _generic_intent_from(intent: GroundingIntent | None) -> GroundingIntent | None:
    if intent is None:
        return None
    return GroundingIntent(
        kind="generic",
        case_ids=intent.case_ids,
        quoted_sections=intent.quoted_sections,
        max_pages_per_doc=2,
        max_total_pages=4,
    )



def _resolve_grounding_context(
    question_text: str,
    intent: GroundingIntent | None,
    first_page_records: list[dict[str, Any]],
) -> tuple[GroundingIntent | None, QuestionAnchors, str | None]:
    anchors = extract_question_anchors(question_text)
    if intent is None or intent.kind != "article_ref":
        return intent, anchors, None

    target_law_doc_id = match_target_law_doc(question_text, first_page_records)
    if not target_law_doc_id:
        return _generic_intent_from(intent), anchors, None
    return intent, anchors, target_law_doc_id



def _build_fallback_page_record(
    doc_id: str,
    page_num: int,
    page_texts_by_doc: dict[str, dict[int, str]],
) -> dict[str, Any]:
    return {
        "doc_id": doc_id,
        "page_num": page_num,
        "doc_title": "",
        "section_path": "",
        "case_ids": [],
        "article_refs": [],
        "text": str(page_texts_by_doc.get(doc_id, {}).get(page_num, "") or ""),
    }



def _get_page_record(
    doc_id: str,
    page_num: int,
    page_texts_by_doc: dict[str, dict[int, str]],
    page_records_by_doc: dict[str, dict[int, dict[str, Any]]],
) -> dict[str, Any]:
    doc_records = page_records_by_doc.get(doc_id, {})
    if page_num in doc_records:
        return doc_records[page_num]
    return _build_fallback_page_record(doc_id, page_num, page_texts_by_doc)



def _page_support_signal_count(
    page_record: dict[str, Any],
    anchors: QuestionAnchors,
    query_terms: set[str],
    answer_terms: set[str],
    cited_page_keys: set[tuple[str, int]],
    *,
    target_law_doc_id: str | None,
) -> int:
    text_blob = " ".join(
        part
        for part in (
            str(page_record.get("doc_title") or ""),
            str(page_record.get("section_path") or ""),
            str(page_record.get("text") or ""),
        )
        if part
    )
    lowered_blob = text_blob.lower()
    page_tokens = _tokenize(text_blob)
    page_key = (str(page_record.get("doc_id") or "").strip(), int(page_record.get("page_num") or 0))
    article_match = classify_article_page_match(page_record, anchors, answer_text=" ".join(sorted(answer_terms)))

    support_signals = 0
    if page_key in cited_page_keys:
        support_signals += 1
    if article_match.kind == "definition":
        support_signals += 1
    elif article_match.kind == "citation":
        support_signals += 1
    if any(section.lower() in lowered_blob for section in anchors.quoted_sections):
        support_signals += 1
    if anchors.law_number and anchors.law_number.lower() in lowered_blob:
        support_signals += 1
    if target_law_doc_id and str(page_record.get("doc_id") or "").strip() == target_law_doc_id:
        support_signals += 1
    if anchors.case_ids and any(case_id in text_blob.upper() for case_id in anchors.case_ids):
        support_signals += 1
    if page_tokens & query_terms:
        support_signals += 1
    if page_tokens & answer_terms:
        support_signals += 1
    return support_signals



def _page_has_exact_article_match(page_record: dict[str, Any], anchors: QuestionAnchors) -> bool:
    return classify_article_page_match(page_record, anchors).kind == "definition"



def _score_selected_pages_for_doc(
    doc_id: str,
    candidate_pages: list[int],
    chunk_terms: set[str],
    query_terms: set[str],
    answer_terms: set[str],
    page_texts_by_doc: dict[str, dict[int, str]],
    intent: GroundingIntent | None,
    anchors: QuestionAnchors,
    cited_page_keys: set[tuple[str, int]],
    target_law_doc_id: str | None,
    page_records_by_doc: dict[str, dict[int, dict[str, Any]]],
) -> list[tuple[float, int, int, int, int, int]]:
    doc_pages = page_texts_by_doc.get(doc_id, {})
    doc_last_page = max(doc_pages.keys(), default=max(candidate_pages))
    scored_pages: list[tuple[float, int, int, int, int, int]] = []
    for page_num in candidate_pages:
        score, support_signals, chunk_overlap, query_overlap, answer_overlap = _score_page(
            _get_page_record(doc_id, page_num, page_texts_by_doc, page_records_by_doc),
            query_terms,
            answer_terms,
            chunk_terms,
            page_num,
            doc_last_page,
            intent,
            anchors,
            cited_page_keys,
            target_law_doc_id,
        )
        scored_pages.append((score, support_signals, chunk_overlap, query_overlap, answer_overlap, page_num))
    scored_pages.sort(key=lambda item: (-item[0], -item[1], -item[2], -item[3], -item[4], item[5]))
    return scored_pages



def _select_top_pages_for_doc(
    doc_id: str,
    candidate_pages: list[int],
    chunk_terms: set[str],
    query_terms: set[str],
    answer_terms: set[str],
    page_texts_by_doc: dict[str, dict[int, str]],
    intent: GroundingIntent | None,
    anchors: QuestionAnchors,
    cited_page_keys: set[tuple[str, int]],
    target_law_doc_id: str | None,
    page_records_by_doc: dict[str, dict[int, dict[str, Any]]],
) -> list[int]:
    limit = _max_pages_per_doc_limit(intent)
    if len(candidate_pages) <= limit and intent is not None and intent.kind not in {"generic", "article_ref"}:
        return _rank_pages_by_intent(candidate_pages, max(candidate_pages), intent)

    scored_pages = _score_selected_pages_for_doc(
        doc_id,
        candidate_pages,
        chunk_terms,
        query_terms,
        answer_terms,
        page_texts_by_doc,
        intent,
        anchors,
        cited_page_keys,
        target_law_doc_id,
        page_records_by_doc,
    )
    if all(item[0] == 0 and item[1] == 0 and item[2] == 0 and item[3] == 0 and item[4] == 0 for item in scored_pages):
        doc_pages = page_texts_by_doc.get(doc_id, {})
        doc_last_page = max(doc_pages.keys(), default=max(candidate_pages))
        return _rank_pages_by_intent(candidate_pages, doc_last_page, intent)

    selected_pages: list[int] = []
    for index, item in enumerate(scored_pages):
        _score, support_signals, _chunk_overlap, query_overlap, answer_overlap, page_num = item
        if index == 0:
            selected_pages.append(page_num)
            continue
        if len(selected_pages) >= limit:
            break
        if (intent is None or intent.kind == "generic" or intent.kind == "article_ref") and support_signals <= 1 and query_overlap == 0 and answer_overlap == 0:
            continue
        selected_pages.append(page_num)

    return selected_pages



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
    anchors: QuestionAnchors | None = None,
    cited_page_keys: set[tuple[str, int]] | None = None,
    target_law_doc_id: str | None = None,
    page_records_by_doc: dict[str, dict[int, dict[str, Any]]] | None = None,
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

    anchors = anchors or QuestionAnchors()
    cited_page_keys = cited_page_keys or set()
    page_records_by_doc = page_records_by_doc or {}
    chunk_terms = _tokenize(str(chunk.get("text") or ""))
    scored_pages = _score_selected_pages_for_doc(
        doc_id,
        candidate_pages,
        chunk_terms,
        query_terms,
        answer_terms,
        page_texts_by_doc,
        intent,
        anchors,
        cited_page_keys,
        target_law_doc_id,
        page_records_by_doc,
    )

    if all(item[0] == 0 and item[1] == 0 and item[2] == 0 and item[3] == 0 and item[4] == 0 for item in scored_pages):
        ranked_pages = _rank_pages_by_intent(candidate_pages, doc_last_page, intent)
        return sorted(ranked_pages[:limit])

    ranked_selected_pages = _select_top_pages_for_doc(
        doc_id,
        candidate_pages,
        chunk_terms,
        query_terms,
        answer_terms,
        page_texts_by_doc,
        intent,
        anchors,
        cited_page_keys,
        target_law_doc_id,
        page_records_by_doc,
    )
    return sorted(ranked_selected_pages[:limit])



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
    anchors: QuestionAnchors,
    cited_page_keys: set[tuple[str, int]],
    target_law_doc_id: str | None,
    page_records_by_doc: dict[str, dict[int, dict[str, Any]]],
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
            anchors,
            cited_page_keys,
            target_law_doc_id,
            page_records_by_doc,
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
            anchors,
            cited_page_keys,
            target_law_doc_id,
            page_records_by_doc,
        )
        best_score = scored_pages[0] if scored_pages else (0.0, 0, 0, 0, 0, selected_pages[0])
        doc_entries.append(
            {
                "doc_id": doc_id,
                "pages": sorted(selected_pages),
                "scored_pages": scored_pages,
                "doc_score": best_score[0],
                "support_signals": best_score[1],
                "chunk_overlap": best_score[2],
                "query_overlap": best_score[3],
                "answer_overlap": best_score[4],
            }
        )

    if not doc_entries:
        return []

    if intent is not None and intent.kind == "article_ref" and _is_structured_answer_type(answer_type):
        narrowed_entries: list[dict[str, Any]] = []
        for entry in doc_entries:
            exact_scored_pages = [
                scored_page
                for scored_page in entry["scored_pages"]
                if _page_has_exact_article_match(
                    _get_page_record(entry["doc_id"], int(scored_page[5]), page_texts_by_doc, page_records_by_doc),
                    anchors,
                )
            ]
            if exact_scored_pages:
                best_exact_page = exact_scored_pages[0]
                entry = dict(entry)
                entry["pages"] = [int(best_exact_page[5])]
                entry["scored_pages"] = [best_exact_page]
                entry["doc_score"] = best_exact_page[0]
                entry["support_signals"] = best_exact_page[1]
                entry["chunk_overlap"] = best_exact_page[2]
                entry["query_overlap"] = best_exact_page[3]
                entry["answer_overlap"] = best_exact_page[4]
            narrowed_entries.append(entry)
        doc_entries = narrowed_entries

    doc_entries.sort(
        key=lambda entry: (
            -entry["doc_score"],
            -entry["support_signals"],
            -entry["chunk_overlap"],
            -entry["query_overlap"],
            -entry["answer_overlap"],
            entry["doc_id"],
        )
    )

    max_docs_limit = _max_docs_limit(intent, answer_type)
    multi_doc_signal = len({doc_id for doc_id, _page_num in cited_page_keys}) > 1 or len(anchors.case_ids) > 1
    if (intent is None or intent.kind == "generic") and _is_structured_answer_type(answer_type) and multi_doc_signal:
        max_docs_limit = max(2, max_docs_limit)

    kept_entries = doc_entries[:max_docs_limit]
    if not kept_entries:
        return []

    total_page_limit = _max_total_pages_limit(intent, answer_type, len(kept_entries))
    if (intent is None or intent.kind == "generic") and _is_structured_answer_type(answer_type) and multi_doc_signal:
        total_page_limit = max(3, total_page_limit)
    if intent is not None and intent.kind == "article_ref" and _is_structured_answer_type(answer_type):
        exact_article_present = any(
            _page_has_exact_article_match(
                _get_page_record(entry["doc_id"], int(page_num), page_texts_by_doc, page_records_by_doc),
                anchors,
            )
            for entry in kept_entries
            for page_num in entry["pages"]
        )
        if exact_article_present:
            total_page_limit = 1
    current_total = sum(len(entry["pages"]) for entry in kept_entries)
    if current_total <= total_page_limit:
        return [{"doc_id": entry["doc_id"], "page_numbers": entry["pages"]} for entry in kept_entries]

    selected_by_doc: dict[str, list[int]] = {entry["doc_id"]: [] for entry in kept_entries}
    remaining_candidates: list[tuple[float, int, int, int, str, int]] = []

    if len(kept_entries) <= total_page_limit:
        for entry in kept_entries:
            if not entry["scored_pages"]:
                continue
            first_page = int(entry["scored_pages"][0][5])
            selected_by_doc[entry["doc_id"]].append(first_page)
            for score, support_signals, chunk_overlap, query_overlap, answer_overlap, page_num in entry["scored_pages"][1:]:
                remaining_candidates.append(
                    (score, support_signals, chunk_overlap, query_overlap, answer_overlap, entry["doc_id"], int(page_num))
                )
    else:
        for entry in kept_entries:
            for score, support_signals, chunk_overlap, query_overlap, answer_overlap, page_num in entry["scored_pages"]:
                remaining_candidates.append(
                    (score, support_signals, chunk_overlap, query_overlap, answer_overlap, entry["doc_id"], int(page_num))
                )

    remaining_candidates.sort(key=lambda item: (-item[0], -item[1], -item[2], -item[3], -item[4], item[5], item[6]))
    current_total = sum(len(pages) for pages in selected_by_doc.values())

    for _score, support_signals, _chunk_overlap, query_overlap, answer_overlap, doc_id, page_num in remaining_candidates:
        if current_total >= total_page_limit:
            break
        if page_num in selected_by_doc[doc_id]:
            continue
        if (intent is None or intent.kind == "generic" or intent.kind == "article_ref") and support_signals <= 1 and query_overlap == 0 and answer_overlap == 0:
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
    page_records_by_doc: dict[str, dict[int, dict[str, Any]]] | None = None,
    first_page_records: list[dict[str, Any]] | None = None,
    intent: GroundingIntent | None = None,
    allowed_doc_ids: set[str] | None = None,
    answer_type: str = "",
    cited_page_keys: set[tuple[str, int]] | None = None,
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
    if page_records_by_doc is None or first_page_records is None:
        loaded_page_records_by_doc, loaded_first_page_records = _load_page_records()
        if page_records_by_doc is None:
            page_records_by_doc = loaded_page_records_by_doc
        if first_page_records is None:
            first_page_records = loaded_first_page_records
    effective_intent, anchors, target_law_doc_id = _resolve_grounding_context(
        question_text,
        intent,
        first_page_records or [],
    )
    cited_page_keys = set(cited_page_keys or set())

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
            intent=effective_intent,
            anchors=anchors,
            cited_page_keys=cited_page_keys,
            target_law_doc_id=target_law_doc_id,
            page_records_by_doc=page_records_by_doc,
        )
        if not selected_pages:
            continue

        pages_by_doc.setdefault(doc_id, set()).update(selected_pages)
        if chunk.get("__is_cited__"):
            cited_page_keys.update((doc_id, int(page_num)) for page_num in selected_pages)

    for page_ref, _score in _retrieve_additional_pages(
        question_text,
        answer_text,
        normalized_allowed_doc_ids,
        effective_intent,
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
        effective_intent,
        answer_type,
        anchors,
        cited_page_keys,
        target_law_doc_id,
        page_records_by_doc,
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
