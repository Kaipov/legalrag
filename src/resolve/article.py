from __future__ import annotations

import re

from src.resolve.metadata_store import PageMetadataStore, load_default_metadata_store
from src.resolve.models import EvidencePage
from src.retrieve.grounding_utils import (
    QuestionAnchors,
    classify_article_page_match,
    extract_question_anchors,
    match_target_law_doc,
    score_grounding_page,
)
from src.retrieve.lexical import tokenize_legal_text

_STRUCTURED_ANSWER_TYPES = {"number", "date", "boolean", "name", "names"}
_ARTICLE_REF_PARSE_RE = re.compile(r"\bArticle\s+(?P<number>\d+[A-Z]?)(?P<tail>(?:\([^)]+\))*)", re.IGNORECASE)
_TOP_LEVEL_ARTICLE_HEADING_RE = re.compile(r"(?im)^\s*(?P<number>\d+[A-Z]?)\.\s+\S")


def _single_article_anchors(anchors: QuestionAnchors, article_ref: str) -> QuestionAnchors:
    return QuestionAnchors(
        article_refs=(article_ref,),
        quoted_sections=anchors.quoted_sections,
        law_number=anchors.law_number,
        law_title_mentions=anchors.law_title_mentions,
        case_ids=anchors.case_ids,
        lexical_tokens=anchors.lexical_tokens,
    )


def _page_text_blob(record: dict) -> str:
    return " ".join(
        part
        for part in (
            str(record.get("doc_title") or ""),
            str(record.get("section_path") or ""),
            str(record.get("text") or ""),
        )
        if part
    )


def _parse_article_reference(article_ref: str) -> tuple[str, tuple[str, ...]] | None:
    match = _ARTICLE_REF_PARSE_RE.search(str(article_ref or ""))
    if not match:
        return None
    article_number = str(match.group("number") or "").strip()
    clause_parts = tuple(part.strip() for part in re.findall(r"\(([^)]+)\)", match.group("tail") or "") if part.strip())
    if not article_number:
        return None
    return article_number, clause_parts


def _line_start_clause_positions(text: str, clause_value: str) -> list[int]:
    if not clause_value:
        return []
    return [
        match.start()
        for match in re.finditer(rf"(?im)^\s*\({re.escape(clause_value)}\)", str(text or ""))
    ]


def _next_clause_start(text: str, start_position: int) -> int:
    next_match = re.search(r"(?im)^\s*\([^)]+\)", str(text or "")[start_position + 1 :])
    if not next_match:
        return len(str(text or ""))
    return start_position + 1 + next_match.start()


def _has_clause_sequence(text: str, clause_parts: tuple[str, ...]) -> bool:
    if not clause_parts:
        return False

    current_position = -1
    for clause_value in clause_parts:
        positions = [position for position in _line_start_clause_positions(text, clause_value) if position > current_position]
        if not positions:
            return False
        current_position = positions[0]
    return True


def _extract_article_heading_numbers(text: str) -> tuple[str, ...]:
    return tuple(match.group("number") for match in _TOP_LEVEL_ARTICLE_HEADING_RE.finditer(str(text or "")))


def _extract_clause_windows(text: str, clause_parts: tuple[str, ...]) -> list[str]:
    normalized_text = str(text or "")
    if not clause_parts:
        return []

    windows: list[str] = []
    first_clause_positions = _line_start_clause_positions(normalized_text, clause_parts[0])
    for start_position in first_clause_positions:
        current_position = start_position
        matched_sequence = True
        for clause_value in clause_parts[1:]:
            next_positions = [
                position
                for position in _line_start_clause_positions(normalized_text, clause_value)
                if position > current_position and position - start_position <= 1400
            ]
            if not next_positions:
                matched_sequence = False
                break
            current_position = next_positions[0]
        if matched_sequence:
            if len(clause_parts) == 1 and clause_parts[0].isdigit():
                end_position = _next_clause_start(normalized_text, current_position)
            else:
                end_position = min(len(normalized_text), current_position + 700)
            windows.append(normalized_text[start_position:end_position])

    if windows:
        return windows

    for start_position in _line_start_clause_positions(normalized_text, clause_parts[-1]):
        if len(clause_parts) == 1 and clause_parts[-1].isdigit():
            end_position = _next_clause_start(normalized_text, start_position)
        else:
            end_position = min(len(normalized_text), start_position + 700)
        windows.append(normalized_text[start_position:end_position])
    return windows


def _best_clause_window_overlap(
    text: str,
    clause_parts: tuple[str, ...],
    *,
    query_terms: set[str],
    answer_terms: set[str],
) -> int:
    if not clause_parts:
        return 0

    focus_terms = set(query_terms) | set(answer_terms)
    if not focus_terms:
        return 0

    best_overlap = 0
    for window in _extract_clause_windows(text, clause_parts):
        window_terms = set(tokenize_legal_text(window))
        best_overlap = max(best_overlap, len(window_terms & focus_terms))
    return best_overlap


def _has_conflicting_article_heading(record: dict, article_number: str) -> int:
    if not article_number:
        return 0
    heading_numbers = _extract_article_heading_numbers(str(record.get("text") or ""))
    return int(any(number != article_number for number in heading_numbers))


def _is_contents_like_page(record: dict) -> bool:
    text_blob = _page_text_blob(record)
    lowered = text_blob.lower()
    if "contents" in lowered[:250]:
        return True
    if text_blob.count("...") >= 3 or text_blob.count("…") >= 3:
        return True
    return False


def _residual_query_terms(question_text: str, anchors: QuestionAnchors) -> set[str]:
    residual = str(question_text or "")
    removable_values = list(anchors.article_refs) + list(anchors.quoted_sections) + list(anchors.law_title_mentions) + list(anchors.case_ids)
    if anchors.law_number:
        removable_values.append(anchors.law_number)
    for value in removable_values:
        residual = residual.replace(str(value), " ")
    return set(tokenize_legal_text(residual))


def _fallback_target_law_doc_id(question_text: str, anchors: QuestionAnchors, store: PageMetadataStore) -> str | None:
    if not anchors.law_title_mentions:
        return None

    query_terms = set(tokenize_legal_text(" ".join(anchors.law_title_mentions)))
    if not query_terms:
        return None

    ranked: list[tuple[int, str]] = []
    for record in store.get_first_page_records():
        doc_id = str(record.get("doc_id") or "").strip()
        if not doc_id:
            continue
        title_terms = set(tokenize_legal_text(str(record.get("doc_title") or ""), doc_title=str(record.get("doc_title") or "")))
        overlap = len(query_terms & title_terms)
        if overlap <= 0:
            continue
        ranked.append((overlap, doc_id))

    if not ranked:
        return None

    ranked.sort(key=lambda item: (-item[0], item[1]))
    best_overlap, best_doc_id = ranked[0]
    if len(ranked) > 1 and ranked[1][0] == best_overlap:
        return None
    return best_doc_id


def _article_specific_signals(record: dict, article_ref: str) -> tuple[int, int, int, int]:
    parsed_ref = _parse_article_reference(article_ref)
    if parsed_ref is None:
        return 0, 0, 0, 0

    article_number, clause_parts = parsed_ref
    text = str(record.get("text") or "")
    normalized_article_refs = {str(value).strip() for value in list(record.get("article_refs") or [])}
    has_exact_metadata_ref = int(article_ref in normalized_article_refs)
    has_article_heading = int(bool(re.search(rf"(?im)^\s*{re.escape(article_number)}\.\s+\S", text)))
    has_explicit_ref = int(bool(re.search(re.escape(article_ref), text, re.IGNORECASE)))
    has_clause_sequence = int(_has_clause_sequence(text, clause_parts))
    return has_exact_metadata_ref, has_article_heading, has_explicit_ref, has_clause_sequence


def _article_record_rank(
    record: dict,
    anchors: QuestionAnchors,
    *,
    question_text: str,
    answer_text: str,
    target_doc_id: str,
) -> tuple[int, int, int, int, int, int, float, int, str]:
    parsed_ref = _parse_article_reference(anchors.article_refs[0])
    clause_parts = parsed_ref[1] if parsed_ref is not None else ()
    has_exact_metadata_ref, has_article_heading, has_explicit_ref, has_clause_sequence = _article_specific_signals(
        record,
        anchors.article_refs[0],
    )
    text = str(record.get("text") or "")
    page_terms = set(tokenize_legal_text(_page_text_blob(record), doc_title=str(record.get("doc_title") or "")))
    query_terms = set(tokenize_legal_text(question_text))
    residual_terms = _residual_query_terms(question_text, anchors)
    answer_terms = set(tokenize_legal_text(answer_text))
    query_overlap = len(page_terms & query_terms)
    residual_overlap = len(page_terms & residual_terms)
    answer_overlap = len(page_terms & answer_terms)
    clause_window_overlap = _best_clause_window_overlap(
        text,
        clause_parts,
        query_terms=residual_terms or query_terms,
        answer_terms=answer_terms,
    )
    contents_penalty = 0 if _is_contents_like_page(record) else 1
    primary_signal = clause_window_overlap if clause_parts else has_exact_metadata_ref + has_explicit_ref
    secondary_signal = (has_exact_metadata_ref + has_explicit_ref) if clause_parts else has_article_heading
    tertiary_signal = has_article_heading if clause_parts else clause_window_overlap
    has_positive_signal = int(
        has_exact_metadata_ref
        or has_article_heading
        or has_explicit_ref
        or clause_window_overlap >= 2
        or (contents_penalty == 1 and residual_overlap >= 6 and not clause_parts)
    )
    if not has_positive_signal:
        return (0, 0, 0, 0, 0, 0, float("-inf"), 0, "")

    doc_id = str(record.get("doc_id") or "").strip()
    page_num = int(record.get("page_num") or 0)
    score = score_grounding_page(
        record,
        anchors,
        answer_text,
        intent=None,
        target_law_doc_id=target_doc_id,
    )
    return (
        has_positive_signal,
        primary_signal,
        secondary_signal,
        tertiary_signal,
        contents_penalty,
        residual_overlap,
        query_overlap + answer_overlap,
        has_clause_sequence - _has_conflicting_article_heading(record, parsed_ref[0] if parsed_ref is not None else ""),
        score,
        -page_num,
        doc_id,
    )


def select_article_evidence_pages(
    question_text: str,
    answer_type: str,
    *,
    answer_text: str = "",
    store: PageMetadataStore | None = None,
) -> list[EvidencePage]:
    normalized_answer_type = str(answer_type or "").lower()
    if normalized_answer_type not in _STRUCTURED_ANSWER_TYPES:
        return []

    metadata_store = store or load_default_metadata_store()
    if metadata_store is None:
        return []

    anchors = extract_question_anchors(question_text)
    if not anchors.article_refs:
        return []

    target_doc_id = match_target_law_doc(question_text, metadata_store.get_first_page_records())
    if not target_doc_id:
        target_doc_id = _fallback_target_law_doc_id(question_text, anchors, metadata_store)
    if not target_doc_id:
        return []

    candidate_records = [
        record
        for record in metadata_store.records
        if str(record.get("doc_id") or "").strip() == target_doc_id
    ]
    if not candidate_records:
        return []

    selected_pages: list[EvidencePage] = []
    seen_pages: set[tuple[str, int]] = set()

    for article_ref in anchors.article_refs:
        article_anchors = _single_article_anchors(anchors, article_ref)
        ranked_candidates = sorted(
            candidate_records,
            key=lambda record: _article_record_rank(
                record,
                article_anchors,
                question_text=question_text,
                answer_text=answer_text,
                target_doc_id=target_doc_id,
            ),
            reverse=True,
        )
        if not ranked_candidates:
            return []

        best_record = ranked_candidates[0]
        best_rank = _article_record_rank(
            best_record,
            article_anchors,
            question_text=question_text,
            answer_text=answer_text,
            target_doc_id=target_doc_id,
        )
        if best_rank[0] == 0:
            return []

        doc_id = str(best_record.get("doc_id") or "").strip()
        page_num = int(best_record.get("page_num") or 0)
        page_key = (doc_id, page_num)
        if not doc_id or page_num <= 0 or page_key in seen_pages:
            continue
        seen_pages.add(page_key)
        selected_pages.append(EvidencePage(doc_id=doc_id, page_num=page_num))

        parsed_ref = _parse_article_reference(article_ref)
        if parsed_ref is None:
            continue
        _article_number, clause_parts = parsed_ref
        if not clause_parts:
            continue
        if len(ranked_candidates) < 2:
            continue

        second_record = ranked_candidates[1]
        second_rank = _article_record_rank(
            second_record,
            article_anchors,
            question_text=question_text,
            answer_text=answer_text,
            target_doc_id=target_doc_id,
        )
        second_doc_id = str(second_record.get("doc_id") or "").strip()
        second_page_num = int(second_record.get("page_num") or 0)
        if second_rank[0] == 0 or second_doc_id != doc_id:
            continue
        if second_page_num != page_num + 1:
            continue
        if second_rank[1] + 2 < best_rank[1]:
            continue
        if second_rank[5] + 2 < best_rank[5]:
            continue
        if _has_conflicting_article_heading(second_record, parsed_ref[0]):
            continue
        second_key = (second_doc_id, second_page_num)
        if second_page_num <= 0 or second_key in seen_pages:
            continue
        seen_pages.add(second_key)
        selected_pages.append(EvidencePage(doc_id=second_doc_id, page_num=second_page_num))

    return selected_pages
