from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable

from src.case_ids import extract_case_ids as _extract_case_id_list
from src.retrieve.lexical import tokenize_legal_text

_ARTICLE_REF_RE = re.compile(r"\bArticle\s+\d+[A-Z]?(?:\(\d+[A-Z]?\)|\([a-z]\))*", re.IGNORECASE)
_LAW_NUMBER_RE = re.compile(r"\b(?:DIFC|DFSA)\s+Law\s+No\.?\s+\d+\s+of\s+\d{4}\b", re.IGNORECASE)
_LAW_TITLE_RE = re.compile(
    r"\b(?P<title>(?:(?:[A-Z][A-Za-z&/\-]*|DIFC|DFSA)\s+){1,6}(?:Law|Regulation|Regulations|Rules?))(?:\s+\d{4})?\b"
)
_ARTICLE_LAW_TITLE_RE = re.compile(
    r"\bArticle\s+\d+[A-Z]?(?:\(\d+[A-Z]?\)|\([a-z]\))*\s+of\s+(?:the\s+)?"
    r"(?P<title>(?:(?:[A-Z][A-Za-z&/\-]*|DIFC|DFSA)\s+){1,6}(?:Law|Regulation|Regulations|Rules?))(?:\s+\d{4})?\b"
)
_SCOPED_LAW_TITLE_RE = re.compile(
    r"\b(?:the|under the|of the|pursuant to the|according to the)\s+"
    r"(?P<title>(?:(?:[A-Z][A-Za-z&/\-]*|DIFC|DFSA)\s+){1,6}(?:Law|Regulation|Regulations|Rules?))(?:\s+\d{4})?\b",
    re.IGNORECASE,
)
_QUOTED_SECTION_RE = re.compile(r"['\"](?P<value>[^'\"]{3,120})['\"]")
_GENERIC_TITLE_PREFIXES = ("the", "this", "that", "which", "what", "under", "according", "article")


@dataclass(frozen=True)
class QuestionAnchors:
    article_refs: tuple[str, ...] = ()
    quoted_sections: tuple[str, ...] = ()
    law_number: str | None = None
    law_title_mentions: tuple[str, ...] = ()
    case_ids: tuple[str, ...] = ()
    lexical_tokens: tuple[str, ...] = ()


@dataclass(frozen=True)
class ArticlePageMatch:
    kind: str = "none"
    article_ref: str | None = None
    has_heading: bool = False
    has_clause_sequence: bool = False
    explicit_reference: bool = False
    explicit_cross_reference: bool = False
    answer_overlap: int = 0


_ARTICLE_REF_PARSE_RE = re.compile(r"\bArticle\s+(?P<number>\d+[A-Z]?)(?P<tail>(?:\([^)]+\))*)", re.IGNORECASE)
_CROSS_REFERENCE_PREFIX_RE = re.compile(
    r"(?:under|subject to|pursuant to|according to|in accordance with|for the purposes of|"
    r"provided under|set out in|specified in|under the provisions of|the provisions of|"
    r"within|pursuant only to)\s+$",
    re.IGNORECASE,
)


def _dedupe_preserve_order(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = " ".join(str(value or "").split()).strip()
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(normalized)
    return tuple(ordered)


def _normalize_law_title(value: str) -> str:
    normalized = " ".join(str(value or "").split()).strip(" ,?.")
    lowered = normalized.lower()
    for prefix in ("the ", "under the ", "under ", "of the ", "of ", "according to the ", "according to "):
        if lowered.startswith(prefix):
            normalized = normalized[len(prefix):]
            lowered = normalized.lower()
    return normalized.strip()


def _extract_law_number_from_text(text: str) -> str | None:
    match = _LAW_NUMBER_RE.search(str(text or ""))
    if not match:
        return None
    return " ".join(match.group(0).split())


def _has_indefinite_article_prefix(text: str, start_index: int) -> bool:
    prefix = str(text or "")[max(0, start_index - 12):start_index].lower()
    return bool(re.search(r"\b(?:a|an)\s+$", prefix))


def _is_law_title_candidate(title: str, raw_text: str, start_index: int) -> bool:
    if not title:
        return False
    if _has_indefinite_article_prefix(raw_text, start_index):
        return False
    first_token = title.split(" ", 1)[0].lower()
    if first_token in _GENERIC_TITLE_PREFIXES:
        return False
    if len(title.split()) < 2:
        return False
    return True


def _extract_law_titles(question_text: str) -> tuple[str, ...]:
    raw_text = str(question_text or "")
    titles: list[str] = []
    for pattern in (_ARTICLE_LAW_TITLE_RE, _SCOPED_LAW_TITLE_RE, _LAW_TITLE_RE):
        for match in pattern.finditer(raw_text):
            title = _normalize_law_title(match.group("title"))
            if not _is_law_title_candidate(title, raw_text, match.start("title")):
                continue
            titles.append(title)
        if titles:
            return _dedupe_preserve_order(titles)
    return ()


def extract_question_anchors(question_text: str) -> QuestionAnchors:
    raw_text = str(question_text or "")
    article_refs = _dedupe_preserve_order(match.group(0) for match in _ARTICLE_REF_RE.finditer(raw_text))
    quoted_sections = _dedupe_preserve_order(match.group("value") for match in _QUOTED_SECTION_RE.finditer(raw_text))
    case_ids = tuple(_extract_case_id_list(raw_text))
    law_number = _extract_law_number_from_text(raw_text)
    law_title_mentions = _extract_law_titles(raw_text)
    lexical_tokens = _dedupe_preserve_order(tokenize_legal_text(raw_text))
    return QuestionAnchors(
        article_refs=article_refs,
        quoted_sections=quoted_sections,
        law_number=law_number,
        law_title_mentions=law_title_mentions,
        case_ids=case_ids,
        lexical_tokens=lexical_tokens,
    )


def _law_title_match_score(question_text: str, doc_title: str) -> tuple[int, int]:
    normalized_title = _normalize_law_title(doc_title)
    if not normalized_title:
        return 0, 0

    lowered_question = str(question_text or "").lower()
    lowered_title = normalized_title.lower()
    title_without_year = re.sub(r"\b\d{4}\b", "", lowered_title).strip()
    question_tokens = set(tokenize_legal_text(question_text))
    title_tokens = set(tokenize_legal_text(normalized_title, doc_title=normalized_title))

    score = 0
    overlap = len(question_tokens & title_tokens)
    if lowered_title and lowered_title in lowered_question:
        score += 100
    if title_without_year and title_without_year in lowered_question:
        score += 60
    score += overlap * 10
    return score, overlap


def match_target_law_doc(question_text: str, first_page_records: list[dict[str, Any]]) -> str | None:
    anchors = extract_question_anchors(question_text)
    if not first_page_records:
        return None

    if anchors.law_number:
        matching_doc_ids = {
            str(record.get("doc_id") or "").strip()
            for record in first_page_records
            if _extract_law_number_from_text(
                " ".join(
                    part
                    for part in (
                        str(record.get("doc_title") or ""),
                        str(record.get("text") or ""),
                    )
                    if part
                )
            ) == anchors.law_number
        }
        matching_doc_ids.discard("")
        if len(matching_doc_ids) == 1:
            return next(iter(matching_doc_ids))
        if len(matching_doc_ids) > 1:
            return None

    ranked_matches: list[tuple[int, int, str]] = []
    for record in first_page_records:
        doc_id = str(record.get("doc_id") or "").strip()
        if not doc_id:
            continue
        score, overlap = _law_title_match_score(question_text, str(record.get("doc_title") or ""))
        if score <= 0:
            continue
        ranked_matches.append((score, overlap, doc_id))

    if not ranked_matches:
        return None

    ranked_matches.sort(key=lambda item: (-item[0], -item[1], item[2]))
    best_score, _best_overlap, best_doc_id = ranked_matches[0]
    if best_score < 40:
        return None
    if len(ranked_matches) > 1 and ranked_matches[1][0] == best_score:
        return None
    return best_doc_id


def _page_text_blob(page_record: dict[str, Any]) -> str:
    return " ".join(
        part
        for part in (
            str(page_record.get("doc_title") or ""),
            str(page_record.get("section_path") or ""),
            str(page_record.get("text") or ""),
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


def _has_article_heading(text: str, article_number: str) -> bool:
    if not article_number:
        return False
    return bool(re.search(rf"(?im)^\s*{re.escape(article_number)}\.\s+\S", str(text or "")))


def _line_start_clause_positions(text: str, clause_value: str) -> list[int]:
    if not clause_value:
        return []
    return [
        match.start()
        for match in re.finditer(rf"(?im)^\s*\({re.escape(clause_value)}\)", str(text or ""))
    ]


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


def _explicit_article_reference_positions(text: str, article_ref: str) -> list[int]:
    if not article_ref:
        return []
    return [match.start() for match in re.finditer(re.escape(article_ref), str(text or ""), re.IGNORECASE)]


def _has_article_definition_intro(text: str, article_ref: str) -> bool:
    if not article_ref:
        return False
    return bool(
        re.search(
            rf"(?im)(?:^|[\n\r])\s*{re.escape(article_ref)}(?:[\s\.:;-]+|$)",
            str(text or ""),
        )
    )


def _is_cross_reference_hit(text: str, match_start: int) -> bool:
    prefix = str(text or "")[max(0, match_start - 48):match_start]
    normalized_prefix = " ".join(prefix.split())
    return bool(_CROSS_REFERENCE_PREFIX_RE.search(normalized_prefix))


def classify_article_page_match(
    page_record: dict[str, Any],
    anchors: QuestionAnchors,
    *,
    answer_text: str = "",
) -> ArticlePageMatch:
    if not anchors.article_refs:
        return ArticlePageMatch()

    page_text = str(page_record.get("text") or "")
    text_blob = _page_text_blob(page_record)
    article_refs = {str(value) for value in list(page_record.get("article_refs") or [])}
    page_tokens = set(tokenize_legal_text(text_blob, doc_title=str(page_record.get("doc_title") or "")))
    answer_tokens = set(tokenize_legal_text(str(answer_text or ""), doc_title=str(page_record.get("doc_title") or "")))
    answer_overlap = len(page_tokens & answer_tokens)

    best_match = ArticlePageMatch()
    best_rank = (-1, -1, -1, -1)

    for article_ref in anchors.article_refs:
        parsed_ref = _parse_article_reference(article_ref)
        if parsed_ref is None:
            continue
        article_number, clause_parts = parsed_ref
        has_heading = _has_article_heading(page_text, article_number)
        has_clause_sequence = _has_clause_sequence(page_text, clause_parts)
        explicit_positions = _explicit_article_reference_positions(page_text, article_ref)
        explicit_reference = bool(explicit_positions or article_ref in article_refs)
        explicit_cross_reference = bool(explicit_positions) and all(_is_cross_reference_hit(page_text, position) for position in explicit_positions)
        has_definition_intro = _has_article_definition_intro(page_text, article_ref)

        kind = "none"
        rank = (0, 0, 0, 0)
        if has_heading and (not clause_parts or has_clause_sequence):
            kind = "definition"
            rank = (3, 2 if has_heading else 0, 2 if has_clause_sequence else 0, min(3, answer_overlap))
        elif has_definition_intro and not explicit_cross_reference:
            kind = "definition"
            rank = (3, 1, 1 if has_clause_sequence else 0, min(3, answer_overlap))
        elif has_clause_sequence:
            kind = "definition"
            rank = (3, 0, 2, min(3, answer_overlap))
        elif explicit_reference and not explicit_cross_reference and not clause_parts:
            kind = "definition"
            rank = (2, 0, 1, min(3, answer_overlap))
        elif explicit_reference and not explicit_cross_reference and answer_overlap > 0:
            kind = "definition"
            rank = (2, 0, 1, min(3, answer_overlap))
        elif explicit_reference:
            kind = "citation"
            rank = (1, 1 if explicit_cross_reference else 0, 0, min(3, answer_overlap))

        match = ArticlePageMatch(
            kind=kind,
            article_ref=article_ref,
            has_heading=has_heading,
            has_clause_sequence=has_clause_sequence,
            explicit_reference=explicit_reference,
            explicit_cross_reference=explicit_cross_reference,
            answer_overlap=answer_overlap,
        )
        if rank > best_rank:
            best_rank = rank
            best_match = match

    return best_match


def score_grounding_page(
    page_record: dict[str, Any],
    anchors: QuestionAnchors,
    answer_value: str,
    intent: Any,
    cited_source_ids: Iterable[tuple[str, int]] = (),
    *,
    target_law_doc_id: str | None = None,
) -> float:
    text_blob = _page_text_blob(page_record)
    lowered_blob = text_blob.lower()
    doc_id = str(page_record.get("doc_id") or "").strip()
    page_num = int(page_record.get("page_num") or 0)
    cited_page_keys = set(cited_source_ids)
    page_tokens = set(tokenize_legal_text(text_blob, doc_title=str(page_record.get("doc_title") or "")))
    answer_tokens = set(tokenize_legal_text(str(answer_value or ""), doc_title=str(page_record.get("doc_title") or "")))
    anchor_tokens = set(anchors.lexical_tokens)

    score = 0.0
    if (doc_id, page_num) in cited_page_keys:
        score += 40.0

    if anchors.article_refs:
        article_match = classify_article_page_match(page_record, anchors, answer_text=str(answer_value or ""))
        if article_match.kind == "definition":
            score += 26.0
            if article_match.has_heading:
                score += 6.0
            if article_match.has_clause_sequence:
                score += 6.0
        elif article_match.kind == "citation":
            score += 8.0
            if article_match.explicit_cross_reference:
                score -= 3.0

    if anchors.quoted_sections:
        quoted_hits = sum(1 for section in anchors.quoted_sections if section.lower() in lowered_blob)
        score += quoted_hits * 12.0

    if anchors.law_number:
        page_law_number = _extract_law_number_from_text(text_blob)
        if page_law_number == anchors.law_number:
            score += 16.0

    if target_law_doc_id:
        if doc_id == target_law_doc_id:
            score += 14.0
        elif getattr(intent, "kind", "") == "article_ref":
            score -= 8.0

    case_ids = {str(value).upper() for value in list(page_record.get("case_ids") or [])}
    case_id_hits = sum(1 for case_id in anchors.case_ids if case_id in case_ids or case_id in text_blob.upper())
    score += min(2, case_id_hits) * 5.0

    lexical_overlap = len(page_tokens & anchor_tokens)
    answer_overlap = len(page_tokens & answer_tokens)
    score += min(6, lexical_overlap) * 1.5
    score += min(5, answer_overlap) * 1.2

    return score
