from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

_COMPARE_MARKERS = (
    "both case",
    "both cases",
    "across all documents",
    "across case",
    "common to both",
    "in both cases",
    "appeared in both",
    "involve any of the same",
)
_PARTY_MARKERS = ("party", "parties", "claimant", "defendant", "main party", "applicant", "respondent")
_CASE_ID_RE = re.compile(r"\b(?:CFI|SCT|ENF|CA|ARB|TCD|DEC)\s*\d{3}/\d{4}\b", re.IGNORECASE)


@dataclass(frozen=True)
class GroundingIntent:
    kind: str
    page_focus: str = "any"
    keyphrases: tuple[str, ...] = ()
    case_ids: tuple[str, ...] = ()
    prefer_unique_docs: bool = False
    generation_top_k: int | None = None
    grounding_chunk_top_k: int | None = None
    max_pages_per_chunk: int | None = None

    @property
    def is_page_local(self) -> bool:
        return self.kind in {"title_page", "date_of_issue", "last_page", "judge_compare", "party_compare"}

    @property
    def is_compare(self) -> bool:
        return self.kind in {"judge_compare", "party_compare"}


def _is_compare_question(text: str) -> bool:
    return any(marker in text for marker in _COMPARE_MARKERS)


def _extract_case_ids(question_text: str) -> tuple[str, ...]:
    seen: set[str] = set()
    case_ids: list[str] = []
    for match in _CASE_ID_RE.finditer(question_text or ""):
        case_id = " ".join(match.group(0).upper().split())
        if case_id in seen:
            continue
        seen.add(case_id)
        case_ids.append(case_id)
    return tuple(case_ids)


def detect_grounding_intent(question_text: str, answer_type: str) -> GroundingIntent:
    text = (question_text or "").lower()
    answer_type = str(answer_type or "").lower()
    case_ids = _extract_case_ids(question_text)

    if any(marker in text for marker in ("title page", "cover page", "header/caption", "header", "caption")):
        return GroundingIntent(
            kind="title_page",
            page_focus="first",
            keyphrases=("claimant", "defendant", "applicant", "respondent", "claim no", "case no", "title", "caption"),
            case_ids=case_ids,
            prefer_unique_docs=True,
            generation_top_k=3,
            grounding_chunk_top_k=2,
            max_pages_per_chunk=2,
        )

    if any(marker in text for marker in ("date of issue", "issue date", "issued first", "earlier issue date")):
        return GroundingIntent(
            kind="date_of_issue",
            page_focus="first",
            keyphrases=("date of issue", "issued on", "issued", "date"),
            case_ids=case_ids,
            prefer_unique_docs=True,
            generation_top_k=3,
            grounding_chunk_top_k=2,
            max_pages_per_chunk=2,
        )

    if any(marker in text for marker in ("last page", "conclusion", "it is hereby ordered that")):
        return GroundingIntent(
            kind="last_page",
            page_focus="last",
            keyphrases=("it is hereby ordered that", "conclusion", "ordered that", "application", "dismissed", "refused", "allowed"),
            case_ids=case_ids,
            prefer_unique_docs=True,
            generation_top_k=3,
            grounding_chunk_top_k=2,
            max_pages_per_chunk=2,
        )

    if "judge" in text and _is_compare_question(text):
        return GroundingIntent(
            kind="judge_compare",
            page_focus="first",
            keyphrases=("before", "justice", "judge", "hearing"),
            case_ids=case_ids,
            prefer_unique_docs=True,
            generation_top_k=4,
            grounding_chunk_top_k=3,
            max_pages_per_chunk=2,
        )

    if answer_type in {"boolean", "name", "names"} and any(marker in text for marker in _PARTY_MARKERS) and _is_compare_question(text):
        return GroundingIntent(
            kind="party_compare",
            page_focus="first",
            keyphrases=("claimant", "defendant", "applicant", "respondent", "party", "parties"),
            case_ids=case_ids,
            prefer_unique_docs=True,
            generation_top_k=4,
            grounding_chunk_top_k=3,
            max_pages_per_chunk=2,
        )

    return GroundingIntent(kind="generic", case_ids=case_ids)


def _page_focus_bias(intent: GroundingIntent, pages: list[int], doc_max_page: int | None) -> float:
    if not pages or intent.page_focus == "any":
        return 0.0

    first_page = min(pages)
    last_page = max(pages)
    if intent.page_focus == "first":
        if first_page <= 1:
            return 3.5
        if first_page == 2:
            return 2.5
        if first_page == 3:
            return 1.0
        return 0.0

    if intent.page_focus == "last":
        doc_max_page = doc_max_page or last_page
        distance_from_end = max(0, doc_max_page - last_page)
        if distance_from_end == 0:
            return 3.5
        if distance_from_end == 1:
            return 2.5
        if distance_from_end == 2:
            return 1.0
    return 0.0


def score_chunk_for_intent(chunk: dict[str, Any], intent: GroundingIntent, doc_max_page: int | None = None) -> float:
    if intent.kind == "generic":
        return 0.0

    pages = sorted(
        int(page)
        for page in chunk.get("page_numbers", [])
        if isinstance(page, int) and page > 0
    )
    text_blob = " ".join(
        str(part or "")
        for part in (
            chunk.get("doc_title"),
            chunk.get("section_path"),
            str(chunk.get("text") or "")[:1200],
        )
    ).lower()

    score = _page_focus_bias(intent, pages, doc_max_page)
    phrase_hits = sum(1 for phrase in intent.keyphrases if phrase and phrase in text_blob)
    score += min(3, phrase_hits) * 1.25

    case_hits = 0
    upper_text_blob = text_blob.upper()
    for case_id in intent.case_ids:
        if case_id and case_id in upper_text_blob:
            case_hits += 1
    score += min(2, case_hits) * 0.8

    return score

