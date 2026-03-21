from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from src.retrieve.grounding_utils import extract_question_anchors


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
    max_pages_per_doc: int | None = None
    max_docs: int | None = None
    max_total_pages: int | None = None
    article_refs: tuple[str, ...] = ()
    law_number: str | None = None
    quoted_sections: tuple[str, ...] = ()

    @property
    def is_page_local(self) -> bool:
        return self.kind in {"title_page", "date_of_issue", "last_page", "judge_compare", "party_compare"}

    @property
    def is_compare(self) -> bool:
        return self.kind in {"judge_compare", "party_compare"}



def detect_grounding_intent(question_text: str, answer_type: str) -> GroundingIntent:
    text = (question_text or "").lower()
    answer_type = str(answer_type or "").lower()
    anchors = extract_question_anchors(question_text)
    case_ids = anchors.case_ids

    compare_markers = (
        "both case",
        "both cases",
        "across all documents",
        "across case",
        "common to both",
        "in both cases",
        "appeared in both",
        "involve any of the same",
    )
    party_markers = ("party", "parties", "claimant", "defendant", "main party", "applicant", "respondent")
    title_page_markers = ("title page", "cover page", "header/caption", "header", "caption")
    is_compare_question = any(marker in text for marker in compare_markers)
    if not is_compare_question and len(case_ids) >= 2:
        is_compare_question = "in common" in text or re.search(r"\b(?:to|in)\s+both\b", text) is not None
    asks_for_case_file_coverage = any(
        marker in text
        for marker in (
            "all documents",
            "every document",
            "each document",
            "full case files",
            "case files",
        )
    )

    if any(marker in text for marker in title_page_markers):
        if len(case_ids) >= 2 and is_compare_question:
            if "judge" in text:
                return GroundingIntent(
                    kind="judge_compare",
                    page_focus="front",
                    keyphrases=("before", "justice", "judge", "hearing"),
                    case_ids=case_ids,
                    prefer_unique_docs=True,
                    generation_top_k=4,
                    grounding_chunk_top_k=4,
                    max_pages_per_chunk=2,
                    max_pages_per_doc=1,
                    max_docs=6,
                    max_total_pages=6,
                    quoted_sections=anchors.quoted_sections,
                )
            if answer_type in {"boolean", "name", "names"} and any(marker in text for marker in party_markers):
                return GroundingIntent(
                    kind="party_compare",
                    page_focus="first",
                    keyphrases=("claimant", "defendant", "applicant", "respondent", "party", "parties"),
                    case_ids=case_ids,
                    prefer_unique_docs=True,
                    generation_top_k=4,
                    grounding_chunk_top_k=4,
                    max_pages_per_chunk=2,
                    max_pages_per_doc=1,
                    max_docs=6,
                    max_total_pages=6,
                    quoted_sections=anchors.quoted_sections,
                )

        max_docs = None
        max_total_pages = None
        if asks_for_case_file_coverage:
            max_docs = 4 if len(case_ids) >= 2 else 3
            max_total_pages = max_docs
        return GroundingIntent(
            kind="title_page",
            page_focus="front",
            keyphrases=("claimant", "defendant", "applicant", "respondent", "claim no", "case no", "title", "caption"),
            case_ids=case_ids,
            prefer_unique_docs=True,
            generation_top_k=3,
            grounding_chunk_top_k=3 if asks_for_case_file_coverage else 2,
            max_pages_per_chunk=2,
            max_pages_per_doc=1,
            max_docs=max_docs,
            max_total_pages=max_total_pages,
            quoted_sections=anchors.quoted_sections,
        )

    if anchors.article_refs and (anchors.law_number or len(anchors.law_title_mentions) == 1):
        keyphrases = list(anchors.article_refs[:2])
        if anchors.law_number:
            keyphrases.append(anchors.law_number)
        elif anchors.law_title_mentions:
            keyphrases.append(anchors.law_title_mentions[0])
        return GroundingIntent(
            kind="article_ref",
            page_focus="any",
            keyphrases=tuple(keyphrases),
            case_ids=case_ids,
            grounding_chunk_top_k=2,
            max_pages_per_chunk=2,
            max_pages_per_doc=2,
            article_refs=anchors.article_refs,
            law_number=anchors.law_number,
            quoted_sections=anchors.quoted_sections,
        )

    if any(marker in text for marker in ("date of issue", "issue date", "issued first", "earlier issue date")):
        return GroundingIntent(
            kind="date_of_issue",
            page_focus="front",
            keyphrases=("date of issue", "issued on", "issued", "date"),
            case_ids=case_ids,
            prefer_unique_docs=True,
            generation_top_k=3,
            grounding_chunk_top_k=2,
            max_pages_per_chunk=2,
            max_pages_per_doc=1,
            quoted_sections=anchors.quoted_sections,
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
            max_pages_per_doc=2,
            quoted_sections=anchors.quoted_sections,
        )

    if "judge" in text and is_compare_question:
        return GroundingIntent(
            kind="judge_compare",
            page_focus="front",
            keyphrases=("before", "justice", "judge", "hearing"),
            case_ids=case_ids,
            prefer_unique_docs=True,
            generation_top_k=4,
            grounding_chunk_top_k=4 if asks_for_case_file_coverage else 3,
            max_pages_per_chunk=2,
            max_pages_per_doc=1,
            max_docs=6 if asks_for_case_file_coverage else 4,
            max_total_pages=6 if asks_for_case_file_coverage else 4,
            quoted_sections=anchors.quoted_sections,
        )

    if answer_type in {"boolean", "name", "names"} and any(marker in text for marker in party_markers) and is_compare_question:
        return GroundingIntent(
            kind="party_compare",
            page_focus="first",
            keyphrases=("claimant", "defendant", "applicant", "respondent", "party", "parties"),
            case_ids=case_ids,
            prefer_unique_docs=True,
            generation_top_k=4,
            grounding_chunk_top_k=4 if asks_for_case_file_coverage else 3,
            max_pages_per_chunk=2,
            max_pages_per_doc=1,
            max_docs=6 if asks_for_case_file_coverage else 4,
            max_total_pages=6 if asks_for_case_file_coverage else 4,
            quoted_sections=anchors.quoted_sections,
        )

    return GroundingIntent(kind="generic", case_ids=case_ids, max_pages_per_doc=2, quoted_sections=anchors.quoted_sections)



def _page_focus_bias(intent: GroundingIntent, pages: list[int], doc_max_page: int | None) -> float:
    if not pages or intent.page_focus == "any":
        return 0.0

    first_page = min(pages)
    last_page = max(pages)
    if intent.page_focus == "front":
        if first_page <= 1:
            return 3.5
        if first_page == 2:
            return 3.0
        if first_page == 3:
            return 2.0
        if first_page == 4:
            return 1.0
        return 0.0

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
