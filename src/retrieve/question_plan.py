from __future__ import annotations

from dataclasses import dataclass
import re

from src.case_ids import extract_case_ids as _extract_case_id_list
from src.generate.null_detect import check_foreign_concepts

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
_FIRST_PAGE_MARKERS = ("first page", "page 1")
_MONETARY_COMPARE_MARKERS = (
    "higher monetary claim",
    "higher claim",
    "higher amount",
    "greater monetary claim",
    "larger sum claimed",
    "larger claim amount",
    "larger claimed sum",
    "higher sum claimed",
)
_DATE_ISSUE_MARKERS = (
    "date of issue",
    "issue date",
    "issued first",
    "issued earlier",
    "issued later",
    "earlier issue date",
    "later issue date",
    "issue its document",
    "issue the document",
)
_JUDGE_TIMELINE_MARKERS = (
    "across its various stages",
    "across the various stages",
    "at any point during the proceedings",
    "during the proceedings",
    "throughout the proceedings",
    "panel of judges change",
    "changes to the judges presiding",
)
_LAW_BASIS_MARKERS = (
    "existing law is the basis",
    "basis for the amendments proposed",
)
_ARTICLE_REF_RE = re.compile(r"\bArticle\s+\d+[A-Z]?(?:\(\d+[A-Z]?\)|\([a-z]\))*", re.IGNORECASE)


@dataclass(frozen=True)
class QuestionPlan:
    mode: str
    answer_type: str
    case_ids: tuple[str, ...] = ()
    article_refs: tuple[str, ...] = ()
    page_hint: str = "any"
    compare_op: str = "none"
    target_field: str | None = None
    abstention_risk: bool = False

    @property
    def is_deterministic_candidate(self) -> bool:
        return self.mode in {
            "date_of_issue_compare",
            "judge_compare",
            "monetary_claim_compare",
            "party_compare",
            "page_local_lookup",
            "title_page_metadata",
            "judge_timeline_change",
            "last_page_outcome",
        }

def _extract_case_ids(question_text: str) -> tuple[str, ...]:
    return tuple(_extract_case_id_list(question_text))



def _extract_article_refs(question_text: str) -> tuple[str, ...]:
    seen: set[str] = set()
    values: list[str] = []
    for match in _ARTICLE_REF_RE.finditer(question_text or ""):
        value = " ".join(match.group(0).split())
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    return tuple(values)



def _is_compare_question(text: str, case_ids: tuple[str, ...] = ()) -> bool:
    if any(marker in text for marker in _COMPARE_MARKERS):
        return True
    if len(case_ids) < 2:
        return False
    if "in common" in text:
        return True
    if re.search(r"\b(?:to|in)\s+both\b", text):
        return True
    return False



def _infer_target_field(text: str) -> str | None:
    if any(marker in text for marker in ("claim number", "claim no", "claim no.", "claim number did the appeal originate")):
        return "claim_number"
    if any(marker in text for marker in _MONETARY_COMPARE_MARKERS):
        return "money_value"
    if any(marker in text for marker in ("claim value", "claim amount", "value in aed")):
        return "money_value"
    if any(marker in text for marker in _DATE_ISSUE_MARKERS):
        return "issue_date"
    if "judge" in text:
        return "judge"
    if any(marker in text for marker in _PARTY_MARKERS):
        return "party"
    if any(marker in text for marker in ("law number", "official law number")):
        return "law_number"
    return None


def _is_judge_timeline_question(text: str) -> bool:
    if "judge" not in text:
        return False
    return any(marker in text for marker in _JUDGE_TIMELINE_MARKERS)


def _asks_for_existing_law_basis(text: str) -> bool:
    return "existing law" in text and any(marker in text for marker in _LAW_BASIS_MARKERS)



def build_question_plan(question_text: str, answer_type: str) -> QuestionPlan:
    text = str(question_text or "").lower()
    normalized_answer_type = str(answer_type or "free_text").lower()
    case_ids = _extract_case_ids(question_text)
    article_refs = _extract_article_refs(question_text)
    target_field = _infer_target_field(text)
    abstention_risk = check_foreign_concepts(question_text)
    title_page_markers = ("title page", "cover page", "header/caption", "header", "caption")

    if abstention_risk:
        return QuestionPlan(
            mode="absence_check",
            answer_type=normalized_answer_type,
            case_ids=case_ids,
            article_refs=article_refs,
            target_field=target_field,
            abstention_risk=True,
        )

    if any(marker in text for marker in ("page 2", "second page")) and case_ids:
        return QuestionPlan(
            mode="page_local_lookup",
            answer_type=normalized_answer_type,
            case_ids=case_ids,
            article_refs=article_refs,
            page_hint="page_2",
            target_field=target_field,
        )

    if len(case_ids) == 1 and any(marker in text for marker in _FIRST_PAGE_MARKERS) and target_field in {
        "claim_number",
        "judge",
        "party",
        "law_number",
    }:
        return QuestionPlan(
            mode="page_local_lookup",
            answer_type=normalized_answer_type,
            case_ids=case_ids,
            article_refs=article_refs,
            page_hint="first",
            target_field=target_field,
        )

    if len(case_ids) == 1 and target_field == "party" and normalized_answer_type in {"name", "names", "number"}:
        return QuestionPlan(
            mode="page_local_lookup",
            answer_type=normalized_answer_type,
            case_ids=case_ids,
            article_refs=article_refs,
            page_hint="first",
            target_field="party",
        )

    if len(case_ids) == 1 and target_field == "judge" and _is_judge_timeline_question(text):
        return QuestionPlan(
            mode="judge_timeline_change",
            answer_type=normalized_answer_type,
            case_ids=case_ids,
            article_refs=article_refs,
            page_hint="front",
            target_field="judge",
        )

    if len(case_ids) >= 2 and any(marker in text for marker in title_page_markers) and _is_compare_question(text, case_ids):
        if target_field == "judge":
            return QuestionPlan(
                mode="judge_compare",
                answer_type=normalized_answer_type,
                case_ids=case_ids,
                article_refs=article_refs,
                page_hint="front",
                compare_op="set_overlap",
                target_field="judge",
            )
        if normalized_answer_type in {"boolean", "name", "names"} and target_field == "party":
            return QuestionPlan(
                mode="party_compare",
                answer_type=normalized_answer_type,
                case_ids=case_ids,
                article_refs=article_refs,
                page_hint="first",
                compare_op="set_overlap",
                target_field="party",
            )

    if any(marker in text for marker in title_page_markers) and (
        case_ids or target_field == "law_number"
    ):
        return QuestionPlan(
            mode="title_page_metadata",
            answer_type=normalized_answer_type,
            case_ids=case_ids,
            article_refs=article_refs,
            page_hint="first",
            target_field=target_field,
        )

    if not case_ids and normalized_answer_type in {"name", "number"} and _asks_for_existing_law_basis(text):
        return QuestionPlan(
            mode="title_page_metadata",
            answer_type=normalized_answer_type,
            case_ids=case_ids,
            article_refs=article_refs,
            page_hint="first",
            target_field="law_number",
        )

    if len(case_ids) >= 2 and any(marker in text for marker in _MONETARY_COMPARE_MARKERS):
        return QuestionPlan(
            mode="monetary_claim_compare",
            answer_type=normalized_answer_type,
            case_ids=case_ids,
            article_refs=article_refs,
            page_hint="page_2",
            compare_op="max_number",
            target_field="money_value",
        )

    if len(case_ids) >= 2 and any(marker in text for marker in _DATE_ISSUE_MARKERS):
        return QuestionPlan(
            mode="date_of_issue_compare",
            answer_type=normalized_answer_type,
            case_ids=case_ids,
            article_refs=article_refs,
            page_hint="front",
            compare_op="min_date",
            target_field="issue_date",
        )

    if len(case_ids) == 1 and any(marker in text for marker in _DATE_ISSUE_MARKERS):
        return QuestionPlan(
            mode="page_local_lookup",
            answer_type=normalized_answer_type,
            case_ids=case_ids,
            article_refs=article_refs,
            page_hint="front",
            target_field="issue_date",
        )

    if len(case_ids) == 1 and target_field == "money_value":
        return QuestionPlan(
            mode="page_local_lookup",
            answer_type=normalized_answer_type,
            case_ids=case_ids,
            article_refs=article_refs,
            page_hint="any",
            target_field="money_value",
        )

    if "judge" in text and _is_compare_question(text, case_ids):
        return QuestionPlan(
            mode="judge_compare",
            answer_type=normalized_answer_type,
            case_ids=case_ids,
            article_refs=article_refs,
            page_hint="front",
            compare_op="set_overlap",
            target_field="judge",
        )

    if normalized_answer_type in {"boolean", "name", "names"} and any(marker in text for marker in _PARTY_MARKERS) and _is_compare_question(text, case_ids):
        return QuestionPlan(
            mode="party_compare",
            answer_type=normalized_answer_type,
            case_ids=case_ids,
            article_refs=article_refs,
            page_hint="first",
            compare_op="set_overlap",
            target_field="party",
        )

    if any(marker in text for marker in ("last page", "conclusion", "it is hereby ordered that")):
        return QuestionPlan(
            mode="last_page_outcome",
            answer_type=normalized_answer_type,
            case_ids=case_ids,
            article_refs=article_refs,
            page_hint="last",
            target_field="outcome",
        )

    if article_refs:
        return QuestionPlan(
            mode="article_lookup",
            answer_type=normalized_answer_type,
            case_ids=case_ids,
            article_refs=article_refs,
            target_field=target_field,
        )

    return QuestionPlan(
        mode="generic_synthesis",
        answer_type=normalized_answer_type,
        case_ids=case_ids,
        article_refs=article_refs,
        target_field=target_field,
    )
