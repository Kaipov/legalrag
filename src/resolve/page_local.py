from __future__ import annotations

import re

from src.generate.verbalize import verbalize_field_answer
from src.resolve.metadata_store import PageMetadataStore
from src.resolve.models import EvidencePage, Resolution
from src.retrieve.question_plan import QuestionPlan

_LAW_NUMBER_RE = re.compile(r"\bDIFC\s+Law\s+No\.?\s+([0-9]+\s+of\s+[0-9]{4})\b", re.IGNORECASE)
_PARTY_PREFIX_RE = re.compile(r"^(Claimant|Defendant|Applicant|Respondent|Appellant|Petitioner|Plaintiff):\s*", re.IGNORECASE)



def _extract_law_number(record: dict) -> str | None:
    for source in (record.get("doc_title"), record.get("text")):
        source_text = str(source or "")
        match = _LAW_NUMBER_RE.search(source_text)
        if match:
            return f"DIFC Law No. {match.group(1)}"
    return None



def _strip_party_prefix(value: str) -> str:
    return _PARTY_PREFIX_RE.sub("", str(value or "")).strip()



def _filter_party_values(values: list[str], question_text: str) -> list[str]:
    lowered_question = str(question_text or "").lower()
    if "claimant" in lowered_question:
        filtered = [value for value in values if value.lower().startswith("claimant:")]
        return filtered or values
    if "defendant" in lowered_question:
        filtered = [value for value in values if value.lower().startswith("defendant:")]
        return filtered or values
    if "applicant" in lowered_question:
        filtered = [value for value in values if value.lower().startswith("applicant:")]
        return filtered or values
    if "respondent" in lowered_question:
        filtered = [value for value in values if value.lower().startswith("respondent:")]
        return filtered or values
    return values



def resolve_page_local_lookup(plan: QuestionPlan, store: PageMetadataStore, question_text: str) -> Resolution | None:
    if not plan.case_ids:
        return None

    case_id = plan.case_ids[0]
    for record in store.get_case_records(case_id, page_hint=plan.page_hint):
        value = _extract_value(record, plan, question_text=question_text)
        if value is None:
            continue
        evidence = EvidencePage(doc_id=str(record.get("doc_id") or ""), page_num=int(record.get("page_num") or 0))
        answer = _shape_answer(value, plan, question_text)
        if answer is None:
            continue
        return Resolution(
            answer=answer,
            evidence_pages=[evidence],
            confidence=0.99,
            method=plan.mode,
            facts={"case_id": case_id, "target_field": plan.target_field, "value": value},
        )
    return None



def _extract_value(record: dict, plan: QuestionPlan, *, question_text: str):
    target_field = plan.target_field
    if target_field == "claim_number":
        values = list(record.get("claim_numbers") or [])
        return values[0] if values else None
    if target_field == "issue_date":
        return record.get("issue_date")
    if target_field == "judge":
        values = list(record.get("judges") or [])
        return values[0] if values else None
    if target_field == "party":
        values = [str(value) for value in list(record.get("parties") or []) if str(value).strip()]
        if not values:
            return None
        values = _filter_party_values(values, question_text)
        if plan.answer_type == "names":
            return values
        return values[0]
    if target_field == "law_number":
        return _extract_law_number(record)
    return None



def _shape_answer(value, plan: QuestionPlan, question_text: str):
    if plan.answer_type == "name":
        if isinstance(value, list):
            value = value[0] if value else None
        if value is None:
            return None
        if plan.target_field == "party":
            return _strip_party_prefix(str(value))
        return str(value)
    if plan.answer_type == "names":
        if isinstance(value, list):
            rendered = [str(item) for item in value if str(item).strip()]
        else:
            rendered = [str(value)]
        if plan.target_field == "party":
            rendered = [_strip_party_prefix(item) for item in rendered if _strip_party_prefix(item)]
        return rendered
    if plan.answer_type == "date":
        return str(value)
    if plan.answer_type == "free_text":
        if isinstance(value, list):
            rendered = "; ".join(str(item) for item in value)
        else:
            rendered = value
        return verbalize_field_answer(plan.target_field, rendered, question_text=question_text)
    return None
