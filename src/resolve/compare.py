from __future__ import annotations

import re

from src.resolve.metadata_store import PageMetadataStore
from src.resolve.models import EvidencePage, Resolution
from src.retrieve.question_plan import QuestionPlan


def _normalize_compare_value(value: str, *, field: str) -> str:
    normalized = str(value or "").strip().casefold()
    normalized = re.sub(r"\s+", " ", normalized)
    if field == "parties" and ":" in normalized:
        normalized = normalized.split(":", 1)[1].strip()
    return normalized


def _collect_case_values(
    store: PageMetadataStore,
    case_id: str,
    *,
    field: str,
    page_hint: str,
) -> tuple[dict[str, str], dict[str, EvidencePage]]:
    values_by_key: dict[str, str] = {}
    evidence_by_key: dict[str, EvidencePage] = {}
    for record in store.get_case_records(case_id, page_hint=page_hint):
        for raw_value in record.get(field, []) or []:
            normalized = _normalize_compare_value(str(raw_value), field=field)
            if not normalized or normalized in values_by_key:
                continue
            values_by_key[normalized] = str(raw_value)
            evidence_by_key[normalized] = EvidencePage(
                doc_id=str(record.get("doc_id") or ""),
                page_num=int(record.get("page_num") or 0),
            )
    return values_by_key, evidence_by_key


def resolve_date_of_issue_compare(plan: QuestionPlan, store: PageMetadataStore) -> Resolution | None:
    if len(plan.case_ids) < 2:
        return None

    resolved_dates: list[tuple[str, str, EvidencePage]] = []
    for case_id in plan.case_ids[:2]:
        selected_record = None
        for record in store.get_case_records(case_id, page_hint=plan.page_hint):
            issue_date = str(record.get("issue_date") or "").strip()
            if not issue_date:
                continue
            selected_record = (issue_date, record)
            break
        if selected_record is None:
            return None
        issue_date, record = selected_record
        resolved_dates.append(
            (
                case_id,
                issue_date,
                EvidencePage(doc_id=str(record.get("doc_id") or ""), page_num=int(record.get("page_num") or 0)),
            )
        )

    if len(resolved_dates) < 2 or resolved_dates[0][1] == resolved_dates[1][1]:
        return None

    earlier_case_id, earlier_date, earlier_evidence = min(resolved_dates, key=lambda item: item[1])
    evidence_pages = [item[2] for item in resolved_dates]

    if plan.answer_type == "boolean":
        answer = resolved_dates[0][1] < resolved_dates[1][1]
    elif plan.answer_type == "date":
        answer = earlier_date
    else:
        answer = earlier_case_id

    return Resolution(
        answer=answer,
        evidence_pages=evidence_pages,
        confidence=0.99,
        method="date_of_issue_compare",
        facts={"dates": {case_id: issue_date for case_id, issue_date, _evidence in resolved_dates}},
    )


def resolve_judge_compare(plan: QuestionPlan, store: PageMetadataStore) -> Resolution | None:
    return _resolve_overlap_compare(plan, store, field="judges", method="judge_compare")


def resolve_party_compare(plan: QuestionPlan, store: PageMetadataStore) -> Resolution | None:
    return _resolve_overlap_compare(plan, store, field="parties", method="party_compare")


def _resolve_overlap_compare(
    plan: QuestionPlan,
    store: PageMetadataStore,
    *,
    field: str,
    method: str,
) -> Resolution | None:
    if len(plan.case_ids) < 2:
        return None

    collected: list[tuple[dict[str, str], dict[str, EvidencePage]]] = []
    for case_id in plan.case_ids[:2]:
        values_by_key, evidence_by_key = _collect_case_values(store, case_id, field=field, page_hint=plan.page_hint)
        if not values_by_key:
            return None
        collected.append((values_by_key, evidence_by_key))

    overlap_keys = sorted(set(collected[0][0]) & set(collected[1][0]))
    evidence_pages: list[EvidencePage] = []
    for values_by_key, evidence_by_key in collected:
        chosen_key = overlap_keys[0] if overlap_keys else next(iter(values_by_key.keys()))
        evidence_pages.append(evidence_by_key[chosen_key])

    if plan.answer_type == "boolean":
        answer = bool(overlap_keys)
    elif plan.answer_type == "names":
        answer = [collected[0][0][key] for key in overlap_keys]
    elif plan.answer_type == "name" and overlap_keys:
        answer = collected[0][0][overlap_keys[0]]
    else:
        return None

    return Resolution(
        answer=answer,
        evidence_pages=evidence_pages,
        confidence=0.99,
        method=method,
        facts={"overlap": [collected[0][0][key] for key in overlap_keys]},
    )
