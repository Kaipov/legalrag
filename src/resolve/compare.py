from __future__ import annotations

import re

from src.resolve.issue_date import select_best_issue_date_record
from src.resolve.metadata_store import PageMetadataStore
from src.resolve.models import EvidencePage, Resolution
from src.retrieve.question_plan import QuestionPlan

_MONEY_RE = re.compile(r"\b(?:USD|AED)\s*([0-9][0-9,]*(?:\.\d+)?)\b", re.IGNORECASE)
_CLAIM_AMOUNT_SIGNAL_RE = re.compile(
    r"\b(?:"
    r"seeking payment|"
    r"claim against|"
    r"outstanding claim|"
    r"claim was for|"
    r"claim is for|"
    r"claims? (?:he is owed|that|for)|"
    r"judgment sum"
    r")\b",
    re.IGNORECASE,
)


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
) -> tuple[dict[str, str], dict[str, EvidencePage], list[EvidencePage]]:
    values_by_key: dict[str, str] = {}
    evidence_by_key: dict[str, EvidencePage] = {}
    coverage_pages: list[EvidencePage] = []
    seen_page_keys: set[tuple[str, int]] = set()
    for record in store.get_case_records(case_id, page_hint=page_hint):
        evidence_page = EvidencePage(
            doc_id=str(record.get("doc_id") or ""),
            page_num=int(record.get("page_num") or 0),
        )
        page_key = (evidence_page.doc_id, evidence_page.page_num)
        if evidence_page.doc_id and evidence_page.page_num > 0 and page_key not in seen_page_keys:
            seen_page_keys.add(page_key)
            coverage_pages.append(evidence_page)
        for raw_value in record.get(field, []) or []:
            normalized = _normalize_compare_value(str(raw_value), field=field)
            if not normalized or normalized in values_by_key:
                continue
            values_by_key[normalized] = str(raw_value)
            evidence_by_key[normalized] = evidence_page
    return values_by_key, evidence_by_key, coverage_pages


def _as_number(value: object) -> float | None:
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _extract_claim_amount(record: dict) -> float | None:
    text = str(record.get("text") or "")
    money_matches = [(_as_number(match.group(1)), match.start()) for match in _MONEY_RE.finditer(text)]
    money_matches = [(value, position) for value, position in money_matches if value is not None]
    signal_positions = [match.start() for match in _CLAIM_AMOUNT_SIGNAL_RE.finditer(text)]

    if money_matches and signal_positions:
        best_value, _best_position = min(
            money_matches,
            key=lambda item: min(abs(item[1] - signal_position) for signal_position in signal_positions),
        )
        return best_value

    fallback_values = [_as_number(value) for value in record.get("money_values", []) or []]
    fallback_values = [value for value in fallback_values if value is not None]
    if not fallback_values:
        return None
    return fallback_values[0] if len(fallback_values) == 1 else max(fallback_values)


def resolve_date_of_issue_compare(plan: QuestionPlan, store: PageMetadataStore) -> Resolution | None:
    if len(plan.case_ids) < 2:
        return None

    resolved_dates: list[tuple[str, str, EvidencePage]] = []
    for case_id in plan.case_ids[:2]:
        selected_record = select_best_issue_date_record(
            store.get_case_records(case_id, page_hint=plan.page_hint)
        )
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

    earlier_case_id, earlier_date, _earlier_evidence = min(resolved_dates, key=lambda item: item[1])
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


def resolve_monetary_claim_compare(plan: QuestionPlan, store: PageMetadataStore) -> Resolution | None:
    if len(plan.case_ids) < 2:
        return None

    resolved_amounts: list[tuple[str, float, EvidencePage]] = []
    for case_id in plan.case_ids[:2]:
        selected_record = None
        for record in store.get_case_records(case_id, page_hint=plan.page_hint):
            claim_amount = _extract_claim_amount(record)
            if claim_amount is None:
                continue
            selected_record = (claim_amount, record)
            break
        if selected_record is None:
            return None
        claim_amount, record = selected_record
        resolved_amounts.append(
            (
                case_id,
                claim_amount,
                EvidencePage(doc_id=str(record.get("doc_id") or ""), page_num=int(record.get("page_num") or 0)),
            )
        )

    if len(resolved_amounts) < 2 or resolved_amounts[0][1] == resolved_amounts[1][1]:
        return None

    higher_case_id, higher_amount, _higher_evidence = max(resolved_amounts, key=lambda item: item[1])
    evidence_pages = [item[2] for item in resolved_amounts]

    if plan.answer_type == "boolean":
        answer = resolved_amounts[0][1] > resolved_amounts[1][1]
    elif plan.answer_type == "number":
        answer = int(higher_amount) if float(higher_amount).is_integer() else higher_amount
    else:
        answer = higher_case_id

    return Resolution(
        answer=answer,
        evidence_pages=evidence_pages,
        confidence=0.99,
        method="monetary_claim_compare",
        facts={"amounts": {case_id: amount for case_id, amount, _evidence in resolved_amounts}},
    )


def _resolve_overlap_compare(
    plan: QuestionPlan,
    store: PageMetadataStore,
    *,
    field: str,
    method: str,
) -> Resolution | None:
    if len(plan.case_ids) < 2:
        return None

    collected: list[tuple[dict[str, str], dict[str, EvidencePage], list[EvidencePage]]] = []
    for case_id in plan.case_ids[:2]:
        values_by_key, evidence_by_key, coverage_pages = _collect_case_values(
            store,
            case_id,
            field=field,
            page_hint=plan.page_hint,
        )
        if not values_by_key:
            return None
        collected.append((values_by_key, evidence_by_key, coverage_pages))

    overlap_keys = sorted(set(collected[0][0]) & set(collected[1][0]))
    evidence_pages: list[EvidencePage] = []
    seen_page_keys: set[tuple[str, int]] = set()

    def append_page(page: EvidencePage) -> None:
        page_key = (page.doc_id, page.page_num)
        if not page.doc_id or page.page_num <= 0 or page_key in seen_page_keys:
            return
        seen_page_keys.add(page_key)
        evidence_pages.append(page)

    if plan.answer_type == "boolean":
        answer = bool(overlap_keys)
        if overlap_keys:
            for _values_by_key, evidence_by_key, _coverage_pages in collected:
                for overlap_key in overlap_keys:
                    evidence_page = evidence_by_key.get(overlap_key)
                    if evidence_page is not None:
                        append_page(evidence_page)
        else:
            for _values_by_key, _evidence_by_key, coverage_pages in collected:
                for evidence_page in coverage_pages:
                    append_page(evidence_page)
    elif plan.answer_type == "names":
        answer = [collected[0][0][key] for key in overlap_keys]
        for _values_by_key, evidence_by_key, _coverage_pages in collected:
            for overlap_key in overlap_keys:
                evidence_page = evidence_by_key.get(overlap_key)
                if evidence_page is not None:
                    append_page(evidence_page)
    elif plan.answer_type == "name" and overlap_keys:
        answer = collected[0][0][overlap_keys[0]]
        for _values_by_key, evidence_by_key, _coverage_pages in collected:
            evidence_page = evidence_by_key.get(overlap_keys[0])
            if evidence_page is not None:
                append_page(evidence_page)
    else:
        return None

    return Resolution(
        answer=answer,
        evidence_pages=evidence_pages,
        confidence=0.99,
        method=method,
        facts={"overlap": [collected[0][0][key] for key in overlap_keys]},
    )
