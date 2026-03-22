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
_JUDGE_CONTEXT_CUTOFF_RE = re.compile(r"\b(?:dated|and upon|upon)\b.*$", re.IGNORECASE)
_PARTY_SIGNAL_RE = re.compile(r"\b(?:claimant|defendant|applicant|respondent|party|parties)\b", re.IGNORECASE)
_JUDGE_SIGNAL_RE = re.compile(r"\b(?:before|justice|judge|order with reasons of|judgment of|amended judgment of)\b", re.IGNORECASE)
_JUDGE_TITLE_RE = re.compile(r"\b(?:chief justice|deputy chief justice|justice|judge)\b", re.IGNORECASE)


def _normalize_compare_value(value: str, *, field: str) -> str:
    normalized = str(value or "").strip().casefold()
    normalized = re.sub(r"\s+", " ", normalized)
    if field == "parties" and ":" in normalized:
        normalized = normalized.split(":", 1)[1].strip()
    if field == "judges":
        normalized = re.sub(r"\bh\.?\s*e\.?\b", "", normalized, flags=re.IGNORECASE)
        title_match = _JUDGE_TITLE_RE.search(normalized)
        if title_match is not None:
            normalized = normalized[title_match.start() :]
        normalized = re.sub(r"\s*\(.*$", "", normalized)
        normalized = re.sub(
            r"\b(?:dated|and upon|upon|for(?: the)?|at(?: the)?|on)\b.*$",
            "",
            normalized,
            flags=re.IGNORECASE,
        )
        normalized = re.sub(r"\bof\s+\d{1,2}\s+[a-z]+\s+\d{4}\b.*$", "", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\b(?:chief justice|deputy chief justice|justice|judge)\b", "", normalized)
        normalized = _JUDGE_CONTEXT_CUTOFF_RE.sub("", normalized)
    normalized = re.sub(r"^[^a-z0-9]+", "", normalized)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _collect_case_values(
    store: PageMetadataStore,
    case_id: str,
    *,
    field: str,
    page_hint: str,
) -> tuple[dict[str, str], dict[str, EvidencePage], dict[str, tuple[int, int]], list[EvidencePage]]:
    values_by_key: dict[str, str] = {}
    evidence_by_key: dict[str, EvidencePage] = {}
    evidence_rank_by_key: dict[str, tuple[int, int]] = {}
    records = store.get_case_records(case_id, page_hint=page_hint)
    coverage_pages = _select_compare_coverage_pages(records, field=field)
    for record in records:
        evidence_page = EvidencePage(
            doc_id=str(record.get("doc_id") or ""),
            page_num=int(record.get("page_num") or 0),
        )
        for raw_value in record.get(field, []) or []:
            normalized = _normalize_compare_value(str(raw_value), field=field)
            if not normalized:
                continue
            evidence_rank = _compare_key_rank(record, field=field, normalized_key=normalized)
            current_rank = evidence_rank_by_key.get(normalized)
            if current_rank is not None and evidence_rank <= current_rank:
                continue
            values_by_key[normalized] = str(raw_value)
            evidence_by_key[normalized] = evidence_page
            evidence_rank_by_key[normalized] = evidence_rank
    return values_by_key, evidence_by_key, evidence_rank_by_key, coverage_pages


def _compare_key_rank(record: dict, *, field: str, normalized_key: str) -> tuple[int, int, int]:
    base_score, page_tiebreak = _compare_record_rank(record, field=field)
    values = sorted(
        {
            _normalize_compare_value(str(raw_value), field=field)
            for raw_value in list(record.get(field, []) or [])
            if _normalize_compare_value(str(raw_value), field=field)
        }
    )
    unrelated_value_count = sum(1 for value in values if value != normalized_key)
    score = base_score

    if field == "judges":
        header_text = str(record.get("text") or "").lower()[:400]
        if "before" in header_text:
            score += 6
        score -= unrelated_value_count * 20

    return score, -unrelated_value_count, page_tiebreak


def _compare_record_rank(record: dict, *, field: str) -> tuple[int, int]:
    page_num = int(record.get("page_num") or 0)
    text = str(record.get("text") or "").lower()
    values = sorted(
        {
        _normalize_compare_value(str(raw_value), field=field)
        for raw_value in list(record.get(field, []) or [])
        if _normalize_compare_value(str(raw_value), field=field)
        }
    )

    score = 0
    if field == "judges":
        score += len(values) * 20
        score += min(len(_JUDGE_SIGNAL_RE.findall(text)), 2) * 5
    else:
        score += len(values) * 12
        score += min(len(_PARTY_SIGNAL_RE.findall(text)), 3) * 3

    if field == "judges":
        if page_num == 1:
            score += 12
        elif page_num == 2:
            score += 10
        elif page_num == 3:
            score += 4
    else:
        if page_num == 1:
            score += 8
        elif page_num == 2:
            score += 6
        elif page_num == 3:
            score += 3

    return score, -(page_num if page_num > 0 else 999)


def _select_compare_coverage_pages(records: list[dict], *, field: str) -> list[EvidencePage]:
    records_by_doc: dict[str, list[dict]] = {}
    for record in records:
        doc_id = str(record.get("doc_id") or "").strip()
        if not doc_id:
            continue
        records_by_doc.setdefault(doc_id, []).append(record)

    selected_pages: list[EvidencePage] = []
    seen_pages: set[tuple[str, int]] = set()
    for doc_id, doc_records in sorted(records_by_doc.items()):
        sorted_records = sorted(doc_records, key=lambda item: int(item.get("page_num") or 0))
        selected_doc_records: list[dict] = []
        covered_values: set[str] = set()
        for record in sorted_records:
            normalized_values = {
                _normalize_compare_value(str(raw_value), field=field)
                for raw_value in list(record.get(field, []) or [])
            }
            normalized_values = {value for value in normalized_values if value}
            if not normalized_values:
                continue
            if normalized_values - covered_values:
                selected_doc_records.append(record)
                covered_values.update(normalized_values)

        if not selected_doc_records:
            best_record = max(sorted_records, key=lambda item: _compare_record_rank(item, field=field))
            best_rank = _compare_record_rank(best_record, field=field)
            selected_doc_records = [best_record if best_rank[0] > 0 else sorted_records[0]]

        for record in selected_doc_records:
            page_num = int(record.get("page_num") or 0)
            page_key = (doc_id, page_num)
            if page_num <= 0 or page_key in seen_pages:
                continue
            seen_pages.add(page_key)
            selected_pages.append(EvidencePage(doc_id=doc_id, page_num=page_num))

    return selected_pages


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


def _iter_case_records_with_fallback(
    store: PageMetadataStore,
    case_id: str,
    *,
    primary_page_hint: str,
):
    seen_page_keys: set[tuple[str, int]] = set()
    for page_hint in _dedupe_preserve_order([primary_page_hint, "front", "any"]):
        for record in store.get_case_records(case_id, page_hint=page_hint):
            page_key = (str(record.get("doc_id") or ""), int(record.get("page_num") or 0))
            if page_key in seen_page_keys:
                continue
            seen_page_keys.add(page_key)
            yield record


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _append_unique_page(
    evidence_pages: list[EvidencePage],
    seen_page_keys: set[tuple[str, int]],
    page: EvidencePage,
) -> None:
    page_key = (page.doc_id, page.page_num)
    if not page.doc_id or page.page_num <= 0 or page_key in seen_page_keys:
        return
    seen_page_keys.add(page_key)
    evidence_pages.append(page)


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


def resolve_judge_timeline_change(plan: QuestionPlan, store: PageMetadataStore) -> Resolution | None:
    if len(plan.case_ids) != 1:
        return None

    case_id = plan.case_ids[0]
    panel_records: dict[tuple[str, ...], tuple[list[str], EvidencePage, tuple[int, int]]] = {}
    records = store.get_case_records(
        case_id,
        page_hint="front" if plan.page_hint in {"first", "front"} else plan.page_hint,
    )
    for record in records:
        raw_values_by_key: dict[str, str] = {}
        for raw_value in record.get("judges", []) or []:
            normalized = _normalize_compare_value(str(raw_value), field="judges")
            if not normalized:
                continue
            raw_values_by_key.setdefault(normalized, str(raw_value))
        if not raw_values_by_key:
            continue

        panel_key = tuple(sorted(raw_values_by_key))
        panel_values = [raw_values_by_key[key] for key in panel_key]
        evidence_page = EvidencePage(
            doc_id=str(record.get("doc_id") or ""),
            page_num=int(record.get("page_num") or 0),
        )
        panel_rank = _compare_record_rank(record, field="judges")
        current = panel_records.get(panel_key)
        if current is not None and panel_rank <= current[2]:
            continue
        panel_records[panel_key] = (panel_values, evidence_page, panel_rank)

    if not panel_records:
        return None

    ordered_panels = sorted(
        panel_records.items(),
        key=lambda item: item[1][2],
        reverse=True,
    )
    answer_changed = len(ordered_panels) > 1
    evidence_pages: list[EvidencePage] = []
    seen_page_keys: set[tuple[str, int]] = set()

    if answer_changed:
        for _panel_key, (_panel_values, evidence_page, _panel_rank) in ordered_panels[:2]:
            _append_unique_page(evidence_pages, seen_page_keys, evidence_page)
    elif ordered_panels:
        _append_unique_page(evidence_pages, seen_page_keys, ordered_panels[0][1][1])

    if plan.answer_type == "boolean":
        answer = answer_changed
    elif plan.answer_type == "names":
        ordered_names = _dedupe_preserve_order(
            [value for _panel_key, (panel_values, _evidence_page, _panel_rank) in ordered_panels for value in panel_values]
        )
        answer = ordered_names
    elif plan.answer_type == "name" and ordered_panels:
        answer = ordered_panels[0][1][0][0]
    else:
        return None

    return Resolution(
        answer=answer,
        evidence_pages=evidence_pages,
        confidence=0.99,
        method="judge_timeline_change",
        facts={"judge_panels": [panel_values for _panel_key, (panel_values, _evidence_page, _panel_rank) in ordered_panels]},
    )


def resolve_monetary_claim_compare(plan: QuestionPlan, store: PageMetadataStore) -> Resolution | None:
    if len(plan.case_ids) < 2:
        return None

    resolved_amounts: list[tuple[str, float, EvidencePage]] = []
    for case_id in plan.case_ids[:2]:
        selected_record = None
        for record in _iter_case_records_with_fallback(store, case_id, primary_page_hint=plan.page_hint):
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

    effective_page_hint = "front" if field == "judges" and plan.page_hint == "first" else plan.page_hint
    collected: list[tuple[dict[str, str], dict[str, EvidencePage], dict[str, tuple[int, int]], list[EvidencePage]]] = []
    for case_id in plan.case_ids[:2]:
        values_by_key, evidence_by_key, evidence_rank_by_key, coverage_pages = _collect_case_values(
            store,
            case_id,
            field=field,
            page_hint=effective_page_hint,
        )
        if not values_by_key:
            return None
        collected.append((values_by_key, evidence_by_key, evidence_rank_by_key, coverage_pages))

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
            best_overlap_key = max(
                overlap_keys,
                key=lambda overlap_key: sum(
                    evidence_rank_by_key.get(overlap_key, (0, 0))[0]
                    for _values_by_key, _evidence_by_key, evidence_rank_by_key, _coverage_pages in collected
                ),
            )
            for _values_by_key, evidence_by_key, _evidence_rank_by_key, _coverage_pages in collected:
                evidence_page = evidence_by_key.get(best_overlap_key)
                if evidence_page is not None:
                    append_page(evidence_page)
        else:
            for _values_by_key, _evidence_by_key, _evidence_rank_by_key, coverage_pages in collected:
                for evidence_page in coverage_pages:
                    append_page(evidence_page)
    elif plan.answer_type == "names":
        answer = [collected[0][0][key] for key in overlap_keys]
        for _values_by_key, evidence_by_key, _evidence_rank_by_key, _coverage_pages in collected:
            for overlap_key in overlap_keys:
                evidence_page = evidence_by_key.get(overlap_key)
                if evidence_page is not None:
                    append_page(evidence_page)
    elif plan.answer_type == "name" and overlap_keys:
        best_overlap_key = max(
            overlap_keys,
            key=lambda overlap_key: sum(
                evidence_rank_by_key.get(overlap_key, (0, 0))[0]
                for _values_by_key, _evidence_by_key, evidence_rank_by_key, _coverage_pages in collected
            ),
        )
        answer = collected[0][0][best_overlap_key]
        for _values_by_key, evidence_by_key, _evidence_rank_by_key, _coverage_pages in collected:
            evidence_page = evidence_by_key.get(best_overlap_key)
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
