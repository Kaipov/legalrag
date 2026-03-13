from __future__ import annotations

import re

from src.generate.verbalize import verbalize_outcome_clauses
from src.resolve.metadata_store import PageMetadataStore
from src.resolve.models import EvidencePage, Resolution
from src.retrieve.question_plan import QuestionPlan

_SECTION_MARKERS = ("IT IS HEREBY ORDERED THAT", "CONCLUSION")
_STOP_MARKERS = ("Issued by:", "SCHEDULE OF REASONS", "REASONS")
_OUTCOME_KEYWORDS = (
    "refused",
    "dismissed",
    "granted",
    "allowed",
    "denied",
    "reconsidered at a hearing",
    "reconsideration",
    "costs",
    "no order as to costs",
    "bear its own costs",
    "bear their own costs",
    "permission to appeal",
    "oral hearing",
    "proceed to trial",
    "set aside",
    "struck out",
)
_NUMBERED_CLAUSE_RE = re.compile(r"(?:^|\n)\s*(?:\(?\d+\)?[.):]?)\s*(.+?)(?=(?:\n\s*(?:\(?\d+\)?[.):]?)\s+)|$)", re.DOTALL)


def _normalize_clause_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().rstrip(".;:")



def _truncate_at_stop_marker(text: str) -> tuple[str, bool]:
    stop_positions = [text.find(marker) for marker in _STOP_MARKERS if marker in text]
    if not stop_positions:
        return text, False
    stop_at = min(position for position in stop_positions if position >= 0)
    return text[:stop_at], True



def _split_outcome_candidates(text: str) -> list[str]:
    if not text.strip():
        return []

    numbered_matches = [
        _normalize_clause_text(match.group(1))
        for match in _NUMBERED_CLAUSE_RE.finditer(text)
        if _normalize_clause_text(match.group(1))
    ]
    if numbered_matches:
        return numbered_matches

    line_candidates: list[str] = []
    for line in text.splitlines():
        cleaned = _normalize_clause_text(line)
        if cleaned:
            line_candidates.append(cleaned)
    return line_candidates



def _is_outcome_clause(clause: str) -> bool:
    lowered = clause.lower()
    return any(keyword in lowered for keyword in _OUTCOME_KEYWORDS)


def _has_terminal_outcome_support(record: dict) -> bool:
    order_signals = {str(value).strip().lower() for value in list(record.get("order_signals") or []) if str(value).strip()}
    if order_signals:
        return True
    text = str(record.get("text") or "")
    return any(_is_outcome_clause(clause) for clause in _split_outcome_candidates(text))



def _extract_order_section_clauses(records: list[dict]) -> list[tuple[str, EvidencePage]]:
    collected: list[tuple[str, EvidencePage]] = []
    in_section = False

    for record in records:
        raw_text = str(record.get("text") or "")
        if not raw_text.strip():
            continue

        section_text = raw_text
        if not in_section:
            upper_text = raw_text.upper()
            marker_index = -1
            marker_value = ""
            for marker in _SECTION_MARKERS:
                candidate_index = upper_text.find(marker)
                if candidate_index == -1:
                    continue
                if marker_index == -1 or candidate_index < marker_index:
                    marker_index = candidate_index
                    marker_value = marker
            if marker_index == -1:
                continue
            in_section = True
            section_text = raw_text[marker_index + len(marker_value):]

        section_text, hit_stop = _truncate_at_stop_marker(section_text)
        clauses = [clause for clause in _split_outcome_candidates(section_text) if _is_outcome_clause(clause)]
        if clauses:
            evidence = EvidencePage(doc_id=str(record.get("doc_id") or ""), page_num=int(record.get("page_num") or 0))
            for clause in clauses:
                collected.append((clause, evidence))
        if hit_stop:
            break

    return collected



def _extract_fallback_outcome_clauses(records: list[dict]) -> list[tuple[str, EvidencePage]]:
    collected: list[tuple[str, EvidencePage]] = []
    for record in records[-2:]:
        text = str(record.get("text") or "")
        clauses = [clause for clause in _split_outcome_candidates(text) if _is_outcome_clause(clause)]
        if not clauses:
            continue
        evidence = EvidencePage(doc_id=str(record.get("doc_id") or ""), page_num=int(record.get("page_num") or 0))
        for clause in clauses:
            collected.append((clause, evidence))
    return collected


def _select_terminal_outcome_evidence(
    records: list[dict],
    existing_pages: list[EvidencePage],
) -> EvidencePage | None:
    if not records or not existing_pages:
        return None

    max_existing_page = max(page.page_num for page in existing_pages)
    terminal_record = None
    for record in reversed(records):
        page_num = int(record.get("page_num") or 0)
        if page_num <= max_existing_page:
            continue
        if not _has_terminal_outcome_support(record):
            continue
        terminal_record = record
        if bool(record.get("is_last_page")):
            break

    if terminal_record is None:
        return None

    doc_id = str(terminal_record.get("doc_id") or "").strip()
    page_num = int(terminal_record.get("page_num") or 0)
    if not doc_id or page_num <= 0:
        return None
    return EvidencePage(doc_id=doc_id, page_num=page_num)



def resolve_last_page_outcome(plan: QuestionPlan, store: PageMetadataStore, *, question_text: str) -> Resolution | None:
    if plan.answer_type != "free_text" or not plan.case_ids:
        return None

    case_id = plan.case_ids[0]
    records = store.get_case_records(case_id, page_hint="any")
    if not records:
        return None

    records = sorted(records, key=lambda item: int(item.get("page_num") or 0))
    ordered_clauses = _extract_order_section_clauses(records)
    if not ordered_clauses:
        ordered_clauses = _extract_fallback_outcome_clauses(records)
    if not ordered_clauses:
        return None

    answer = verbalize_outcome_clauses([clause for clause, _evidence in ordered_clauses], question_text=question_text)
    evidence_pages: list[EvidencePage] = []
    seen_pages: set[tuple[str, int]] = set()
    for _clause, evidence in ordered_clauses:
        page_key = (evidence.doc_id, evidence.page_num)
        if page_key in seen_pages or not evidence.doc_id or evidence.page_num <= 0:
            continue
        seen_pages.add(page_key)
        evidence_pages.append(evidence)

    terminal_evidence = _select_terminal_outcome_evidence(records, evidence_pages)
    if terminal_evidence is not None:
        terminal_key = (terminal_evidence.doc_id, terminal_evidence.page_num)
        if terminal_key not in seen_pages:
            seen_pages.add(terminal_key)
            evidence_pages.append(terminal_evidence)

    if not answer or not evidence_pages:
        return None

    return Resolution(
        answer=answer,
        evidence_pages=evidence_pages,
        confidence=0.97,
        method="last_page_outcome",
        facts={
            "case_id": case_id,
            "clauses": [clause for clause, _evidence in ordered_clauses],
        },
    )
