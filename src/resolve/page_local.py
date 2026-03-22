from __future__ import annotations

import re

from src.generate.verbalize import verbalize_field_answer
from src.resolve.issue_date import select_best_issue_date_record
from src.resolve.metadata_store import PageMetadataStore
from src.resolve.models import EvidencePage, Resolution
from src.retrieve.lexical import tokenize_legal_text
from src.retrieve.question_plan import QuestionPlan

_LAW_NUMBER_RE = re.compile(r"\bDIFC\s+Law\s+No\.?\s+([0-9]+\s+of\s+[0-9]{4})\b", re.IGNORECASE)
_LAW_NUMBER_VALUE_RE = re.compile(r"\bDIFC\s+Law\s+No\.?\s+([0-9]+)\s+of\s+[0-9]{4}\b", re.IGNORECASE)
_PARTY_PREFIX_RE = re.compile(r"^(Claimant|Defendant|Applicant|Respondent|Appellant|Petitioner|Plaintiff):\s*", re.IGNORECASE)
_MONEY_RE = re.compile(r"\bAED\s*([0-9][0-9,]*(?:\.\d+)?)\b", re.IGNORECASE)
_CLAIM_VALUE_SIGNAL_RE = re.compile(
    r"\b(?:"
    r"claim value|"
    r"claim amount|"
    r"claims debt or damages|"
    r"seeking payment|"
    r"claim was for|"
    r"outstanding claim"
    r")\b",
    re.IGNORECASE,
)
_TITLE_PAGE_QUERY_RE = re.compile(
    r"(?:title|cover)\s+page\s+of\s+(?:the\s+)?(?P<title>.+?)(?:,|\?|$|\s+what\b)",
    re.IGNORECASE,
)
_QUOTED_TITLE_RE = re.compile(r"[\"“](?P<title>[^\"”]+)[\"”]")



def _extract_law_number(record: dict) -> str | None:
    for source in (record.get("doc_title"), record.get("text")):
        source_text = str(source or "")
        match = _LAW_NUMBER_RE.search(source_text)
        if match:
            law_number = re.sub(r"\s+of\s+", " of ", match.group(1), flags=re.IGNORECASE)
            return f"DIFC Law No. {law_number}"
    return None



def _extract_law_number_value(value: str) -> int | None:
    match = _LAW_NUMBER_VALUE_RE.search(str(value or ""))
    if not match:
        return None
    return int(match.group(1))



def _strip_party_prefix(value: str) -> str:
    return _PARTY_PREFIX_RE.sub("", str(value or "")).strip()



def _coerce_numeric_value(value: object) -> int | float | None:
    try:
        number = float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() else number



def _extract_money_value(record: dict) -> int | float | None:
    text = str(record.get("text") or "")
    money_matches = [(_coerce_numeric_value(match.group(1)), match.start()) for match in _MONEY_RE.finditer(text)]
    money_matches = [(value, position) for value, position in money_matches if value is not None]
    signal_positions = [match.start() for match in _CLAIM_VALUE_SIGNAL_RE.finditer(text)]

    if money_matches and signal_positions:
        for signal_position in signal_positions:
            post_signal_matches = [item for item in money_matches if item[1] >= signal_position]
            if post_signal_matches:
                return min(post_signal_matches, key=lambda item: item[1])[0]

        best_value, _best_position = min(
            money_matches,
            key=lambda item: min(abs(item[1] - signal_position) for signal_position in signal_positions),
        )
        return best_value

    if not signal_positions:
        return None

    numeric_values = [_coerce_numeric_value(value) for value in list(record.get("money_values") or [])]
    numeric_values = [value for value in numeric_values if value is not None]
    if not numeric_values:
        return None
    return max(numeric_values)



def _normalize_title_query(value: str) -> str:
    normalized = " ".join(str(value or "").split()).strip(" ,?.")
    lowered = normalized.lower()
    for prefix in ("according to the ", "from the ", "on the "):
        if lowered.startswith(prefix):
            normalized = normalized[len(prefix):]
            lowered = normalized.lower()
    return normalized.strip()



def _extract_title_page_query(question_text: str) -> str | None:
    match = _TITLE_PAGE_QUERY_RE.search(str(question_text or ""))
    if match:
        title_query = _normalize_title_query(match.group("title"))
        return title_query or None

    quoted_match = _QUOTED_TITLE_RE.search(str(question_text or ""))
    if not quoted_match:
        return None
    title_query = _normalize_title_query(quoted_match.group("title"))
    return title_query or None


def _asks_for_case_file_coverage(question_text: str) -> bool:
    lowered = str(question_text or "").lower()
    return any(
        marker in lowered
        for marker in (
            "all documents",
            "every document",
            "each document",
            "full case files",
            "case files",
        )
    )



def _score_title_page_record(record: dict, title_query: str) -> tuple[int, int, str]:
    normalized_query = _normalize_title_query(title_query).lower()
    if not normalized_query:
        return 0, 0, ""

    doc_title = " ".join(str(record.get("doc_title") or "").split())
    search_blob = " ".join(f"{doc_title} {str(record.get('text') or '')}".split())
    lowered_title = doc_title.lower()
    lowered_blob = search_blob.lower()

    query_tokens = set(tokenize_legal_text(normalized_query, doc_title=normalized_query))
    record_tokens = set(tokenize_legal_text(search_blob, doc_title=doc_title))
    overlap = len(query_tokens & record_tokens)

    score = overlap * 10
    if normalized_query in lowered_blob:
        score += 80
    if normalized_query in lowered_title:
        score += 40
    if _extract_law_number(record) is not None:
        score += 5

    return score, overlap, str(record.get("doc_id") or "")



def _resolve_law_title_page_lookup(plan: QuestionPlan, store: PageMetadataStore, question_text: str) -> Resolution | None:
    if plan.target_field != "law_number" or plan.mode != "title_page_metadata":
        return None

    title_query = _extract_title_page_query(question_text)
    if not title_query:
        return None

    best_record: dict | None = None
    best_value: str | None = None
    best_rank: tuple[int, int, str] | None = None

    for record in store.get_first_page_records():
        value = _extract_law_number(record)
        if value is None:
            continue

        rank = _score_title_page_record(record, title_query)
        if rank[0] <= 0:
            continue

        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_record = record
            best_value = value

    if best_record is None or best_value is None:
        return None

    evidence = EvidencePage(doc_id=str(best_record.get("doc_id") or ""), page_num=int(best_record.get("page_num") or 0))
    answer = _shape_answer(best_value, plan, question_text)
    if answer is None:
        return None

    return Resolution(
        answer=answer,
        evidence_pages=[evidence],
        confidence=0.99,
        method=plan.mode,
        facts={"target_field": plan.target_field, "value": best_value, "title_query": title_query},
    )



def _filter_party_values(values: list[str], question_text: str) -> list[str]:
    lowered_question = str(question_text or "").lower()
    if any(marker in lowered_question for marker in ("initiated the proceedings", "initiated proceedings")):
        filtered = [
            value for value in values
            if value.lower().startswith(("claimant:", "applicant:", "plaintiff:", "petitioner:"))
        ]
        return filtered or values
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


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _resolve_case_title_page_coverage(plan: QuestionPlan, store: PageMetadataStore, question_text: str) -> Resolution | None:
    if len(plan.case_ids) != 1 or plan.page_hint != "first" or not _asks_for_case_file_coverage(question_text):
        return None
    if plan.target_field != "party":
        return None

    case_id = plan.case_ids[0]
    records = store.get_case_records(case_id, page_hint=plan.page_hint)
    if not records:
        return None

    collected_values: list[str] = []
    evidence_pages: list[EvidencePage] = []
    seen_evidence: set[tuple[str, int]] = set()

    for record in records:
        value = _extract_value(record, plan, question_text=question_text)
        if isinstance(value, list):
            collected_values.extend(str(item) for item in value if str(item).strip())
        elif value is not None and str(value).strip():
            collected_values.append(str(value))

        evidence_key = (str(record.get("doc_id") or ""), int(record.get("page_num") or 0))
        if not evidence_key[0] or evidence_key[1] <= 0 or evidence_key in seen_evidence:
            continue
        seen_evidence.add(evidence_key)
        evidence_pages.append(EvidencePage(doc_id=evidence_key[0], page_num=evidence_key[1]))

    collected_values = _dedupe_preserve_order(collected_values)
    if not collected_values or not evidence_pages:
        return None

    answer = _shape_answer(collected_values, plan, question_text)
    if answer is None:
        return None

    return Resolution(
        answer=answer,
        evidence_pages=evidence_pages,
        confidence=0.99,
        method=plan.mode,
        facts={"case_id": case_id, "target_field": plan.target_field, "value": collected_values},
    )



def resolve_page_local_lookup(plan: QuestionPlan, store: PageMetadataStore, question_text: str) -> Resolution | None:
    if not plan.case_ids:
        return _resolve_law_title_page_lookup(plan, store, question_text)

    title_page_coverage_resolution = _resolve_case_title_page_coverage(plan, store, question_text)
    if title_page_coverage_resolution is not None:
        return title_page_coverage_resolution

    case_id = plan.case_ids[0]
    if plan.target_field == "issue_date":
        selected_record = select_best_issue_date_record(store.get_case_records(case_id, page_hint=plan.page_hint))
        if selected_record is None:
            return None
        value, record = selected_record
        evidence = EvidencePage(doc_id=str(record.get("doc_id") or ""), page_num=int(record.get("page_num") or 0))
        answer = _shape_answer(value, plan, question_text)
        if answer is None:
            return None
        return Resolution(
            answer=answer,
            evidence_pages=[evidence],
            confidence=0.99,
            method=plan.mode,
            facts={"case_id": case_id, "target_field": plan.target_field, "value": value},
        )

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
    if target_field == "money_value":
        return _extract_money_value(record)
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
        if plan.answer_type in {"names", "number"}:
            return values
        return values[0]
    if target_field == "law_number":
        return _extract_law_number(record)
    return None



def _shape_answer(value, plan: QuestionPlan, question_text: str):
    if plan.answer_type == "number":
        if plan.target_field == "party":
            if isinstance(value, list):
                normalized_parties = [_strip_party_prefix(str(item)) for item in value if _strip_party_prefix(str(item))]
                return len(_dedupe_preserve_order(normalized_parties)) if normalized_parties else None
            normalized_value = _strip_party_prefix(str(value))
            return 1 if normalized_value else None
        if plan.target_field == "law_number":
            return _extract_law_number_value(str(value))
        return _coerce_numeric_value(value)
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
