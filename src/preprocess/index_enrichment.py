from __future__ import annotations

from typing import Any, Iterable

from src.preprocess.page_metadata import (
    extract_article_refs,
    extract_case_ids,
    extract_claim_numbers,
    extract_issue_date,
    extract_judges,
    extract_money_values,
    extract_order_signals,
    extract_parties,
)

_MAX_CASE_IDS = 4
_MAX_CLAIM_NUMBERS = 4
_MAX_ARTICLE_REFS = 8
_MAX_NAMES = 8
_MAX_ORDER_SIGNALS = 6
_MAX_MONEY_VALUES = 8


def _normalize_scalar(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _normalize_string_list(values: Any, *, limit: int | None = None) -> list[str]:
    if values is None:
        return []

    if isinstance(values, (list, tuple, set)):
        raw_values: Iterable[Any] = values
    else:
        raw_values = [values]

    output: list[str] = []
    seen: set[str] = set()
    for value in raw_values:
        normalized = _normalize_scalar(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
        if limit is not None and len(output) >= limit:
            break
    return output


def _normalize_money_values(values: Any, *, limit: int = _MAX_MONEY_VALUES) -> list[str]:
    if values is None:
        return []

    if isinstance(values, (list, tuple, set)):
        raw_values: Iterable[Any] = values
    else:
        raw_values = [values]

    output: list[str] = []
    seen: set[str] = set()
    for value in raw_values:
        if isinstance(value, float) and value.is_integer():
            normalized = str(int(value))
        else:
            normalized = _normalize_scalar(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
        if len(output) >= limit:
            break
    return output


def build_doc_level_metadata(page_records: Iterable[dict[str, Any]]) -> dict[str, dict[str, list[str]]]:
    """
    Propagate stable document identifiers, especially case citations, to every
    chunk/page index record built from the same source document.
    """
    by_doc: dict[str, dict[str, list[str]]] = {}

    for record in page_records:
        doc_id = _normalize_scalar(record.get("doc_id"))
        if not doc_id:
            continue

        entry = by_doc.setdefault(
            doc_id,
            {
                "doc_case_ids": [],
                "doc_claim_numbers": [],
            },
        )

        for case_id in _normalize_string_list(record.get("case_ids"), limit=_MAX_CASE_IDS):
            if case_id not in entry["doc_case_ids"]:
                entry["doc_case_ids"].append(case_id)

        for claim_number in _normalize_string_list(record.get("claim_numbers"), limit=_MAX_CLAIM_NUMBERS):
            if claim_number not in entry["doc_claim_numbers"]:
                entry["doc_claim_numbers"].append(claim_number)

    return by_doc


def enrich_record_for_indexing(
    record: dict[str, Any],
    doc_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Attach compact legal metadata used by both BM25 and dense indexing.

    The key improvement is doc-level case/claim propagation so later pages and
    deep chunks remain retrievable for case-specific questions.
    """
    enriched = dict(record)

    doc_title = _normalize_scalar(enriched.get("doc_title"))
    section_path = _normalize_scalar(enriched.get("section_path"))
    text = str(enriched.get("text") or "")
    composite_text = "\n".join(part for part in (doc_title, section_path, text) if part)

    propagated_case_ids = _normalize_string_list((doc_metadata or {}).get("doc_case_ids"), limit=_MAX_CASE_IDS)
    propagated_claim_numbers = _normalize_string_list(
        (doc_metadata or {}).get("doc_claim_numbers"),
        limit=_MAX_CLAIM_NUMBERS,
    )

    case_ids = _normalize_string_list(
        [*(_normalize_string_list(enriched.get("case_ids"))), *propagated_case_ids, *extract_case_ids(composite_text)],
        limit=_MAX_CASE_IDS,
    )
    claim_numbers = _normalize_string_list(
        [
            *(_normalize_string_list(enriched.get("claim_numbers"))),
            *propagated_claim_numbers,
            *extract_claim_numbers(composite_text),
        ],
        limit=_MAX_CLAIM_NUMBERS,
    )

    enriched["doc_case_ids"] = propagated_case_ids
    enriched["doc_claim_numbers"] = propagated_claim_numbers
    enriched["case_ids"] = case_ids
    enriched["claim_numbers"] = claim_numbers
    enriched["article_refs"] = _normalize_string_list(
        enriched.get("article_refs") or extract_article_refs("\n".join(part for part in (section_path, text) if part)),
        limit=_MAX_ARTICLE_REFS,
    )
    enriched["issue_date"] = _normalize_scalar(enriched.get("issue_date") or extract_issue_date(text))
    enriched["judges"] = _normalize_string_list(enriched.get("judges") or extract_judges(text), limit=_MAX_NAMES)
    enriched["parties"] = _normalize_string_list(enriched.get("parties") or extract_parties(text), limit=_MAX_NAMES)
    enriched["order_signals"] = _normalize_string_list(
        enriched.get("order_signals") or extract_order_signals(text),
        limit=_MAX_ORDER_SIGNALS,
    )
    enriched["money_values"] = _normalize_money_values(
        enriched.get("money_values") or extract_money_values(text),
        limit=_MAX_MONEY_VALUES,
    )
    return enriched


def build_embedding_input(record: dict[str, Any]) -> str:
    """
    Build a dense-retrieval body that preserves short structured signals without
    replacing the original passage text.
    """
    text = str(record.get("text") or "").strip()
    metadata_lines: list[str] = []

    page_num = record.get("page_num")
    if isinstance(page_num, int) and page_num > 0:
        metadata_lines.append(f"Page number: {page_num}")

    page_roles: list[str] = []
    if bool(record.get("is_first_page")):
        page_roles.append("first page")
    if bool(record.get("is_last_page")):
        page_roles.append("last page")
    if page_roles:
        metadata_lines.append(f"Page role: {', '.join(page_roles)}")

    field_specs = (
        ("case_ids", "Case IDs"),
        ("claim_numbers", "Claim numbers"),
        ("article_refs", "Article references"),
        ("issue_date", "Issue date"),
        ("judges", "Judges"),
        ("parties", "Parties"),
        ("order_signals", "Order signals"),
        ("money_values", "Money values"),
    )
    for field_name, label in field_specs:
        values = record.get(field_name)
        normalized_values = (
            _normalize_money_values(values)
            if field_name == "money_values"
            else _normalize_string_list(values)
        )
        if normalized_values:
            metadata_lines.append(f"{label}: {'; '.join(normalized_values)}")

    if not metadata_lines:
        return text

    if text:
        metadata_lines.append("Passage:")
        metadata_lines.append(text)
    return "\n".join(metadata_lines)
