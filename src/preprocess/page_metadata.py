from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.case_ids import extract_case_ids as _shared_extract_case_ids, normalize_case_id as _shared_normalize_case_id
from src.config import ARTICLE_PAGE_MAP_JSON, CASE_METADATA_JSON, PAGE_METADATA_JSONL, PAGES_JSONL
from src.preprocess.chunk import _detect_doc_title

logger = logging.getLogger(__name__)

_ARTICLE_REF_RE = re.compile(r"\bArticle\s+\d+[A-Z]?(?:\(\d+[A-Z]?\)|\([a-z]\))*", re.IGNORECASE)
_CLAIM_NUMBER_RE = re.compile(
    r"\b(?:Claim\s+No\.?|Case\s+No\.?|Claim\s+number|claim number)\s*[:\-]?\s*([A-Z]{2,5}[- ]?\d+[-/]\d+(?:/\d+)?)",
    re.IGNORECASE,
)
_CASE_STYLE_NUMBER_RE = re.compile(r"\b[A-Z]{2,5}-\d+-\d+(?:/\d+)?\b")
_MONEY_RE = re.compile(r"\b(?:USD|AED)\s*([0-9][0-9,]*(?:\.\d+)?)\b", re.IGNORECASE)
_PARTY_LINE_RE = re.compile(
    r"^\s*(Claimant|Defendant|Applicant|Respondent|Appellant|Petitioner|Plaintiff)\s*[:\-]?\s*(.+?)\s*$",
    re.IGNORECASE,
)
_BEFORE_LINE_RE = re.compile(r"^\s*Before\s*[:\-]?\s*(.+?)\s*$", re.IGNORECASE)
_JUSTICE_NAME_RE = re.compile(
    r"\b(?:Chief\s+Justice|Deputy\s+Chief\s+Justice|Justice|Judge)\s+[A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*){0,5}"
)
_JUDGE_TITLE_RE = re.compile(r"\b(?:Chief\s+Justice|Deputy\s+Chief\s+Justice|Justice|Judge)\b", re.IGNORECASE)
_JUDGE_HEADING_RE = re.compile(
    r"(?:^|\n)\s*(?:CASE MANAGEMENT ORDER OF|JUDGMENT OF|AMENDED JUDGMENT OF|ORDER WITH REASONS OF|ORDER OF)\s+"
    r"(.+?)(?=(?:\n|(?:\s+(?:UPON|AND UPON|TRIAL\b|HEARING\b|COUNSEL\b|JUDGMENT\s+\d|INTRODUCTION\b|REASONS\b|BETWEEN\b|IN THE COURT\b|IT IS HEREBY\b)))|$)",
    re.IGNORECASE | re.DOTALL,
)
_JUDGE_INLINE_HEADING_RE = re.compile(
    r"\b(?:CASE MANAGEMENT ORDER OF|JUDGMENT OF|AMENDED JUDGMENT OF|ORDER WITH REASONS OF|ORDER OF)\s+"
    r"(.+?)(?=(?:\s+(?:UPON|AND UPON|TRIAL\b|HEARING\b|COUNSEL\b|JUDGMENT\s+\d|INTRODUCTION\b|REASONS\b|BETWEEN\b|IN THE COURT\b|IT IS HEREBY\b))|$)",
    re.IGNORECASE,
)
_JUDGE_BEFORE_BLOCK_RE = re.compile(
    r"(?:^|\n)\s*BEFORE\s+(.+?)(?=(?:\nBETWEEN|\nREASONS\b|\nINTRODUCTION\b|\nORDER\b|\nUPON\b|$))",
    re.IGNORECASE | re.DOTALL,
)
_JUDGE_HELD_BEFORE_RE = re.compile(
    r"\bheld before\s+(.+?)(?=\s*\(|\s+(?:IT IS HEREBY\b|FOR REASONS\b|REASONS\b|INTRODUCTION\b)|$)",
    re.IGNORECASE,
)
_CAPTION_ROLE_RE = re.compile(
    r"^(Claimant|Defendant|Applicant|Respondent|Appellant|Petitioner|Plaintiff|Claimants|Defendants|Applicants|Respondents|Appellants|Petitioners|Plaintiffs)(?:/[A-Za-z]+)?$",
    re.IGNORECASE,
)
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_DMY_DATE_RE = re.compile(
    r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b",
    re.IGNORECASE,
)
_MDY_DATE_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b",
    re.IGNORECASE,
)
_DATE_OF_ISSUE_LABEL_RE = re.compile(r"date of issue\s*[:\-]?\s*", re.IGNORECASE)
_DATE_CONTEXT_RE = re.compile(r"(?:date of issue|issued on|issued first|issue date)", re.IGNORECASE)
_ORDER_SIGNAL_PATTERNS = (
    "dismissed",
    "refused",
    "granted",
    "allowed",
    "discharged",
    "restored",
    "proceed to trial",
    "no order as to costs",
    "costs",
)
_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}



def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()



def _dedupe_preserve_order(values: list[Any]) -> list[Any]:
    seen: set[Any] = set()
    output: list[Any] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output



def normalize_case_id(raw_value: str) -> str:
    return _shared_normalize_case_id(raw_value)



def extract_case_ids(text: str) -> list[str]:
    return _dedupe_preserve_order(_shared_extract_case_ids(text))



def extract_article_refs(text: str) -> list[str]:
    values = []
    for match in _ARTICLE_REF_RE.finditer(text or ""):
        values.append(_normalize_space(match.group(0)))
    return _dedupe_preserve_order(values)



def _iso_date_from_match(year: str, month: str, day: str) -> str:
    return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"



def _collect_date_matches(normalized: str) -> list[tuple[int, str]]:
    matches: list[tuple[int, str]] = []
    for match in _ISO_DATE_RE.finditer(normalized):
        matches.append((match.start(), _iso_date_from_match(*match.groups())))
    for match in _DMY_DATE_RE.finditer(normalized):
        day, month_name, year = match.groups()
        matches.append((match.start(), _iso_date_from_match(year, str(_MONTHS[month_name.lower()]), day)))
    for match in _MDY_DATE_RE.finditer(normalized):
        month_name, day, year = match.groups()
        matches.append((match.start(), _iso_date_from_match(year, str(_MONTHS[month_name.lower()]), day)))
    return matches


def extract_issue_date(text: str) -> str | None:
    normalized = _normalize_space(text)
    if not normalized:
        return None

    matches = _collect_date_matches(normalized)
    if not matches:
        return None

    for label_match in _DATE_OF_ISSUE_LABEL_RE.finditer(normalized):
        label_end = label_match.end()
        explicit_matches = [item for item in matches if item[0] >= label_end and item[0] - label_end <= 80]
        if explicit_matches:
            return min(explicit_matches, key=lambda item: item[0])[1]

    contextual_positions = [match.start() for match in _DATE_CONTEXT_RE.finditer(normalized)]
    if contextual_positions:
        best = min(
            matches,
            key=lambda item: min(abs(item[0] - context_pos) for context_pos in contextual_positions),
        )
        return best[1]

    return min(matches, key=lambda item: item[0])[1]



def extract_claim_numbers(text: str) -> list[str]:
    explicit_matches = [_normalize_space(match.group(1)).replace(" ", "-") for match in _CLAIM_NUMBER_RE.finditer(text or "")]
    if explicit_matches:
        return _dedupe_preserve_order(explicit_matches)
    fallback_matches = [_normalize_space(match.group(0)) for match in _CASE_STYLE_NUMBER_RE.finditer(text or "")]
    return _dedupe_preserve_order(fallback_matches)



def _normalize_judge_name(value: str) -> str | None:
    normalized = _normalize_space(re.sub(r"\bH\.?\s*E\.?\b", "", str(value or ""), flags=re.IGNORECASE))
    title_match = _JUDGE_TITLE_RE.search(normalized)
    if title_match is not None:
        normalized = normalized[title_match.start() :]
    normalized = re.sub(r"\s*\(.*$", "", normalized)
    normalized = re.sub(
        r"\b(?:dated|on|upon|and upon|for(?: the)?|at(?: the)?)\b.*$",
        "",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r"\bof\s+\d{1,2}\s+[A-Za-z]+\s+\d{4}\b.*$", "", normalized, flags=re.IGNORECASE)
    normalized = normalized.strip(" ,;:-")
    normalized = re.sub(r"^[^A-Za-z]+", "", normalized)
    if not normalized:
        return None
    if not re.search(r"\b(chief justice|deputy chief justice|justice|judge)\b", normalized, re.IGNORECASE):
        return None
    if "assistant registrar" in normalized.lower():
        return None
    if normalized == normalized.upper():
        normalized = normalized.title()
        normalized = re.sub(r"\bKc\b", "KC", normalized)
        normalized = re.sub(r"\bAc\b", "AC", normalized)
    return normalized



def _split_judge_blob(blob: str) -> list[str]:
    normalized_blob = _normalize_space(str(blob or "").replace("\n", " "))
    if not normalized_blob:
        return []
    normalized_blob = re.sub(r"\s+and\s+", ", ", normalized_blob, flags=re.IGNORECASE)
    values: list[str] = []
    for part in re.split(r"\s*,\s*", normalized_blob):
        candidate = _normalize_judge_name(part)
        if candidate:
            values.append(candidate)
    return values



def extract_judges(text: str) -> list[str]:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    values: list[str] = []
    joined_head = "\n".join(lines[:20])
    flat_head = _normalize_space(" ".join(lines[:20]))

    for line in lines[:12]:
        before_match = _BEFORE_LINE_RE.match(line)
        if before_match:
            values.extend(_split_judge_blob(before_match.group(1)))

    for pattern, search_blob in (
        (_JUDGE_BEFORE_BLOCK_RE, joined_head),
        (_JUDGE_HELD_BEFORE_RE, flat_head),
        (_JUDGE_HEADING_RE, joined_head),
    ):
        for match in pattern.finditer(search_blob):
            values.extend(_split_judge_blob(match.group(1)))

    if not values:
        for match in _JUDGE_INLINE_HEADING_RE.finditer(flat_head):
            inline_values = _split_judge_blob(match.group(1))
            if inline_values:
                values.extend(inline_values)
                break

    if not values:
        for match in _JUSTICE_NAME_RE.finditer(flat_head):
            candidate = _normalize_judge_name(match.group(0))
            if candidate:
                values.append(candidate)

    return _dedupe_preserve_order(values)



def _extract_party_role(line: str) -> str | None:
    normalized = _normalize_space(line)
    if not normalized:
        return None
    match = _CAPTION_ROLE_RE.match(normalized)
    if not match:
        return None
    role = match.group(1)
    if role.lower().endswith("s"):
        role = role[:-1]
    return role.title()



def _clean_party_entity(line: str) -> str | None:
    normalized = _normalize_space(re.sub(r"^(?:\(\d+\)|\d+\.)\s*", "", str(line or "")))
    if not normalized:
        return None
    if normalized.lower() in {"between", "and"}:
        return None
    if _extract_party_role(normalized):
        return None
    upper_normalized = normalized.upper()
    if upper_normalized.startswith(("ORDER", "UPON", "IT IS HEREBY", "IN THE", "REASONS")):
        return None
    return normalized



def _extract_caption_parties(lines: list[str]) -> list[str]:
    start_index = 0
    for index, line in enumerate(lines):
        if line.strip().upper() == "BETWEEN":
            start_index = index + 1
            break
    caption_lines = lines[start_index:]

    values: list[str] = []
    for index, line in enumerate(caption_lines):
        role = _extract_party_role(line)
        if not role:
            continue

        entities: list[str] = []
        cursor = index - 1
        while cursor >= 0:
            raw_candidate = caption_lines[cursor]
            if _extract_party_role(raw_candidate):
                break
            stripped = raw_candidate.strip()
            if stripped.lower() == "and":
                cursor -= 1
                continue
            candidate = _clean_party_entity(stripped)
            if candidate is None:
                if entities:
                    break
                cursor -= 1
                continue
            entities.append(candidate)
            cursor -= 1
        for entity in reversed(entities):
            values.append(f"{role}: {entity}")
    return values



def extract_parties(text: str) -> list[str]:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    values: list[str] = []

    for line in lines[:20]:
        match = _PARTY_LINE_RE.match(line)
        if not match:
            continue
        role, value = match.groups()
        normalized_value = _normalize_space(value)
        if not normalized_value or normalized_value.startswith("/") or _extract_party_role(normalized_value):
            continue
        values.append(f"{role.title()}: {normalized_value}")

    values.extend(_extract_caption_parties(lines[:25]))
    return _dedupe_preserve_order(values)



def extract_money_values(text: str) -> list[float | int]:
    values: list[float | int] = []
    for match in _MONEY_RE.finditer(text or ""):
        raw_value = match.group(1).replace(",", "")
        number = float(raw_value)
        values.append(int(number) if number.is_integer() else number)
    return _dedupe_preserve_order(values)



def extract_order_signals(text: str) -> list[str]:
    lowered = str(text or "").lower()
    hits = [pattern for pattern in _ORDER_SIGNAL_PATTERNS if pattern in lowered]
    return _dedupe_preserve_order(hits)



def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
    return records



def build_page_metadata_records(pages_path: Path | str | None = None) -> list[dict[str, Any]]:
    pages_path = Path(pages_path) if pages_path else PAGES_JSONL
    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in _read_jsonl_records(pages_path):
        doc_id = str(record.get("doc_id") or "").strip()
        page_num = int(record.get("page_num") or 0)
        if not doc_id or page_num <= 0:
            continue
        grouped_rows[doc_id].append(record)

    page_records: list[dict[str, Any]] = []
    for doc_id in sorted(grouped_rows):
        rows = sorted(grouped_rows[doc_id], key=lambda row: int(row.get("page_num") or 0))
        title_sample = "\n\n".join(str(row.get("text") or "") for row in rows[:2])
        doc_title = _detect_doc_title(title_sample)
        last_page = max(int(row.get("page_num") or 0) for row in rows)
        doc_case_ids = _dedupe_preserve_order(
            [
                case_id
                for row in rows[: min(3, len(rows))]
                for case_id in extract_case_ids(str(row.get("text") or ""))
            ]
        )

        for row in rows:
            page_num = int(row.get("page_num") or 0)
            text = str(row.get("text") or "")
            row_case_ids = extract_case_ids(text)
            metadata = {
                "doc_id": doc_id,
                "page_num": page_num,
                "doc_title": doc_title,
                "is_first_page": page_num == 1,
                "is_last_page": page_num == last_page,
                "case_ids": row_case_ids or doc_case_ids,
                "issue_date": extract_issue_date(text),
                "judges": extract_judges(text),
                "parties": extract_parties(text),
                "claim_numbers": extract_claim_numbers(text),
                "article_refs": extract_article_refs(text),
                "money_values": extract_money_values(text),
                "order_signals": extract_order_signals(text),
                "text": text,
            }
            page_records.append(metadata)

    return page_records



def build_case_metadata(page_records: list[dict[str, Any]]) -> dict[str, Any]:
    by_case: dict[str, dict[str, Any]] = {}
    for record in page_records:
        for case_id in record.get("case_ids", []):
            entry = by_case.setdefault(case_id, {"doc_ids": [], "pages": []})
            if record["doc_id"] not in entry["doc_ids"]:
                entry["doc_ids"].append(record["doc_id"])
            entry["pages"].append(
                {
                    "doc_id": record["doc_id"],
                    "page_num": record["page_num"],
                    "doc_title": record.get("doc_title") or "",
                    "is_first_page": bool(record.get("is_first_page")),
                    "is_last_page": bool(record.get("is_last_page")),
                    "issue_date": record.get("issue_date"),
                    "judges": list(record.get("judges") or []),
                    "parties": list(record.get("parties") or []),
                    "claim_numbers": list(record.get("claim_numbers") or []),
                    "money_values": list(record.get("money_values") or []),
                    "order_signals": list(record.get("order_signals") or []),
                }
            )

    for case_id, entry in by_case.items():
        entry["doc_ids"] = sorted(entry["doc_ids"])
        entry["pages"] = sorted(entry["pages"], key=lambda item: (int(item["page_num"]), str(item["doc_id"])))
    return dict(sorted(by_case.items()))



def build_article_page_map(page_records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    article_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in page_records:
        for article_ref in record.get("article_refs", []):
            article_map[article_ref].append(
                {
                    "doc_id": record["doc_id"],
                    "page_num": record["page_num"],
                    "doc_title": record.get("doc_title") or "",
                }
            )

    output: dict[str, list[dict[str, Any]]] = {}
    for article_ref, refs in article_map.items():
        output[article_ref] = sorted(refs, key=lambda item: (str(item["doc_title"]), int(item["page_num"]), str(item["doc_id"])))
    return dict(sorted(output.items()))



def build_page_metadata_indices(
    pages_path: Path | str | None = None,
    page_metadata_path: Path | str | None = None,
    case_metadata_path: Path | str | None = None,
    article_page_map_path: Path | str | None = None,
) -> tuple[Path, Path, Path]:
    pages_path = Path(pages_path) if pages_path else PAGES_JSONL
    page_metadata_path = Path(page_metadata_path) if page_metadata_path else PAGE_METADATA_JSONL
    case_metadata_path = Path(case_metadata_path) if case_metadata_path else CASE_METADATA_JSON
    article_page_map_path = Path(article_page_map_path) if article_page_map_path else ARTICLE_PAGE_MAP_JSON

    page_records = build_page_metadata_records(pages_path)
    case_metadata = build_case_metadata(page_records)
    article_page_map = build_article_page_map(page_records)

    page_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(page_metadata_path, "w", encoding="utf-8") as handle:
        for record in page_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    with open(case_metadata_path, "w", encoding="utf-8") as handle:
        json.dump(case_metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    with open(article_page_map_path, "w", encoding="utf-8") as handle:
        json.dump(article_page_map, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    logger.info(
        "Page metadata indices saved: %s records, %s cases, %s article refs",
        len(page_records),
        len(case_metadata),
        len(article_page_map),
    )
    return page_metadata_path, case_metadata_path, article_page_map_path
