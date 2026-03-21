from __future__ import annotations

import re

from src.preprocess.page_metadata import extract_issue_date

_DATE_OF_ISSUE_LABEL_RE = re.compile(r"date of issue\s*[:\-]?\s*", re.IGNORECASE)
_DATE_CONTEXT_RE = re.compile(r"(?:date of issue|issued on|issue date)", re.IGNORECASE)


def resolve_issue_date_record(record: dict) -> tuple[str, tuple[int, int]] | None:
    text = str(record.get("text") or "")
    text_issue_date = extract_issue_date(text)
    stored_issue_date = str(record.get("issue_date") or "").strip() or None
    resolved_issue_date = text_issue_date or stored_issue_date
    if not resolved_issue_date:
        return None

    normalized_text = " ".join(text.split())
    page_num = int(record.get("page_num") or 0)
    is_first_page = bool(record.get("is_first_page")) or page_num == 1
    has_explicit_label = bool(_DATE_OF_ISSUE_LABEL_RE.search(normalized_text))
    has_issue_context = has_explicit_label or bool(_DATE_CONTEXT_RE.search(normalized_text))

    # Prefer an explicit issue-date clause on the first two pages. When the
    # same signal only appears deeper in the front matter, trust the first page
    # document metadata instead of later procedural dates.
    rank_score = 0
    if has_explicit_label:
        rank_score += 130 if page_num <= 2 else 10
    elif has_issue_context:
        rank_score += 70 if page_num <= 2 else 5
    if is_first_page:
        rank_score += 50
    elif page_num == 2:
        rank_score += 10

    rank = (rank_score, -(page_num if page_num > 0 else 999))
    return resolved_issue_date, rank


def select_best_issue_date_record(records: list[dict]) -> tuple[str, dict] | None:
    best_resolution: tuple[str, dict] | None = None
    best_rank: tuple[int, int, int] | None = None

    for record in records:
        resolved = resolve_issue_date_record(record)
        if resolved is None:
            continue
        issue_date, rank = resolved
        if best_rank is None or rank > best_rank:
            best_resolution = (issue_date, record)
            best_rank = rank

    return best_resolution
