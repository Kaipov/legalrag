from __future__ import annotations

import re
from typing import Iterable

CASE_ID_RE = re.compile(
    r"\b(?P<prefix>CFI|SCT|ENF|CA|ARB|TCD|DEC)\s*(?:[-/]\s*|\s+)?(?P<number>\d{3})\s*[-/]\s*(?P<year>\d{4})\b(?!\s*[-/]\s*\d)",
    re.IGNORECASE,
)


def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = " ".join(str(value or "").split()).strip()
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(normalized)
    return ordered


def normalize_case_id(raw_value: str) -> str:
    match = CASE_ID_RE.search(str(raw_value or ""))
    if not match:
        return " ".join(str(raw_value or "").upper().split())
    prefix = str(match.group("prefix") or "").upper()
    number = str(match.group("number") or "").strip()
    year = str(match.group("year") or "").strip()
    return f"{prefix} {number}/{year}"


def extract_case_ids(text: str) -> list[str]:
    return _dedupe_preserve_order(normalize_case_id(match.group(0)) for match in CASE_ID_RE.finditer(text or ""))
