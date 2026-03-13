"""
Answer parsing and validation per answer type.

Based on logic from starter kit baseline examples.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Marker for null/unanswerable detection
NULL_MARKER = "NULL_ANSWER"
_SOURCE_LINE_RE = re.compile(r"^\s*SOURCES?\s*:\s*(.*?)\s*$", re.IGNORECASE | re.MULTILINE)
_ANSWER_PREFIX_RE = re.compile(r"^\s*ANSWER\s*:\s*", re.IGNORECASE | re.MULTILINE)
_CASE_CANDIDATE_RE = re.compile(r"\b(?:CFI|SCT|ENF|CA|ARB|TCD|DEC)\s*\d{3}/\d{4}\b", re.IGNORECASE)
_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?")
_ISO_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})(?:\D+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?)?", re.IGNORECASE)
_DMY_DATE_RE = re.compile(
    r"(\d{1,2})\s+"
    r"(january|february|march|april|may|june|july|august|september|october|november|december)"
    r"\s+(\d{4})(?:\D+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?)?",
    re.IGNORECASE,
)
_MDY_DATE_RE = re.compile(
    r"(january|february|march|april|may|june|july|august|september|october|november|december)"
    r"\s+(\d{1,2}),?\s+(\d{4})(?:\D+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?)?",
    re.IGNORECASE,
)
_FREE_TEXT_LEADING_BOILERPLATE_RE = re.compile(
    r"^(?:"
    r"according to the (?:provided )?context|"
    r"according to the provided documents|"
    r"based on the (?:provided )?(?:documents|context)|"
    r"the context (?:states|shows|indicates)(?: that)?|"
    r"the provided (?:documents|difc documents) (?:state|show|indicate)(?: that)?|"
    r"the documents (?:state|show|indicate)(?: that)?|"
    r"the answer is"
    r")\s*[:,]?\s*",
    re.IGNORECASE,
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
    return re.sub(r"\s+", " ", text or "").strip()


def extract_source_ids(raw_text: str) -> list[int]:
    """Extract 1-based source ids from the model response."""
    text = (raw_text or "").strip()
    match = _SOURCE_LINE_RE.search(text)
    raw_ids: list[str]

    if match:
        source_blob = match.group(1).strip()
        if not source_blob or source_blob.upper() in {"NONE", "NULL", "N/A"}:
            return []
        raw_ids = re.findall(r"\d+", source_blob)
    else:
        raw_ids = re.findall(r"\bsource\s*(\d+)\b", text, re.IGNORECASE)

    seen: set[int] = set()
    source_ids: list[int] = []
    for raw_id in raw_ids:
        source_id = int(raw_id)
        if source_id <= 0 or source_id in seen:
            continue
        seen.add(source_id)
        source_ids.append(source_id)

    return source_ids


def extract_answer_text(raw_text: str) -> str:
    """Strip SOURCES/ANSWER scaffolding and return only the answer body."""
    text = (raw_text or "").strip()
    answer_match = _ANSWER_PREFIX_RE.search(text)
    if answer_match:
        return text[answer_match.end():].strip()

    lines: list[str] = []
    for line in text.splitlines():
        if _SOURCE_LINE_RE.match(line):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _extract_case_candidates(question_text: str) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for match in _CASE_CANDIDATE_RE.finditer(question_text or ""):
        candidate = _normalize_space(match.group(0).upper())
        if candidate in seen:
            continue
        seen.add(candidate)
        candidates.append(candidate)
    return candidates


def _select_case_candidate_from_text(text: str, candidates: list[str]) -> str | None:
    text_upper = (text or "").upper()
    matches: list[tuple[int, str]] = []
    for candidate in candidates:
        index = text_upper.find(candidate)
        if index >= 0:
            matches.append((index, candidate))
    if not matches:
        return None
    matches.sort(key=lambda item: item[0])
    return matches[0][1]


def _comparison_mode(question_text: str) -> str | None:
    lower = (question_text or "").lower()
    if any(phrase in lower for phrase in (
        "higher monetary claim",
        "higher claim",
        "higher amount",
        "greater monetary claim",
    )):
        return "max_number"
    if any(phrase in lower for phrase in (
        "earlier issue date",
        "earlier date of issue",
        "which was issued first",
        "issued earlier",
        "issued first",
        "earlier",
    )):
        return "min_datetime"
    return None


def _parse_hour(hour_text: str | None, minute_text: str | None, meridiem_text: str | None) -> tuple[int, int]:
    if hour_text is None:
        return 0, 0

    hour = int(hour_text)
    minute = int(minute_text or "0")
    if meridiem_text:
        meridiem = meridiem_text.lower()
        hour %= 12
        if meridiem == "pm":
            hour += 12
    return hour, minute


def _extract_datetime_sort_key(text: str) -> tuple[int, int, int, int, int] | None:
    normalized = _normalize_space(text)

    match = _ISO_DATE_RE.search(normalized)
    if match:
        year, month, day, hour_text, minute_text, meridiem_text = match.groups()
        hour, minute = _parse_hour(hour_text, minute_text, meridiem_text)
        return int(year), int(month), int(day), hour, minute

    match = _DMY_DATE_RE.search(normalized)
    if match:
        day, month_name, year, hour_text, minute_text, meridiem_text = match.groups()
        hour, minute = _parse_hour(hour_text, minute_text, meridiem_text)
        return int(year), _MONTHS[month_name.lower()], int(day), hour, minute

    match = _MDY_DATE_RE.search(normalized)
    if match:
        month_name, day, year, hour_text, minute_text, meridiem_text = match.groups()
        hour, minute = _parse_hour(hour_text, minute_text, meridiem_text)
        return int(year), _MONTHS[month_name.lower()], int(day), hour, minute

    return None


def _extract_numeric_sort_key(text: str) -> float | None:
    values: list[float] = []
    for match in _NUMBER_RE.findall(text or ""):
        try:
            values.append(float(match.replace(",", "")))
        except ValueError:
            continue
    if not values:
        return None
    return max(values)


def _infer_case_candidate_from_segments(text: str, question_text: str, candidates: list[str]) -> str | None:
    if len(candidates) != 2:
        return None

    mode = _comparison_mode(question_text)
    if mode is None:
        return None

    segments = [_normalize_space(segment) for segment in re.split(r"\s*;\s*|\n+", text or "") if _normalize_space(segment)]
    if len(segments) != 2:
        return None

    if mode == "min_datetime":
        values = [_extract_datetime_sort_key(segment) for segment in segments]
        if any(value is None for value in values) or values[0] == values[1]:
            return None
        return candidates[0] if values[0] < values[1] else candidates[1]

    if mode == "max_number":
        values = [_extract_numeric_sort_key(segment) for segment in segments]
        if any(value is None for value in values) or values[0] == values[1]:
            return None
        return candidates[0] if values[0] > values[1] else candidates[1]

    return None


def parse_model_output(raw_text: str, answer_type: str, question_text: str = ""):
    """
    Parse the raw LLM response into answer value, cited source ids, and answer text.

    Returns:
        (parsed_answer, source_ids, answer_text)
    """
    answer_text = extract_answer_text(raw_text)
    source_ids = extract_source_ids(raw_text)
    return parse_answer(answer_text, answer_type, question_text=question_text), source_ids, answer_text


def parse_answer(raw_text: str, answer_type: str, question_text: str = ""):
    """
    Parse the raw LLM response into the correct format for submission.

    Args:
        raw_text: Raw text from the LLM
        answer_type: Expected type (number, boolean, name, etc.)
        question_text: Original question text for targeted cleanup heuristics

    Returns:
        Parsed answer value (int/float/bool/str/list[str]/None)
    """
    text = extract_answer_text(raw_text)
    at = str(answer_type or "free_text").lower()

    # Check for null marker first (any type can be null)
    if NULL_MARKER in text.upper():
        return None

    if at == "null":
        return None

    if at == "number":
        return _parse_number(text)

    if at == "boolean":
        return _parse_boolean(text)

    if at == "date":
        return _parse_date(text)

    if at == "name":
        return _parse_name(text, question_text=question_text)

    if at == "names":
        return _parse_names(text)

    if at == "free_text":
        return _parse_free_text(text)

    # Unknown type, return as-is
    return text


def _parse_number(text: str):
    """Parse numeric answer. Returns float or int."""
    cleaned = text.strip()
    cleaned = re.sub(r"[^\d.,\-+eE]", " ", cleaned)
    cleaned = cleaned.strip()

    match = re.search(r"[-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?", cleaned)
    if match:
        num_str = match.group(0).replace(",", "")
        try:
            val = float(num_str)
            if val == int(val) and "." not in num_str and "e" not in num_str.lower():
                return int(val)
            return val
        except (TypeError, ValueError):
            pass

    try:
        return float(text.replace(",", "."))
    except (TypeError, ValueError):
        logger.warning(f"Failed to parse number from: {text[:100]}")
        return text


def _parse_boolean(text: str):
    """Parse boolean answer."""
    lower = text.lower().strip()
    if lower.startswith("true") or lower.startswith("yes"):
        return True
    if lower.startswith("false") or lower.startswith("no"):
        return False
    if "true" in lower:
        return True
    if "false" in lower:
        return False
    logger.warning(f"Failed to parse boolean from: {text[:100]}")
    return text


def _parse_date(text: str) -> str:
    """Parse date, ensure YYYY-MM-DD format."""
    match = re.search(r"\d{4}-\d{2}-\d{2}", text)
    if match:
        return match.group(0)

    match = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", text)
    if match:
        day, month, year = match.groups()
        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"

    match = re.search(r"(\w+)\s+(\d{1,2}),?\s+(\d{4})", text, re.IGNORECASE)
    if match:
        month_name = match.group(1).lower()
        if month_name in _MONTHS:
            day = match.group(2).zfill(2)
            year = match.group(3)
            return f"{year}-{str(_MONTHS[month_name]).zfill(2)}-{day}"

    logger.warning(f"Failed to parse date from: {text[:100]}")
    return text.strip()


def _parse_name(text: str, question_text: str = "") -> str:
    """Parse a single name/entity or normalize a pairwise case comparison answer."""
    result = _normalize_space(text.strip().strip('"').strip("'").strip())
    if not result:
        return result

    candidates = _extract_case_candidates(question_text)
    selected_candidate = _select_case_candidate_from_text(result, candidates)
    if selected_candidate:
        return selected_candidate

    inferred_candidate = _infer_case_candidate_from_segments(result, question_text, candidates)
    if inferred_candidate:
        return inferred_candidate

    return result.split("\n", 1)[0].strip()


def _parse_names(text: str) -> list[str]:
    """Parse multiple names (semicolon or comma separated)."""
    if ";" in text:
        parts = text.split(";")
    else:
        parts = text.split(",")

    names = []
    seen: set[str] = set()
    for part in parts:
        name = part.strip().strip('"').strip("'").strip()
        if not name or name.upper() == NULL_MARKER or name in seen:
            continue
        seen.add(name)
        names.append(name)

    return names if names else [text.strip()]


def _parse_free_text(text: str) -> str:
    """Parse free-text answer. Normalize style and truncate to 280 chars."""
    result = _normalize_space(text)
    while result:
        cleaned = _FREE_TEXT_LEADING_BOILERPLATE_RE.sub("", result, count=1).strip()
        if cleaned == result:
            break
        result = cleaned
    if len(result) > 280:
        result = result[:277]
        last_space = result.rfind(" ")
        if last_space > 200:
            result = result[:last_space]
        result += "..."
    return result
