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


def parse_model_output(raw_text: str, answer_type: str):
    """
    Parse the raw LLM response into answer value, cited source ids, and answer text.

    Returns:
        (parsed_answer, source_ids, answer_text)
    """
    answer_text = extract_answer_text(raw_text)
    source_ids = extract_source_ids(raw_text)
    return parse_answer(answer_text, answer_type), source_ids, answer_text


def parse_answer(raw_text: str, answer_type: str):
    """
    Parse the raw LLM response into the correct format for submission.

    Args:
        raw_text: Raw text from the LLM
        answer_type: Expected type (number, boolean, name, etc.)

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
        return _parse_name(text)

    if at == "names":
        return _parse_names(text)

    if at == "free_text":
        return _parse_free_text(text)

    # Unknown type, return as-is
    return text


def _parse_number(text: str):
    """Parse numeric answer. Returns float or int."""
    # Remove common non-numeric prefixes/suffixes
    cleaned = text.strip()
    # Remove currency symbols, percent signs, etc.
    cleaned = re.sub(r"[^\d.,\-+eE]", " ", cleaned)
    cleaned = cleaned.strip()

    # Take the first number-like token
    match = re.search(r"[-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?", cleaned)
    if match:
        num_str = match.group(0).replace(",", "")
        try:
            val = float(num_str)
            # Return int if it's a whole number
            if val == int(val) and "." not in num_str and "e" not in num_str.lower():
                return int(val)
            return val
        except (TypeError, ValueError):
            pass

    # Fallback: try the whole text
    try:
        return float(text.replace(",", "."))
    except (TypeError, ValueError):
        logger.warning(f"Failed to parse number from: {text[:100]}")
        return text


def _parse_boolean(text: str):
    """Parse boolean answer."""
    lower = text.lower().strip()
    # Check for true/false at the start
    if lower.startswith("true") or lower.startswith("yes"):
        return True
    if lower.startswith("false") or lower.startswith("no"):
        return False
    # Check if it's somewhere in the text
    if "true" in lower:
        return True
    if "false" in lower:
        return False
    logger.warning(f"Failed to parse boolean from: {text[:100]}")
    return text


def _parse_date(text: str) -> str:
    """Parse date, ensure YYYY-MM-DD format."""
    # Look for YYYY-MM-DD pattern
    match = re.search(r"\d{4}-\d{2}-\d{2}", text)
    if match:
        return match.group(0)

    # Try other common date formats and convert
    # DD/MM/YYYY
    match = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", text)
    if match:
        day, month, year = match.groups()
        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"

    # Month DD, YYYY
    months = {
        "january": "01", "february": "02", "march": "03", "april": "04",
        "may": "05", "june": "06", "july": "07", "august": "08",
        "september": "09", "october": "10", "november": "11", "december": "12",
    }
    match = re.search(r"(\w+)\s+(\d{1,2}),?\s+(\d{4})", text, re.IGNORECASE)
    if match:
        month_name = match.group(1).lower()
        if month_name in months:
            day = match.group(2).zfill(2)
            year = match.group(3)
            return f"{year}-{months[month_name]}-{day}"

    logger.warning(f"Failed to parse date from: {text[:100]}")
    return text.strip()


def _parse_name(text: str) -> str:
    """Parse a single name/entity."""
    # Strip quotes, leading/trailing whitespace
    result = text.strip().strip('"').strip("'").strip()
    # Take only first line if multi-line
    if "\n" in result:
        result = result.split("\n")[0].strip()
    return result


def _parse_names(text: str) -> list[str]:
    """Parse multiple names (semicolon or comma separated)."""
    # Try semicolons first, then commas
    if ";" in text:
        parts = text.split(";")
    else:
        parts = text.split(",")

    names = []
    for part in parts:
        name = part.strip().strip('"').strip("'").strip()
        if name and name.upper() != NULL_MARKER:
            names.append(name)

    return names if names else [text.strip()]


def _parse_free_text(text: str) -> str:
    """Parse free-text answer. Truncate to 280 chars."""
    result = text.strip()
    if len(result) > 280:
        # Truncate at word boundary
        result = result[:277]
        last_space = result.rfind(" ")
        if last_space > 200:
            result = result[:last_space]
        result += "..."
    return result
