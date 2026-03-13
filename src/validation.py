"""Validation helpers for local submission checks."""
from __future__ import annotations

import re
from numbers import Number
from typing import Any

from src.constants import NULL_FREE_TEXT_ANSWER

_NULL_LIKE_FREE_TEXT_PREFIXES = (
    NULL_FREE_TEXT_ANSWER,
    "The provided DIFC documents do not contain",
    "The provided DIFC documents do not state",
)


def is_null_like_answer(answer: Any) -> bool:
    """Project-specific null handling for local validation."""
    if answer is None:
        return True
    if not isinstance(answer, str):
        return False
    normalized = answer.strip()
    return any(normalized.startswith(prefix) for prefix in _NULL_LIKE_FREE_TEXT_PREFIXES)


def validate_answer_value(answer: Any, answer_type: str) -> list[str]:
    """Validate the answer shape against the expected answer_type."""
    issues: list[str] = []
    answer_type = str(answer_type or "free_text").lower()

    if answer is None:
        return issues

    if answer_type == "null":
        issues.append("answer_type is null but answer is not null")
        return issues

    if answer_type == "number":
        if isinstance(answer, bool) or not isinstance(answer, Number):
            issues.append("number answer must be int or float")
        return issues

    if answer_type == "boolean":
        if not isinstance(answer, bool):
            issues.append("boolean answer must be true or false")
        return issues

    if answer_type == "name":
        if not isinstance(answer, str) or not answer.strip():
            issues.append("name answer must be a non-empty string")
        return issues

    if answer_type == "names":
        if not isinstance(answer, list) or not answer:
            issues.append("names answer must be a non-empty list of strings")
            return issues
        if not all(isinstance(item, str) and item.strip() for item in answer):
            issues.append("names answer must contain only non-empty strings")
        return issues

    if answer_type == "date":
        if not isinstance(answer, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", answer):
            issues.append("date answer must be YYYY-MM-DD")
        return issues

    if answer_type == "free_text":
        if not isinstance(answer, str) or not answer.strip():
            issues.append("free_text answer must be a non-empty string")
        elif len(answer) > 280:
            issues.append("free_text answer exceeds 280 characters")
        return issues

    return issues


def validate_telemetry_payload(answer_payload: dict[str, Any]) -> list[str]:
    """Validate telemetry fields against the starter-kit spec."""
    issues: list[str] = []
    tel = answer_payload.get("telemetry", {})

    timing = tel.get("timing", {})
    if not timing:
        issues.append("missing timing")
    else:
        ttft = timing.get("ttft_ms", -1)
        tpot = timing.get("tpot_ms", -1)
        total = timing.get("total_time_ms", -1)
        for field_name, value in {
            "ttft_ms": ttft,
            "tpot_ms": tpot,
            "total_time_ms": total,
        }.items():
            if not isinstance(value, Number) or value < 0:
                issues.append(f"invalid {field_name}")
        if isinstance(ttft, Number) and isinstance(total, Number) and ttft > total:
            issues.append(f"ttft_ms ({ttft}) > total_time_ms ({total})")

    usage = tel.get("usage", {})
    if not usage:
        issues.append("missing usage")
    else:
        for field_name in ("input_tokens", "output_tokens"):
            value = usage.get(field_name, -1)
            if not isinstance(value, Number) or value < 0:
                issues.append(f"invalid {field_name}")

    retrieval = tel.get("retrieval", {})
    if "retrieved_chunk_pages" not in retrieval:
        issues.append("missing retrieved_chunk_pages")
        chunks: list[dict[str, Any]] = []
    else:
        chunks = retrieval.get("retrieved_chunk_pages", [])
        if not isinstance(chunks, list):
            issues.append("retrieved_chunk_pages must be a list")
            chunks = []

    answer_value = answer_payload.get("answer")
    if is_null_like_answer(answer_value):
        if chunks:
            issues.append("null answer must have empty retrieved_chunk_pages")
    elif not chunks:
        issues.append("non-null answer but empty retrieved_chunk_pages")

    for chunk in chunks:
        if not isinstance(chunk, dict):
            issues.append("retrieved_chunk_pages entries must be objects")
            continue

        doc_id = chunk.get("doc_id")
        pages = chunk.get("page_numbers", [])
        if not isinstance(doc_id, str) or not doc_id.strip():
            issues.append("retrieved_chunk_pages doc_id must be a non-empty string")
        if not isinstance(pages, list) or not all(isinstance(page, int) and page > 0 for page in pages):
            issues.append("retrieved_chunk_pages page_numbers must be a list of positive integers")

    return issues