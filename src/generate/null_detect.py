"""
Three-tier null/adversarial question detection.

Tier 1: Keyword pre-filter (instant) — concepts foreign to DIFC law
Tier 2: Reranker confidence (free) — low scores = likely unanswerable
Tier 3: LLM verification (part of generation prompt) — handled in parse.py
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Concepts that do NOT exist in DIFC/UAE legal system
# These are adversarial traps designed to trigger hallucination
FOREIGN_CONCEPTS = {
    # US-specific legal concepts
    "jury trial", "jury", "juror", "grand jury",
    "miranda rights", "miranda warning", "miranda",
    "plea bargain", "plea bargaining", "plea deal",
    "parole", "parole board", "parole officer",
    "fifth amendment", "first amendment", "second amendment",
    "habeas corpus",
    "bail bond", "bail bondsman",
    "district attorney", "public defender",
    "supreme court of the united states", "scotus",
    "federal court", "state court",
    # UK-specific (not applicable to DIFC common law system)
    "house of lords", "house of commons",
    "crown prosecution",
    # Criminal law concepts typically not in DIFC civil/commercial
    "felony", "misdemeanor",
    "death penalty", "capital punishment",
    "three strikes", "mandatory minimum",
}

# Regex patterns for faster matching
FOREIGN_PATTERNS = [
    re.compile(r"\bjur(?:y|ies|or)\b", re.IGNORECASE),
    re.compile(r"\bmiranda\b", re.IGNORECASE),
    re.compile(r"\bplea\s+bargain", re.IGNORECASE),
    re.compile(r"\bparole\b", re.IGNORECASE),
    re.compile(r"\bhabeas\s+corpus\b", re.IGNORECASE),
    re.compile(r"\b(?:fifth|first|second)\s+amendment\b", re.IGNORECASE),
    re.compile(r"\bgrand\s+jury\b", re.IGNORECASE),
]


def check_foreign_concepts(question: str) -> bool:
    """
    Tier 1: Check if question mentions concepts foreign to DIFC law.
    Returns True if question is likely adversarial/unanswerable.
    """
    question_lower = question.lower()

    # Check exact phrases
    for concept in FOREIGN_CONCEPTS:
        if concept in question_lower:
            logger.info(f"Foreign concept detected: '{concept}'")
            return True

    # Check regex patterns
    for pattern in FOREIGN_PATTERNS:
        if pattern.search(question):
            logger.info(f"Foreign pattern matched: {pattern.pattern}")
            return True

    return False


def check_low_retrieval_confidence(
    reranked_chunks: list[tuple[dict, float]],
    threshold: float = -5.0,
) -> bool:
    """
    Tier 2: Check if top reranker scores are too low.
    Low scores across all chunks = question likely not answerable from corpus.

    Returns True if confidence is too low (likely null).
    """
    if not reranked_chunks:
        return True

    top_score = reranked_chunks[0][1]
    if top_score < threshold:
        logger.info(f"Low reranker confidence: top score = {top_score:.2f}")
        return True

    return False


def detect_null(
    question: str,
    answer_type: str,
    reranked_chunks: list[tuple[dict, float]],
    reranker_threshold: float = -5.0,
) -> tuple[bool, str]:
    """
    Combined null detection across all tiers.

    Args:
        question: Question text
        answer_type: Expected answer type
        reranked_chunks: Retrieved and reranked chunks with scores
        reranker_threshold: Score threshold for Tier 2

    Returns:
        (is_null, reason) tuple.
        is_null=True means we should return null answer.
    """
    # answer_type "null" = organizers expect null
    if answer_type == "null":
        return True, "answer_type is null"

    # Tier 1: Foreign concept check
    if check_foreign_concepts(question):
        return True, "foreign legal concept detected"

    # Tier 2: Reranker confidence
    if check_low_retrieval_confidence(reranked_chunks, reranker_threshold):
        return True, "low retrieval confidence"

    # Tier 3 is handled in the LLM response (NULL_ANSWER marker)
    # That check happens in parse.py

    return False, ""
