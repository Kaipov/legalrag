"""
Prompt templates for answer generation.

Each answer type gets specific formatting instructions.
Context includes source markers for grounding traceability.
"""
from __future__ import annotations

from src.retrieve.grounding_policy import GroundingIntent

SYSTEM_PROMPT = """You are a legal expert specializing in DIFC (Dubai International Financial Centre) law and regulations.

CRITICAL RULES:
1. Answer ONLY based on the provided context. Never use external legal knowledge.
2. If the context does NOT contain sufficient information to answer the question, respond with exactly: NULL_ANSWER
3. Be precise and cite specific articles, sections, or provisions from the context.
4. For legal questions, accuracy is paramount - do not guess or speculate."""

TYPE_INSTRUCTIONS = {
    "number": (
        "Return ONLY the numeric value (integer or decimal). "
        "No units, no currency symbols, no explanation. Just the number."
    ),
    "boolean": (
        "Return ONLY 'true' or 'false' (lowercase). "
        "No explanation, no qualification."
    ),
    "name": (
        "Return ONLY one exact name, case identifier, claim number, or document name. "
        "If the question asks you to choose between options, return ONLY the winning option exactly as written in the question. "
        "Do not return dates, amounts, both options, or a sentence."
    ),
    "names": (
        "Return a semicolon-separated list of exact names only. "
        "Example: Name One; Name Two; Name Three. "
        "No explanation, no numbering."
    ),
    "date": (
        "Return the date in YYYY-MM-DD format only. "
        "No explanation, no additional text."
    ),
    "free_text": (
        "Answer in one short self-contained sentence by default, and use two short sentences only when the question clearly asks for two distinct facts. "
        "Target 60-180 characters and never exceed 220 characters total. "
        "State the answer directly in a standalone sentence, not as a fragment like 'USD 1,500.' or 'The DIFCA.' "
        "Include every fact the question asks for, but do not add extra details that were not requested. "
        "For outcome or order questions, state the ruling first and mention costs only if the question asks for costs or costs are part of the ruling itself. "
        "Do not restate the question. Do not begin with 'According to the context', 'The context states', or 'The context does not specify'. If the context is insufficient, return NULL_ANSWER."
    ),
}

TYPE_INSTRUCTIONS["null"] = "This question has answer_type 'null'. Return exactly: NULL_ANSWER"

SOURCE_SELECTION_INSTRUCTIONS = """OUTPUT FORMAT:
SOURCES: <comma-separated source numbers you directly used, e.g. 1,2. Use NONE if the answer is NULL_ANSWER.>
ANSWER: <the answer only, following the format instruction above>

Use the minimum number of sources needed to support the answer.
Do not mention source numbers inside the answer text itself."""

FREE_TEXT_POLICY_INSTRUCTIONS = """FREE-TEXT POLICY:
- Prefer one standalone sentence that can be read without the question.
- If the answer is a person, entity, date, amount, obligation, or publication date, write a full sentence rather than a noun phrase.
- If the question asks for multiple requested items, include all of them and nothing extra.
- For result or ruling questions, lead with the result; include costs only when asked or clearly integral to the ruling.
- Do not hedge when the answer is explicit, and do not speculate when it is not explicit.
- If the answer is missing from the context, return NULL_ANSWER rather than saying that the result is unspecified or unclear."""


def _intent_instruction(intent: GroundingIntent | None) -> str:
    if intent is None or intent.kind == "generic":
        return ""
    if intent.kind in {"title_page", "date_of_issue"}:
        return (
            "QUESTION-SPECIFIC FOCUS: This is a page-local question. Prioritize first-page title/header/date-of-issue evidence "
            "over general body text when selecting sources and answering."
        )
    if intent.kind == "last_page":
        return (
            "QUESTION-SPECIFIC FOCUS: This is a page-local question. Prioritize the last page, conclusion, and 'IT IS HEREBY ORDERED THAT' "
            "language over earlier procedural background."
        )
    if intent.kind in {"judge_compare", "party_compare"}:
        return (
            "QUESTION-SPECIFIC FOCUS: This question compares case-file metadata. Prioritize header/caption/title-page details and prefer the "
            "smallest set of documents needed to resolve the comparison."
        )
    return ""


def _free_text_question_policy(question: str) -> str:
    lower = str(question or "").lower()
    policies: list[str] = []

    outcome_markers = (
        "what was the outcome",
        "what was the result",
        "final ruling",
        "how did the court of appeal rule",
        "how did the court rule",
        "what did the court decide",
        "it is hereby ordered that",
        "conclusion section",
        "what was the court's final ruling",
    )
    if any(marker in lower for marker in outcome_markers):
        policies.append(
            "OUTCOME QUESTION RULE: If any source states a ruling such as dismissed, refused, granted, allowed, discharged, restored, proceed to trial, or no order as to costs, state that ruling directly. Do not answer that the outcome or result is unspecified if such ruling language appears in the context. Do not substitute hearing dates, judge names, or procedural background for the ruling."
        )

    if any(marker in lower for marker in ("is there any information about", "plea bargain", "jury decide", "miranda rights")):
        policies.append(
            "ABSTENTION RULE: If the requested fact is absent, prefer NULL_ANSWER rather than a vague sentence about the context being unclear or unspecified."
        )

    return "\n".join(policies)


def _law_scope_question_policy(question: str, answer_type: str) -> str:
    lower = str(question or "").lower()
    normalized_answer_type = str(answer_type or "").lower()
    if normalized_answer_type != "boolean":
        return ""
    if "deal with" not in lower or "difc law no." not in lower:
        return ""
    return (
        "LAW SCOPE RULE: For questions asking whether a named DIFC Law deals with a topic, determine the law's subject matter from its title, scope, and substantive provisions. "
        "Do not answer true based only on passing definitions, cross-references, or mentions of another law or topic inside the named law."
    )


def build_context_block(chunks: list[tuple[dict, float]]) -> str:
    """
    Build the context block from retrieved chunks with source markers.

    Args:
        chunks: List of (chunk_dict, score) from retriever

    Returns:
        Formatted context string with source markers
    """
    blocks = []
    for i, (chunk, _score) in enumerate(chunks):
        doc_id = chunk["doc_id"]
        pages = chunk.get("page_numbers", [])
        section = chunk.get("section_path", "")
        title = chunk.get("doc_title", "")
        text = chunk["text"]

        page_str = ", ".join(str(page) for page in pages)
        header_parts = []
        if title:
            header_parts.append(title)
        if section:
            header_parts.append(section)
        header_parts.append(f"pages {page_str}")
        header_parts.append(f"doc {doc_id[:8]}")

        header = " | ".join(header_parts)
        blocks.append(f"[Source {i + 1}: {header}]\n{text}")

    return "\n\n---\n\n".join(blocks)


def build_prompt(
    question: str,
    answer_type: str,
    chunks: list[tuple[dict, float]],
    max_chunks: int | None = None,
    intent: GroundingIntent | None = None,
    allow_scoped_insufficiency: bool = False,
) -> list[dict]:
    """
    Build the full prompt (messages format) for the LLM.

    Args:
        question: Question text
        answer_type: Expected answer type (number, boolean, name, etc.)
        chunks: Retrieved chunks with scores
        max_chunks: Optional cap on how many retrieved chunks to include
        intent: Optional retrieval/grounding intent for page-local steering

    Returns:
        List of message dicts for OpenAI chat API
    """
    normalized_answer_type = str(answer_type or "free_text").lower()
    instruction = TYPE_INSTRUCTIONS.get(normalized_answer_type, TYPE_INSTRUCTIONS["free_text"])
    selected_chunks = chunks[:max_chunks] if max_chunks is not None else chunks
    context = build_context_block(selected_chunks)
    extra_instruction = _intent_instruction(intent)
    free_text_policy = FREE_TEXT_POLICY_INSTRUCTIONS if normalized_answer_type == "free_text" else ""
    question_policy = _free_text_question_policy(question) if normalized_answer_type == "free_text" else ""
    law_scope_policy = _law_scope_question_policy(question, normalized_answer_type)
    scoped_insufficiency_policy = ""
    if normalized_answer_type == "free_text" and allow_scoped_insufficiency:
        scoped_insufficiency_policy = (
            "SCOPED INSUFFICIENCY RULE: If the sources clearly discuss the target DIFC case, document, application, "
            "or order but do not state the specific requested fact, answer that the provided context does not specify it. "
            "In that situation, cite the supporting sources and do not output NULL_ANSWER. Reserve NULL_ANSWER for facts "
            "that are absent from the provided DIFC context entirely."
        )

    user_message = (
        f"FORMAT INSTRUCTION: {instruction}\n\n"
        f"{SOURCE_SELECTION_INSTRUCTIONS}\n\n"
        f"{free_text_policy}\n\n"
        f"{question_policy}\n\n"
        f"{law_scope_policy}\n\n"
        f"{scoped_insufficiency_policy}\n\n"
        f"{extra_instruction}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "Respond using the exact two-line format above."
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
