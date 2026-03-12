"""
Prompt templates for answer generation.

Each answer type gets specific formatting instructions.
Context includes source markers for grounding traceability.
"""
from __future__ import annotations

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
        "Answer in 1-3 concise sentences (maximum 280 characters total). "
        "Ground every statement in the provided context. "
        "Express appropriate uncertainty where the context is ambiguous. "
        "Be clear, relevant, and directly address the question."
    ),
}

TYPE_INSTRUCTIONS["null"] = "This question has answer_type 'null'. Return exactly: NULL_ANSWER"

SOURCE_SELECTION_INSTRUCTIONS = """OUTPUT FORMAT:
SOURCES: <comma-separated source numbers you directly used, e.g. 1,2. Use NONE if the answer is NULL_ANSWER.>
ANSWER: <the answer only, following the format instruction above>

Use the minimum number of sources needed to support the answer.
Do not mention source numbers inside the answer text itself."""


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
) -> list[dict]:
    """
    Build the full prompt (messages format) for the LLM.

    Args:
        question: Question text
        answer_type: Expected answer type (number, boolean, name, etc.)
        chunks: Retrieved chunks with scores
        max_chunks: Optional cap on how many retrieved chunks to include

    Returns:
        List of message dicts for OpenAI chat API
    """
    instruction = TYPE_INSTRUCTIONS.get(answer_type, TYPE_INSTRUCTIONS["free_text"])
    selected_chunks = chunks[:max_chunks] if max_chunks is not None else chunks
    context = build_context_block(selected_chunks)

    user_message = (
        f"FORMAT INSTRUCTION: {instruction}\n\n"
        f"{SOURCE_SELECTION_INSTRUCTIONS}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "Respond using the exact two-line format above."
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
