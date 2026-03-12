"""
Main RAG pipeline: question -> retrieval -> null check -> generation -> answer + grounding.

Orchestrates all components into a single answer_question() function.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter_kit"))

import tiktoken

from src.config import GENERATION_MODEL, GENERATION_TOP_K
from src.constants import NULL_FREE_TEXT_ANSWER
from src.generate.llm import stream_generate
from src.generate.null_detect import detect_null
from src.generate.parse import parse_model_output
from src.generate.prompts import build_prompt
from src.retrieve.grounding import collect_grounding_pages
from src.retrieve.hybrid import HybridRetriever

from arlc.submission import SubmissionAnswer
from arlc.telemetry import RetrievalRef, Telemetry, TelemetryTimer, TimingMetrics, UsageMetrics

logger = logging.getLogger(__name__)

_tokenizer = None
_GENERATION_TOP_K_BY_TYPE = {
    "number": 2,
    "date": 2,
    "name": 2,
    "names": 3,
}
_GROUNDING_FALLBACK_TOP_K_BY_TYPE = {
    "number": 1,
    "date": 1,
    "name": 1,
    "names": 2,
}


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        try:
            _tokenizer = tiktoken.encoding_for_model(GENERATION_MODEL)
        except KeyError:
            _tokenizer = tiktoken.get_encoding("o200k_base")
    return _tokenizer


def _generation_top_k_for(answer_type: str) -> int:
    """Use narrower prompt context for answer types that are usually single-hop."""
    answer_type = str(answer_type or "free_text").lower()
    capped_top_k = _GENERATION_TOP_K_BY_TYPE.get(answer_type, GENERATION_TOP_K)
    return max(1, min(GENERATION_TOP_K, capped_top_k))


def _select_grounding_chunks(
    answer_type: str,
    generation_chunks: list[tuple[dict, float]],
    cited_source_ids: list[int],
) -> list[tuple[dict, float]]:
    """Prefer chunks explicitly cited by the model, with a conservative fallback."""
    if not generation_chunks:
        return []

    selected_chunks: list[tuple[dict, float]] = []
    seen_indexes: set[int] = set()

    for source_id in cited_source_ids:
        chunk_index = source_id - 1
        if chunk_index < 0 or chunk_index >= len(generation_chunks) or chunk_index in seen_indexes:
            continue
        seen_indexes.add(chunk_index)
        selected_chunks.append(generation_chunks[chunk_index])

    if selected_chunks:
        return selected_chunks

    answer_type = str(answer_type or "free_text").lower()
    fallback_top_k = _GROUNDING_FALLBACK_TOP_K_BY_TYPE.get(answer_type, len(generation_chunks))
    return generation_chunks[: max(1, min(len(generation_chunks), fallback_top_k))]


class RAGPipeline:
    """
    Full RAG pipeline for ARLC competition.

    Usage:
        pipeline = RAGPipeline()
        answer = pipeline.answer_question(question_dict)
    """

    def __init__(
        self,
        grounding_threshold: float = -1.0,
        reranker_null_threshold: float = -5.0,
    ):
        """
        Initialize pipeline. Loads all models and indices.

        Args:
            grounding_threshold: Score threshold for including pages in grounding
            reranker_null_threshold: Score threshold for null detection (Tier 2)
        """
        logger.info("Initializing RAG pipeline...")
        self.retriever = HybridRetriever()
        self.grounding_threshold = grounding_threshold
        self.reranker_null_threshold = reranker_null_threshold
        logger.info("RAG pipeline ready")

    def answer_question(self, question_item: dict) -> SubmissionAnswer:
        """
        Answer a single question end-to-end.

        Args:
            question_item: Dict with 'id', 'question', 'answer_type'

        Returns:
            SubmissionAnswer with answer, telemetry, and grounding
        """
        question_id = question_item["id"]
        question_text = question_item["question"]
        answer_type = question_item.get("answer_type", "free_text")

        tokenizer = _get_tokenizer()
        timer = TelemetryTimer()

        reranked_chunks = self.retriever.retrieve(question_text)

        is_null, null_reason = detect_null(
            question_text,
            answer_type,
            reranked_chunks,
            reranker_threshold=self.reranker_null_threshold,
        )

        if is_null:
            logger.info(f"[{question_id[:8]}] NULL detected: {null_reason}")
            timer.mark_token()
            timing = timer.finish()
            answer_value = NULL_FREE_TEXT_ANSWER if answer_type == "free_text" else None

            return SubmissionAnswer(
                question_id=question_id,
                answer=answer_value,
                telemetry=Telemetry(
                    timing=TimingMetrics(
                        ttft_ms=timing.ttft_ms,
                        tpot_ms=timing.tpot_ms,
                        total_time_ms=timing.total_time_ms,
                    ),
                    retrieval=[],
                    usage=UsageMetrics(input_tokens=0, output_tokens=0),
                    model_name=GENERATION_MODEL,
                ),
            )

        generation_top_k = _generation_top_k_for(answer_type)
        generation_chunks = reranked_chunks[:generation_top_k]
        messages = build_prompt(
            question_text,
            answer_type,
            generation_chunks,
        )

        prompt_text = " ".join(message["content"] for message in messages)
        input_tokens = len(tokenizer.encode(prompt_text))

        response_chunks: list[str] = []
        for token_chunk in stream_generate(messages):
            timer.mark_token()
            response_chunks.append(token_chunk)

        response_text = "".join(response_chunks)
        output_tokens = len(tokenizer.encode(response_text))
        answer_value, cited_source_ids, answer_text = parse_model_output(
            response_text,
            answer_type,
            question_text=question_text,
        )
        is_llm_null = answer_value is None

        grounding_chunks = _select_grounding_chunks(
            answer_type,
            generation_chunks,
            cited_source_ids,
        )
        grounding_refs = collect_grounding_pages(
            grounding_chunks,
            score_threshold=self.grounding_threshold,
            is_null=is_llm_null,
            question_text=question_text,
            answer_text=answer_text,
        )
        retrieval_refs = [
            RetrievalRef(doc_id=ref["doc_id"], page_numbers=ref["page_numbers"])
            for ref in grounding_refs
        ]

        timing = timer.finish()
        telemetry = Telemetry(
            timing=TimingMetrics(
                ttft_ms=timing.ttft_ms,
                tpot_ms=timing.tpot_ms,
                total_time_ms=timing.total_time_ms,
            ),
            retrieval=retrieval_refs,
            usage=UsageMetrics(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
            model_name=GENERATION_MODEL,
        )

        if is_llm_null and answer_type == "free_text":
            answer_value = NULL_FREE_TEXT_ANSWER

        logger.info(
            f"[{question_id[:8]}] type={answer_type} "
            f"ttft={timing.ttft_ms}ms total={timing.total_time_ms}ms "
            f"prompt_chunks={len(generation_chunks)} "
            f"used_sources={cited_source_ids or 'fallback'} "
            f"pages={sum(len(ref.page_numbers) for ref in retrieval_refs)} "
            f"answer={str(answer_value)[:50]}"
        )

        return SubmissionAnswer(
            question_id=question_id,
            answer=answer_value,
            telemetry=telemetry,
        )

