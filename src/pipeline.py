"""
Main RAG pipeline: question в†’ retrieval в†’ null check в†’ generation в†’ answer + grounding.

Orchestrates all components into a single answer_question() function.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add starter_kit to path for arlc imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter_kit"))

import tiktoken

from src.config import GENERATION_MODEL
from src.constants import NULL_FREE_TEXT_ANSWER
from src.retrieve.hybrid import HybridRetriever
from src.retrieve.grounding import collect_grounding_pages
from src.generate.prompts import build_prompt
from src.generate.llm import stream_generate
from src.generate.parse import parse_answer
from src.generate.null_detect import detect_null

from arlc.telemetry import TelemetryTimer, TimingMetrics, UsageMetrics, RetrievalRef, Telemetry
from arlc.submission import SubmissionAnswer

logger = logging.getLogger(__name__)

# Token counter
_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = tiktoken.encoding_for_model("gpt-4o")
    return _tokenizer


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
            grounding_threshold: Reranker score threshold for including pages in grounding
            reranker_null_threshold: Reranker score threshold for null detection (Tier 2)
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

        # --- Start TTFT timer ---
        timer = TelemetryTimer()

        # --- Step 1: Retrieve ---
        reranked_chunks = self.retriever.retrieve(question_text)

        # --- Step 2: Null detection (Tier 1 + 2) ---
        is_null, null_reason = detect_null(
            question_text, answer_type, reranked_chunks,
            reranker_threshold=self.reranker_null_threshold,
        )

        if is_null:
            logger.info(f"[{question_id[:8]}] NULL detected: {null_reason}")
            timer.mark_token()
            timing = timer.finish()

            # For free_text null, give a natural language response
            if answer_type == "free_text":
                answer_value = NULL_FREE_TEXT_ANSWER
            else:
                answer_value = None

            return SubmissionAnswer(
                question_id=question_id,
                answer=answer_value,
                telemetry=Telemetry(
                    timing=TimingMetrics(
                        ttft_ms=timing.ttft_ms,
                        tpot_ms=timing.tpot_ms,
                        total_time_ms=timing.total_time_ms,
                    ),
                    retrieval=[],  # Empty for null
                    usage=UsageMetrics(input_tokens=0, output_tokens=0),
                    model_name=GENERATION_MODEL,
                ),
            )

        # --- Step 3: Generate answer ---
        messages = build_prompt(question_text, answer_type, reranked_chunks)

        # Count input tokens
        prompt_text = " ".join(m["content"] for m in messages)
        input_tokens = len(tokenizer.encode(prompt_text))

        # Stream response
        response_chunks: list[str] = []
        for token_chunk in stream_generate(messages):
            timer.mark_token()
            response_chunks.append(token_chunk)

        response_text = "".join(response_chunks)
        output_tokens = len(tokenizer.encode(response_text))

        # --- Step 4: Parse answer ---
        answer_value = parse_answer(response_text, answer_type)

        # Tier 3 null check (LLM said NULL_ANSWER)
        is_llm_null = answer_value is None

        # --- Step 5: Collect grounding ---
        grounding_refs = collect_grounding_pages(
            reranked_chunks,
            score_threshold=self.grounding_threshold,
            is_null=is_llm_null,
        )

        # Convert to RetrievalRef objects
        retrieval_refs = [
            RetrievalRef(doc_id=ref["doc_id"], page_numbers=ref["page_numbers"])
            for ref in grounding_refs
        ]

        # --- Step 6: Build telemetry ---
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

        # For free_text null from LLM
        if is_llm_null and answer_type == "free_text":
            answer_value = NULL_FREE_TEXT_ANSWER

        logger.info(
            f"[{question_id[:8]}] type={answer_type} "
            f"ttft={timing.ttft_ms}ms total={timing.total_time_ms}ms "
            f"pages={sum(len(r.page_numbers) for r in retrieval_refs)} "
            f"answer={str(answer_value)[:50]}"
        )

        return SubmissionAnswer(
            question_id=question_id,
            answer=answer_value,
            telemetry=telemetry,
        )
