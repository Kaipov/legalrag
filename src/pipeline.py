"""
Main RAG pipeline: question -> retrieval -> null check -> generation -> answer + grounding.

Orchestrates all components into a single answer_question() function.
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter_kit"))

import tiktoken

from src.config import GENERATION_MODEL, GENERATION_TOP_K
from src.constants import NULL_FREE_TEXT_ANSWER
from src.generate.llm import stream_generate
from src.generate.null_detect import detect_null
from src.generate.parse import parse_model_output
from src.generate.prompts import build_prompt
from src.retrieve.grounding import collect_grounding_pages
from src.retrieve.grounding_policy import GroundingIntent, detect_grounding_intent
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
_COMPARE_NULL_RELAXED_TYPES = {"boolean", "name", "names"}


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        try:
            _tokenizer = tiktoken.encoding_for_model(GENERATION_MODEL)
        except KeyError:
            _tokenizer = tiktoken.get_encoding("o200k_base")
    return _tokenizer


_AUXILIARY_NULL_QUESTION_RE = re.compile(r"^(is|are|was|were|did|does|do|can|could|has|have|had)\s+(.+)$", re.IGNORECASE)


def _free_text_null_answer(question_text: str) -> str:
    normalized = re.sub(r"\s+", " ", str(question_text or "")).strip().rstrip(" ?")
    if not normalized:
        return NULL_FREE_TEXT_ANSWER

    lower = normalized.lower()
    if lower.startswith("is there any information about "):
        detail = normalized[len("Is there any information about "):].strip()
        if detail:
            return f"The provided DIFC documents do not contain information about {detail}."

    if lower.startswith("what was the plea bargain in "):
        detail = normalized[len("What was the plea bargain in "):].strip()
        if detail:
            return f"The provided DIFC documents do not contain information about any plea bargain in {detail}."

    if lower.startswith("what did the jury decide in "):
        detail = normalized[len("What did the jury decide in "):].strip()
        if detail:
            return f"The provided DIFC documents do not contain information about any jury decision in {detail}."

    if lower.startswith("were the miranda rights properly administered in "):
        detail = normalized[len("Were the Miranda rights properly administered in "):].strip()
        if detail:
            return f"The provided DIFC documents do not contain information showing whether Miranda rights were properly administered in {detail}."

    if lower.startswith("on what date was ") and lower.endswith(" enacted"):
        detail = normalized[len("On what date was "):-len(" enacted")].strip()
        if detail:
            return f"The provided DIFC documents do not state the enactment date of {detail}."

    auxiliary_match = _AUXILIARY_NULL_QUESTION_RE.match(normalized)
    if auxiliary_match:
        detail = auxiliary_match.group(2).strip()
        if detail:
            detail = detail[0].lower() + detail[1:] if len(detail) > 1 else detail.lower()
            return f"The provided DIFC documents do not contain information showing whether {detail}."

    return NULL_FREE_TEXT_ANSWER


def _chunk_identity(chunk_with_score: tuple[dict, float]) -> str:
    chunk = chunk_with_score[0]
    chunk_id = str(chunk.get("chunk_id") or "").strip()
    if chunk_id:
        return chunk_id
    return "|".join(
        [
            str(chunk.get("doc_id") or ""),
            str(chunk.get("section_path") or ""),
            ",".join(str(page) for page in chunk.get("page_numbers", [])),
        ]
    )


def _chunk_case_search_blob(chunk: dict[str, Any]) -> str:
    return " ".join(
        str(part or "")
        for part in (
            chunk.get("doc_title"),
            chunk.get("section_path"),
            str(chunk.get("text") or "")[:1500],
        )
    ).upper()


def _chunk_matches_case_id(chunk: dict[str, Any], case_id: str) -> bool:
    if not case_id:
        return False
    return case_id.upper() in _chunk_case_search_blob(chunk)


def _chunk_first_page_rank(chunk: dict[str, Any]) -> int:
    pages = sorted(
        int(page)
        for page in chunk.get("page_numbers", [])
        if isinstance(page, int) and page > 0
    )
    if not pages:
        return 99

    first_page = pages[0]
    if first_page <= 1:
        return 0
    if first_page == 2:
        return 1
    if first_page == 3:
        return 2
    return 3


def _select_compare_case_coverage_chunks(
    retrieved_chunks: list[tuple[dict, float]],
    case_ids: tuple[str, ...],
    top_k: int,
) -> list[tuple[dict, float]]:
    if top_k <= 0 or not retrieved_chunks or not case_ids:
        return []

    selected: list[tuple[dict, float]] = []
    seen_chunk_keys: set[str] = set()

    for case_id in case_ids:
        best_choice: tuple[tuple[int, int], tuple[dict, float], str] | None = None
        for index, chunk_with_score in enumerate(retrieved_chunks):
            chunk = chunk_with_score[0]
            chunk_key = _chunk_identity(chunk_with_score)
            if chunk_key in seen_chunk_keys or not _chunk_matches_case_id(chunk, case_id):
                continue

            candidate_rank = (_chunk_first_page_rank(chunk), index)
            if best_choice is None or candidate_rank < best_choice[0]:
                best_choice = (candidate_rank, chunk_with_score, chunk_key)

        if best_choice is None:
            continue

        selected.append(best_choice[1])
        seen_chunk_keys.add(best_choice[2])
        if len(selected) >= top_k:
            break

    return selected


def _has_compare_first_page_coverage(
    retrieved_chunks: list[tuple[dict, float]],
    intent: GroundingIntent | None,
) -> bool:
    if intent is None or not intent.is_compare or len(intent.case_ids) < 2:
        return False

    coverage_chunks = _select_compare_case_coverage_chunks(
        retrieved_chunks,
        intent.case_ids,
        len(intent.case_ids),
    )
    if len(coverage_chunks) < len(intent.case_ids):
        return False

    covered_case_ids: set[str] = set()
    for chunk_with_score in coverage_chunks:
        chunk = chunk_with_score[0]
        if _chunk_first_page_rank(chunk) > 1:
            continue
        for case_id in intent.case_ids:
            if _chunk_matches_case_id(chunk, case_id):
                covered_case_ids.add(case_id)

    return len(covered_case_ids) >= len(intent.case_ids)


def _should_override_compare_null(
    answer_type: str,
    retrieved_chunks: list[tuple[dict, float]],
    intent: GroundingIntent | None,
) -> bool:
    answer_type = str(answer_type or "").lower()
    if answer_type not in _COMPARE_NULL_RELAXED_TYPES:
        return False
    return _has_compare_first_page_coverage(retrieved_chunks, intent)


def _generation_top_k_for(answer_type: str, intent: GroundingIntent | None = None) -> int:
    """Use narrower prompt context for answer types that are usually single-hop."""
    answer_type = str(answer_type or "free_text").lower()
    capped_top_k = _GENERATION_TOP_K_BY_TYPE.get(answer_type, GENERATION_TOP_K)
    if intent is not None and intent.generation_top_k is not None:
        capped_top_k = max(capped_top_k, min(GENERATION_TOP_K, intent.generation_top_k))
    return max(1, min(GENERATION_TOP_K, capped_top_k))


def _select_generation_chunks(
    retrieved_chunks: list[tuple[dict, float]],
    top_k: int,
    intent: GroundingIntent | None = None,
    disable_unique_doc_preference: bool = False,
) -> list[tuple[dict, float]]:
    if top_k <= 0 or not retrieved_chunks:
        return []

    selected: list[tuple[dict, float]] = []
    seen_chunks: set[str] = set()
    seen_docs: set[str] = set()

    if intent is not None and intent.is_compare and intent.case_ids:
        for chunk_with_score in _select_compare_case_coverage_chunks(retrieved_chunks, intent.case_ids, top_k):
            chunk = chunk_with_score[0]
            chunk_key = _chunk_identity(chunk_with_score)
            if chunk_key in seen_chunks:
                continue
            selected.append(chunk_with_score)
            seen_chunks.add(chunk_key)
            doc_id = str(chunk.get("doc_id") or "").strip()
            if doc_id:
                seen_docs.add(doc_id)
            if len(selected) >= top_k:
                return selected

    if intent is None or disable_unique_doc_preference or not intent.prefer_unique_docs:
        for chunk_with_score in retrieved_chunks:
            chunk_key = _chunk_identity(chunk_with_score)
            if chunk_key in seen_chunks:
                continue
            selected.append(chunk_with_score)
            seen_chunks.add(chunk_key)
            if len(selected) >= top_k:
                break
        return selected

    for chunk_with_score in retrieved_chunks:
        chunk = chunk_with_score[0]
        doc_id = str(chunk.get("doc_id") or "").strip()
        chunk_key = _chunk_identity(chunk_with_score)
        if chunk_key in seen_chunks:
            continue
        if doc_id and doc_id not in seen_docs:
            selected.append(chunk_with_score)
            seen_docs.add(doc_id)
            seen_chunks.add(chunk_key)
            if len(selected) >= top_k:
                return selected

    for chunk_with_score in retrieved_chunks:
        chunk_key = _chunk_identity(chunk_with_score)
        if chunk_key in seen_chunks:
            continue
        selected.append(chunk_with_score)
        seen_chunks.add(chunk_key)
        if len(selected) >= top_k:
            break

    return selected


def _generation_doc_ids(generation_chunks: list[tuple[dict, float]]) -> set[str]:
    doc_ids: set[str] = set()
    for chunk_with_score in generation_chunks:
        chunk = chunk_with_score[0]
        doc_id = str(chunk.get("doc_id") or "").strip()
        if doc_id:
            doc_ids.add(doc_id)
    return doc_ids


def _select_grounding_chunks(
    answer_type: str,
    generation_chunks: list[tuple[dict, float]],
    cited_source_ids: list[int],
    intent: GroundingIntent | None = None,
    disable_unique_doc_preference: bool = False,
) -> list[tuple[dict, float]]:
    """Prefer chunks explicitly cited by the model, with a conservative fallback."""
    if not generation_chunks:
        return []

    selected_chunks: list[tuple[dict, float]] = []
    seen_indexes: set[int] = set()
    seen_chunk_keys: set[str] = set()

    for source_id in cited_source_ids:
        chunk_index = source_id - 1
        if chunk_index < 0 or chunk_index >= len(generation_chunks) or chunk_index in seen_indexes:
            continue
        seen_indexes.add(chunk_index)
        chunk_with_score = generation_chunks[chunk_index]
        selected_chunks.append(chunk_with_score)
        seen_chunk_keys.add(_chunk_identity(chunk_with_score))

    answer_type = str(answer_type or "free_text").lower()
    fallback_top_k = _GROUNDING_FALLBACK_TOP_K_BY_TYPE.get(answer_type, len(generation_chunks))
    if intent is not None and intent.grounding_chunk_top_k is not None:
        fallback_top_k = max(fallback_top_k, min(len(generation_chunks), intent.grounding_chunk_top_k))
    fallback_top_k = max(1, min(len(generation_chunks), fallback_top_k))

    if selected_chunks:
        if intent is None or intent.kind == "generic":
            return selected_chunks
        if len(selected_chunks) >= fallback_top_k:
            return selected_chunks[:fallback_top_k]

        for chunk_with_score in _select_generation_chunks(
            generation_chunks,
            fallback_top_k,
            intent=intent,
            disable_unique_doc_preference=disable_unique_doc_preference,
        ):
            chunk_key = _chunk_identity(chunk_with_score)
            if chunk_key in seen_chunk_keys:
                continue
            selected_chunks.append(chunk_with_score)
            seen_chunk_keys.add(chunk_key)
            if len(selected_chunks) >= fallback_top_k:
                break
        return selected_chunks

    return _select_generation_chunks(
        generation_chunks,
        fallback_top_k,
        intent=intent,
        disable_unique_doc_preference=disable_unique_doc_preference,
    )


def _run_generation_pass(
    question_text: str,
    answer_type: str,
    reranked_chunks: list[tuple[dict, float]],
    tokenizer,
    timer: TelemetryTimer,
    grounding_threshold: float,
    intent: GroundingIntent | None = None,
    disable_unique_doc_preference: bool = False,
) -> dict[str, Any]:
    generation_top_k = _generation_top_k_for(answer_type, intent=intent)
    generation_chunks = _select_generation_chunks(
        reranked_chunks,
        generation_top_k,
        intent=intent,
        disable_unique_doc_preference=disable_unique_doc_preference,
    )
    messages = build_prompt(
        question_text,
        answer_type,
        generation_chunks,
        intent=intent,
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
        intent=intent,
        disable_unique_doc_preference=disable_unique_doc_preference,
    )
    grounding_refs = collect_grounding_pages(
        grounding_chunks,
        score_threshold=grounding_threshold,
        is_null=is_llm_null,
        question_text=question_text,
        answer_text=answer_text,
        intent=intent,
        allowed_doc_ids=_generation_doc_ids(generation_chunks),
    )
    retrieval_refs = [
        RetrievalRef(doc_id=ref["doc_id"], page_numbers=ref["page_numbers"])
        for ref in grounding_refs
    ]

    return {
        "answer_value": answer_value,
        "answer_text": answer_text,
        "cited_source_ids": cited_source_ids,
        "generation_chunks": generation_chunks,
        "grounding_refs": grounding_refs,
        "retrieval_refs": retrieval_refs,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "is_llm_null": is_llm_null,
    }


def _should_retry_compare_without_intent(
    intent: GroundingIntent | None,
    is_llm_null: bool,
    grounding_refs: list[dict[str, Any]],
) -> bool:
    return bool(intent and intent.is_compare and (is_llm_null or not grounding_refs))


def _should_use_fallback_attempt(
    primary_attempt: dict[str, Any],
    fallback_attempt: dict[str, Any],
) -> bool:
    if fallback_attempt["is_llm_null"]:
        return False
    if primary_attempt["is_llm_null"]:
        return True
    if not primary_attempt["grounding_refs"] and fallback_attempt["grounding_refs"]:
        return True
    return False


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
        active_intent = detect_grounding_intent(question_text, answer_type)

        tokenizer = _get_tokenizer()
        timer = TelemetryTimer()

        reranked_chunks = self.retriever.retrieve(question_text, intent=active_intent)
        generic_reranked_chunks: list[tuple[dict, float]] | None = None
        compare_bias_disabled = False

        is_null, null_reason = detect_null(
            question_text,
            answer_type,
            reranked_chunks,
            reranker_threshold=self.reranker_null_threshold,
        )

        if active_intent is not None and active_intent.is_compare and is_null:
            logger.info(
                f"[{question_id[:8]}] compare fallback after null detection; retrying retrieval without intent bias"
            )
            generic_reranked_chunks = self.retriever.retrieve(question_text, intent=None)
            fallback_is_null, fallback_reason = detect_null(
                question_text,
                answer_type,
                generic_reranked_chunks,
                reranker_threshold=self.reranker_null_threshold,
            )
            if _should_override_compare_null(answer_type, generic_reranked_chunks, active_intent):
                reranked_chunks = generic_reranked_chunks
                is_null = False
                null_reason = None
                compare_bias_disabled = True
                logger.info(
                    f"[{question_id[:8]}] compare null override: generic retrieval covers first-page evidence for all case ids"
                )
            elif not fallback_is_null:
                reranked_chunks = generic_reranked_chunks
                is_null = False
                null_reason = None
                compare_bias_disabled = True
            else:
                null_reason = fallback_reason or null_reason

        if is_null:
            logger.info(f"[{question_id[:8]}] NULL detected: {null_reason}")
            timer.mark_token()
            timing = timer.finish()
            answer_value = _free_text_null_answer(question_text) if answer_type == "free_text" else None

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

        attempt = _run_generation_pass(
            question_text,
            answer_type,
            reranked_chunks,
            tokenizer,
            timer,
            self.grounding_threshold,
            intent=active_intent,
            disable_unique_doc_preference=compare_bias_disabled,
        )
        input_tokens = int(attempt["input_tokens"])
        output_tokens = int(attempt["output_tokens"])

        if (
            _should_retry_compare_without_intent(active_intent, attempt["is_llm_null"], attempt["grounding_refs"])
            and not compare_bias_disabled
        ):
            logger.info(
                f"[{question_id[:8]}] compare fallback after answer/grounding; retrying retrieval without intent bias"
            )
            if generic_reranked_chunks is None:
                generic_reranked_chunks = self.retriever.retrieve(question_text, intent=None)
            fallback_attempt = _run_generation_pass(
                question_text,
                answer_type,
                generic_reranked_chunks,
                tokenizer,
                timer,
                self.grounding_threshold,
                intent=active_intent,
                disable_unique_doc_preference=True,
            )
            input_tokens += int(fallback_attempt["input_tokens"])
            output_tokens += int(fallback_attempt["output_tokens"])
            if _should_use_fallback_attempt(attempt, fallback_attempt):
                attempt = fallback_attempt
                compare_bias_disabled = True

        answer_value = attempt["answer_value"]
        answer_text = str(attempt["answer_text"])
        is_llm_null = bool(attempt["is_llm_null"])
        generation_chunks = list(attempt["generation_chunks"])
        retrieval_refs = list(attempt["retrieval_refs"])
        cited_source_ids = list(attempt["cited_source_ids"])

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
            answer_value = _free_text_null_answer(question_text)

        retrieval_mode = "compare_fallback" if compare_bias_disabled else "intent_bias"
        if active_intent is None or active_intent.kind == "generic":
            retrieval_mode = "generic"

        logger.info(
            f"[{question_id[:8]}] type={answer_type} intent={(active_intent.kind if active_intent else 'generic')} "
            f"retrieval_mode={retrieval_mode} "
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
