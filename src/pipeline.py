"""
Main RAG pipeline: question -> retrieval -> null check -> generation -> answer + grounding.

Orchestrates all components into a single answer_question() function.
"""
from __future__ import annotations

from dataclasses import replace
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
from src.generate.parse import parse_answer, parse_model_output
from src.generate.prompts import build_prompt
from src.resolve.article import select_article_evidence_pages
from src.resolve.resolver import try_resolve_question
from src.retrieve.grounding import collect_grounding_pages
from src.retrieve.grounding_utils import extract_question_anchors
from src.retrieve.grounding_policy import GroundingIntent, detect_grounding_intent
from src.retrieve.lexical import tokenize_legal_text
from src.retrieve.question_plan import QuestionPlan, build_question_plan
from src.retrieve.hybrid import HybridRetriever

from arlc.submission import SubmissionAnswer
from arlc.telemetry import RetrievalRef, Telemetry, TelemetryTimer, TimingMetrics, UsageMetrics, normalize_retrieved_pages

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
_STRUCTURED_ANSWER_TYPES = {"number", "date", "boolean", "name", "names"}
_GENERATION_SELECTION_STOPWORDS = {
    "who",
    "what",
    "which",
    "identify",
    "under",
    "according",
    "case",
    "cases",
    "court",
    "law",
    "difc",
    "document",
    "documents",
    "specific",
    "referenced",
    "stated",
    "listed",
    "higher",
    "earlier",
}
_MONEY_SIGNAL_RE = re.compile(
    r"\b(?:aed|claim value|claim amount|higher monetary claim|seeking payment|assessed and fixed in the amount)\b",
    re.IGNORECASE,
)
_ISSUE_DATE_SIGNAL_RE = re.compile(r"\b(?:date of issue|issued by|issued on)\b", re.IGNORECASE)
_CLAIM_NUMBER_SIGNAL_RE = re.compile(r"\b(?:claim no\.?|claim number)\b", re.IGNORECASE)
_PARTY_SIGNAL_RE = re.compile(r"\b(?:claimant|defendant|applicant|respondent|party|parties)\b", re.IGNORECASE)
_JUDGE_SIGNAL_RE = re.compile(r"\b(?:judge|justice|before)\b", re.IGNORECASE)
_ADMINISTRATION_SIGNAL_RE = re.compile(r"\b(?:administration of this law|administered by|administer)\b", re.IGNORECASE)


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        try:
            _tokenizer = tiktoken.encoding_for_model(GENERATION_MODEL)
        except KeyError:
            _tokenizer = tiktoken.get_encoding("o200k_base")
    return _tokenizer


_AUXILIARY_NULL_QUESTION_RE = re.compile(r"^(is|are|was|were|did|does|do|can|could|has|have|had)\s+(.+)$", re.IGNORECASE)
_SCOPED_INSUFFICIENCY_QUESTION_MARKERS = (
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


def _resolution_to_retrieval_refs(evidence_pages: list[Any]) -> list[RetrievalRef]:
    raw_refs = []
    for evidence_page in evidence_pages:
        doc_id = str(getattr(evidence_page, "doc_id", "") or "").strip()
        page_num = int(getattr(evidence_page, "page_num", 0) or 0)
        if not doc_id or page_num <= 0:
            continue
        raw_refs.append({"doc_id": doc_id, "page_numbers": [page_num]})
    return normalize_retrieved_pages(raw_refs)


def _submission_from_resolution(
    question_id: str,
    resolution: Any,
    timer: TelemetryTimer,
    *,
    answer_type: str,
    question_text: str,
) -> SubmissionAnswer:
    timer.mark_token()
    timing = timer.finish()
    retrieval_refs = _resolution_to_retrieval_refs(list(getattr(resolution, "evidence_pages", []) or []))
    return SubmissionAnswer(
        question_id=question_id,
        answer=_finalize_answer_value(getattr(resolution, "answer"), answer_type, question_text=question_text),
        telemetry=Telemetry(
            timing=TimingMetrics(
                ttft_ms=timing.ttft_ms,
                tpot_ms=timing.tpot_ms,
                total_time_ms=timing.total_time_ms,
            ),
            retrieval=retrieval_refs,
            usage=UsageMetrics(input_tokens=0, output_tokens=0),
            model_name="deterministic-resolver",
        ),
    )


def _finalize_answer_value(answer_value: Any, answer_type: str, *, question_text: str) -> Any:
    normalized_answer_type = str(answer_type or "").lower()
    if normalized_answer_type != "free_text" or answer_value is None:
        return answer_value
    if not isinstance(answer_value, str):
        return answer_value
    return parse_answer(answer_value, normalized_answer_type, question_text=question_text)


def _override_structural_grounding_refs(
    plan: Any,
    *,
    question_text: str,
    answer_type: str,
    answer_text: str,
    is_llm_null: bool,
    retrieval_refs: list[RetrievalRef],
) -> list[RetrievalRef]:
    if is_llm_null or str(answer_type or "").lower() not in _STRUCTURED_ANSWER_TYPES:
        return retrieval_refs
    if getattr(plan, "mode", "") != "article_lookup":
        return retrieval_refs

    evidence_pages = select_article_evidence_pages(
        question_text,
        answer_type,
        answer_text=answer_text,
    )
    if not evidence_pages:
        return retrieval_refs
    return _resolution_to_retrieval_refs(evidence_pages)


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


def _chunk_page_numbers(chunk: dict[str, Any]) -> list[int]:
    return sorted(
        int(page)
        for page in chunk.get("page_numbers", [])
        if isinstance(page, int) and page > 0
    )


def _selection_token_root(token: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "", str(token or "").lower())
    if len(normalized) <= 3:
        return normalized
    for suffix in ("ations", "ation", "ments", "ment", "ingly", "edly", "ers", "er", "ing", "ed", "es", "s"):
        if len(normalized) > len(suffix) + 2 and normalized.endswith(suffix):
            return normalized[: -len(suffix)]
    return normalized


def _question_selection_terms(question_text: str) -> tuple[set[str], set[str]]:
    anchors = extract_question_anchors(question_text)
    removable_tokens: set[str] = set()
    removable_values = list(anchors.case_ids) + list(anchors.article_refs) + list(anchors.law_title_mentions) + list(anchors.quoted_sections)
    if anchors.law_number:
        removable_values.append(anchors.law_number)
    for value in removable_values:
        removable_tokens.update(tokenize_legal_text(value, doc_title=str(value or "")))

    base_terms = {
        token
        for token in anchors.lexical_tokens
        if token and token not in _GENERATION_SELECTION_STOPWORDS
    }
    terms = {token for token in base_terms if token not in removable_tokens}
    if not terms:
        terms = base_terms
    roots = {_selection_token_root(token) for token in terms if _selection_token_root(token)}
    return terms, roots


def _chunk_selection_terms(chunk: dict[str, Any]) -> tuple[set[str], set[str], str]:
    text_blob = " ".join(
        str(part or "")
        for part in (
            chunk.get("doc_title"),
            chunk.get("section_path"),
            str(chunk.get("text") or "")[:1600],
        )
    )
    tokens = set(tokenize_legal_text(text_blob, doc_title=str(chunk.get("doc_title") or "")))
    roots = {_selection_token_root(token) for token in tokens if _selection_token_root(token)}
    return tokens, roots, text_blob.lower()


def _page_hint_bonus(chunk: dict[str, Any], plan: QuestionPlan | None) -> float:
    if plan is None:
        return 0.0

    pages = _chunk_page_numbers(chunk)
    if not pages:
        return 0.0

    first_page = pages[0]
    if plan.page_hint == "front":
        if first_page <= 1:
            return 3.5
        if first_page == 2:
            return 3.0
        if first_page == 3:
            return 2.0
        if first_page == 4:
            return 1.0
        return 0.0
    if plan.page_hint == "page_2":
        if 2 in pages:
            return 4.0
        if first_page <= 1:
            return -0.5
        return 0.0
    if plan.page_hint == "first":
        if first_page <= 1:
            return 3.0
        if first_page == 2:
            return 1.5
    return 0.0


def _generic_target_field_bonus(chunk: dict[str, Any], plan: QuestionPlan | None, lowered_blob: str) -> float:
    if plan is None or not plan.target_field:
        return 0.0

    if plan.target_field == "money_value":
        bonus = 0.0
        if _MONEY_SIGNAL_RE.search(lowered_blob):
            bonus += 3.5
        if chunk.get("money_values"):
            bonus += 2.0
        pages = _chunk_page_numbers(chunk)
        if pages and min(pages) <= 1:
            bonus -= 5.5
        return bonus
    if plan.target_field == "issue_date":
        bonus = 5.0 if _ISSUE_DATE_SIGNAL_RE.search(lowered_blob) else 0.0
        pages = _chunk_page_numbers(chunk)
        if plan.page_hint != "front" and pages and min(pages) <= 1:
            bonus -= 3.5
        return bonus
    if plan.target_field == "claim_number":
        return 4.0 if _CLAIM_NUMBER_SIGNAL_RE.search(lowered_blob) else 0.0
    if plan.target_field == "party":
        return min(3.0, 1.2 * len(_PARTY_SIGNAL_RE.findall(lowered_blob)))
    if plan.target_field == "judge":
        return min(3.0, 1.2 * len(_JUDGE_SIGNAL_RE.findall(lowered_blob)))
    if plan.target_field == "law_number":
        return 4.0 if "law no" in lowered_blob or "law number" in lowered_blob else 0.0
    return 0.0


def _front_matter_penalty(lowered_blob: str) -> float:
    header = lowered_blob[:240]
    penalty = 0.0
    if "contents" in header:
        penalty += 3.5
    if "term definition" in header:
        penalty += 4.0
    if "table of contents" in header:
        penalty += 4.0
    if "law no." in header and len(header.split()) <= 20:
        penalty += 2.5
    if "consolidated version" in header and len(header.split()) <= 40:
        penalty += 2.0
    return penalty


def _generic_selection_score(
    chunk_with_score: tuple[dict[str, Any], float],
    *,
    question_text: str,
    answer_type: str,
    intent: GroundingIntent | None,
    plan: QuestionPlan | None,
    rank_index: int,
) -> float:
    chunk, retrieval_score = chunk_with_score
    chunk_tokens, chunk_roots, lowered_blob = _chunk_selection_terms(chunk)
    question_terms, question_roots = _question_selection_terms(question_text)

    exact_overlap = len(question_terms & chunk_tokens)
    root_overlap = len(question_roots & chunk_roots)
    score = float(retrieval_score) - (rank_index * 0.01)
    score += exact_overlap * 0.8
    score += root_overlap * 1.05
    score += _page_hint_bonus(chunk, plan)
    score += _generic_target_field_bonus(chunk, plan, lowered_blob)

    if any(root.startswith("admin") for root in question_roots):
        if "administration of this law" in lowered_blob:
            score += 6.0
        if "administered by" in lowered_blob:
            score += 5.0
        if "registrar" in lowered_blob:
            score += 2.0

    if intent is not None and intent.case_ids:
        case_hits = sum(1 for case_id in intent.case_ids if case_id and case_id in lowered_blob.upper())
        score += min(2, case_hits) * 1.4

    score -= _front_matter_penalty(lowered_blob)
    if str(answer_type or "").lower() in {"number", "date"} and "contents" in lowered_blob[:240]:
        score -= 1.0
    return score


def _selection_mode(intent: GroundingIntent | None) -> str:
    if intent is None:
        return "default"

    mode = str(getattr(intent, "selection_mode", "") or "").strip().lower()
    if mode and mode != "default":
        return mode

    if intent.kind in {"title_page", "date_of_issue"}:
        return "frontmatter"
    if intent.is_compare:
        return "compare_balanced"
    return "default"


def _intent_selection_score(
    chunk_with_score: tuple[dict[str, Any], float],
    *,
    question_text: str | None,
    answer_type: str,
    intent: GroundingIntent,
    plan: QuestionPlan | None,
    rank_index: int,
) -> float:
    chunk, retrieval_score = chunk_with_score
    chunk_tokens, chunk_roots, lowered_blob = _chunk_selection_terms(chunk)
    question_terms, question_roots = _question_selection_terms(question_text or "")
    selection_mode = _selection_mode(intent)

    exact_overlap = len(question_terms & chunk_tokens)
    root_overlap = len(question_roots & chunk_roots)
    score = float(retrieval_score) - (rank_index * 0.01)
    score += exact_overlap * 0.5
    score += root_overlap * 0.7
    score += _page_hint_bonus(chunk, plan)
    score += _generic_target_field_bonus(chunk, plan, lowered_blob)

    phrase_hits = sum(1 for phrase in intent.keyphrases if phrase and phrase in lowered_blob)
    score += min(3, phrase_hits) * 0.9

    if intent.case_ids:
        case_hits = sum(1 for case_id in intent.case_ids if case_id and case_id in lowered_blob.upper())
        score += min(2, case_hits) * 1.4

    pages = _chunk_page_numbers(chunk)
    first_page = pages[0] if pages else 99
    if plan is None:
        if selection_mode == "frontmatter":
            if first_page <= 1:
                score += 3.5
            elif first_page == 2:
                score += 3.0
            elif first_page == 3:
                score += 2.0
            elif first_page == 4:
                score += 1.0
        elif selection_mode == "compare_balanced":
            if first_page <= 1:
                score += 3.0
            elif first_page == 2:
                score += 2.0
            elif first_page == 3:
                score += 0.75

    if selection_mode == "frontmatter":
        score -= _front_matter_penalty(lowered_blob)
        if first_page > 4:
            score -= 4.0
        elif first_page == 4:
            score -= 1.5
        elif first_page == 3:
            score -= 1.5
    elif selection_mode == "compare_balanced":
        score -= _front_matter_penalty(lowered_blob) * 0.35
        if first_page > 4:
            score -= 2.0
        if intent.kind == "judge_compare":
            score += min(3.5, 1.0 * len(_JUDGE_SIGNAL_RE.findall(lowered_blob)))
        if intent.kind == "party_compare":
            score += min(3.5, 1.0 * len(_PARTY_SIGNAL_RE.findall(lowered_blob)))

    if str(answer_type or "").lower() in {"number", "date"} and "contents" in lowered_blob[:240]:
        score -= 1.0
    return score


def _dedupe_chunks_by_page(
    retrieved_chunks: list[tuple[dict, float]],
) -> list[tuple[dict, float]]:
    deduped: list[tuple[dict, float]] = []
    seen_pages: set[tuple[str, int]] = set()

    for chunk_with_score in retrieved_chunks:
        chunk = chunk_with_score[0]
        doc_id = str(chunk.get("doc_id") or "").strip()
        pages = _chunk_page_numbers(chunk)
        if pages:
            page_key = (doc_id, pages[0])
            if page_key in seen_pages:
                continue
            seen_pages.add(page_key)
        deduped.append(chunk_with_score)

    return deduped


def _select_case_coverage_chunks(
    retrieved_chunks: list[tuple[dict, float]],
    case_ids: tuple[str, ...],
    top_k: int,
) -> list[tuple[dict, float]]:
    if top_k <= 0 or not retrieved_chunks or not case_ids:
        return []

    selected: list[tuple[dict, float]] = []
    seen_chunk_keys: set[str] = set()
    for case_id in case_ids:
        for chunk_with_score in retrieved_chunks:
            chunk_key = _chunk_identity(chunk_with_score)
            if chunk_key in seen_chunk_keys or not _chunk_matches_case_id(chunk_with_score[0], case_id):
                continue
            selected.append(chunk_with_score)
            seen_chunk_keys.add(chunk_key)
            break
        if len(selected) >= top_k:
            break
    return selected


def _order_generation_candidates(
    retrieved_chunks: list[tuple[dict, float]],
    *,
    question_text: str | None = None,
    answer_type: str = "free_text",
    intent: GroundingIntent | None = None,
    plan: QuestionPlan | None = None,
) -> list[tuple[dict, float]]:
    if not retrieved_chunks or intent is None:
        return list(retrieved_chunks)

    selection_mode = _selection_mode(intent)
    if selection_mode == "default" and intent.kind != "generic":
        return list(retrieved_chunks)
    if intent.kind == "generic" and not question_text:
        return list(retrieved_chunks)

    scored: list[tuple[float, float, int, tuple[dict, float]]] = []
    for index, chunk_with_score in enumerate(retrieved_chunks):
        if intent.kind == "generic":
            selection_score = _generic_selection_score(
                chunk_with_score,
                question_text=question_text,
                answer_type=answer_type,
                intent=intent,
                plan=plan,
                rank_index=index,
            )
        else:
            selection_score = _intent_selection_score(
                chunk_with_score,
                question_text=question_text,
                answer_type=answer_type,
                intent=intent,
                plan=plan,
                rank_index=index,
            )
        scored.append((selection_score, float(chunk_with_score[1]), index, chunk_with_score))

    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [item[3] for item in scored]


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


def _select_balanced_compare_chunks(
    retrieved_chunks: list[tuple[dict, float]],
    case_ids: tuple[str, ...],
    top_k: int,
) -> list[tuple[dict, float]]:
    if top_k <= 0 or not retrieved_chunks or not case_ids:
        return []

    deduped_chunks = _dedupe_chunks_by_page(retrieved_chunks)
    case_buckets: dict[str, list[tuple[dict, float]]] = {}
    for case_id in case_ids:
        case_buckets[case_id] = [
            chunk_with_score
            for chunk_with_score in deduped_chunks
            if _chunk_matches_case_id(chunk_with_score[0], case_id)
        ]

    selected: list[tuple[dict, float]] = []
    seen_chunk_keys: set[str] = set()
    seen_doc_ids_by_case: dict[str, set[str]] = {case_id: set() for case_id in case_ids}

    def _pick_for_case(case_id: str, *, prefer_new_doc: bool) -> tuple[dict, float] | None:
        fallback: tuple[dict, float] | None = None
        for chunk_with_score in case_buckets.get(case_id, []):
            chunk = chunk_with_score[0]
            chunk_key = _chunk_identity(chunk_with_score)
            if chunk_key in seen_chunk_keys:
                continue

            doc_id = str(chunk.get("doc_id") or "").strip()
            if prefer_new_doc and doc_id and doc_id in seen_doc_ids_by_case[case_id]:
                if fallback is None:
                    fallback = chunk_with_score
                continue
            return chunk_with_score
        return fallback

    for case_id in case_ids:
        choice = _pick_for_case(case_id, prefer_new_doc=False)
        if choice is None:
            continue
        selected.append(choice)
        seen_chunk_keys.add(_chunk_identity(choice))
        doc_id = str(choice[0].get("doc_id") or "").strip()
        if doc_id:
            seen_doc_ids_by_case[case_id].add(doc_id)
        if len(selected) >= top_k:
            return selected

    while len(selected) < top_k:
        progress = False
        for case_id in case_ids:
            choice = _pick_for_case(case_id, prefer_new_doc=True)
            if choice is None:
                continue
            selected.append(choice)
            seen_chunk_keys.add(_chunk_identity(choice))
            doc_id = str(choice[0].get("doc_id") or "").strip()
            if doc_id:
                seen_doc_ids_by_case[case_id].add(doc_id)
            progress = True
            if len(selected) >= top_k:
                break
        if not progress:
            break

    return selected


def _select_doc_coverage_chunks(
    retrieved_chunks: list[tuple[dict, float]],
    doc_ids: tuple[str, ...],
    top_k: int,
) -> list[tuple[dict, float]]:
    if top_k <= 0 or not retrieved_chunks or not doc_ids:
        return []

    selected: list[tuple[dict, float]] = []
    seen_chunk_keys: set[str] = set()
    for doc_id in doc_ids:
        for chunk_with_score in retrieved_chunks:
            chunk = chunk_with_score[0]
            chunk_key = _chunk_identity(chunk_with_score)
            if chunk_key in seen_chunk_keys or str(chunk.get("doc_id") or "").strip() != doc_id:
                continue
            selected.append(chunk_with_score)
            seen_chunk_keys.add(chunk_key)
            break
        if len(selected) >= top_k:
            break
    return selected


def _preferred_case_doc_ids(
    retrieved_chunks: list[tuple[dict, float]],
    case_ids: tuple[str, ...],
) -> tuple[str, ...]:
    if not retrieved_chunks or not case_ids:
        return ()

    preferred_doc_ids: list[str] = []
    seen_doc_ids: set[str] = set()
    for chunk_with_score in retrieved_chunks:
        chunk = chunk_with_score[0]
        doc_id = str(chunk.get("doc_id") or "").strip()
        if not doc_id or doc_id in seen_doc_ids:
            continue
        if not any(_chunk_matches_case_id(chunk, case_id) for case_id in case_ids):
            continue
        seen_doc_ids.add(doc_id)
        preferred_doc_ids.append(doc_id)
    return tuple(preferred_doc_ids)


def _filter_chunks_to_doc_ids(
    retrieved_chunks: list[tuple[dict, float]],
    allowed_doc_ids: set[str],
) -> list[tuple[dict, float]]:
    if not retrieved_chunks or not allowed_doc_ids:
        return []
    return [
        chunk_with_score
        for chunk_with_score in retrieved_chunks
        if str(chunk_with_score[0].get("doc_id") or "").strip() in allowed_doc_ids
    ]


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
    question_text: str | None = None,
    answer_type: str = "free_text",
    plan: QuestionPlan | None = None,
    disable_unique_doc_preference: bool = False,
) -> list[tuple[dict, float]]:
    if top_k <= 0 or not retrieved_chunks:
        return []

    selection_mode = _selection_mode(intent)
    ordered_chunks = _order_generation_candidates(
        retrieved_chunks,
        question_text=question_text,
        answer_type=answer_type,
        intent=intent,
        plan=plan,
    )
    if selection_mode == "frontmatter":
        ordered_chunks = _dedupe_chunks_by_page(ordered_chunks)
    selected: list[tuple[dict, float]] = []
    seen_chunks: set[str] = set()
    seen_docs: set[str] = set()

    if intent is not None and selection_mode == "compare_balanced" and intent.case_ids:
        for chunk_with_score in _select_balanced_compare_chunks(ordered_chunks, intent.case_ids, top_k):
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

    elif intent is not None and intent.is_compare and intent.case_ids:
        for chunk_with_score in _select_compare_case_coverage_chunks(ordered_chunks, intent.case_ids, top_k):
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

    if intent is not None and not intent.is_compare and intent.case_ids:
        preferred_doc_ids = _preferred_case_doc_ids(ordered_chunks, intent.case_ids)
        if len(intent.case_ids) >= 2 and preferred_doc_ids:
            for chunk_with_score in _select_doc_coverage_chunks(
                ordered_chunks,
                preferred_doc_ids,
                min(top_k, len(preferred_doc_ids)),
            ):
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

        preferred_doc_id_set = set(preferred_doc_ids)
        if preferred_doc_id_set:
            for chunk_with_score in ordered_chunks:
                chunk = chunk_with_score[0]
                doc_id = str(chunk.get("doc_id") or "").strip()
                chunk_key = _chunk_identity(chunk_with_score)
                if chunk_key in seen_chunks or doc_id not in preferred_doc_id_set:
                    continue
                selected.append(chunk_with_score)
                seen_chunks.add(chunk_key)
                if doc_id:
                    seen_docs.add(doc_id)
                if len(selected) >= top_k:
                    return selected

    if intent is None or disable_unique_doc_preference or not intent.prefer_unique_docs:
        for chunk_with_score in ordered_chunks:
            chunk_key = _chunk_identity(chunk_with_score)
            if chunk_key in seen_chunks:
                continue
            selected.append(chunk_with_score)
            seen_chunks.add(chunk_key)
            if len(selected) >= top_k:
                break
        return selected

    for chunk_with_score in ordered_chunks:
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

    for chunk_with_score in ordered_chunks:
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
    question_text: str | None = None,
    plan: QuestionPlan | None = None,
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
            question_text=question_text,
            answer_type=answer_type,
            plan=plan,
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
        question_text=question_text,
        answer_type=answer_type,
        plan=plan,
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
    plan: QuestionPlan | None = None,
    disable_unique_doc_preference: bool = False,
    allow_scoped_insufficiency: bool = False,
) -> dict[str, Any]:
    generation_top_k = _generation_top_k_for(answer_type, intent=intent)
    generation_chunks = _select_generation_chunks(
        reranked_chunks,
        generation_top_k,
        intent=intent,
        question_text=question_text,
        answer_type=answer_type,
        plan=plan,
        disable_unique_doc_preference=disable_unique_doc_preference,
    )
    messages = build_prompt(
        question_text,
        answer_type,
        generation_chunks,
        intent=intent,
        allow_scoped_insufficiency=allow_scoped_insufficiency,
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
        question_text=question_text,
        plan=plan,
        disable_unique_doc_preference=disable_unique_doc_preference,
    )
    cited_page_keys: set[tuple[str, int]] = set()
    for source_id in cited_source_ids:
        chunk_index = source_id - 1
        if chunk_index < 0 or chunk_index >= len(generation_chunks):
            continue
        cited_chunk = generation_chunks[chunk_index][0]
        doc_id = str(cited_chunk.get("doc_id") or "").strip()
        if not doc_id:
            continue
        for page_num in cited_chunk.get("page_numbers", []):
            if isinstance(page_num, int) and page_num > 0:
                cited_page_keys.add((doc_id, page_num))
    grounding_refs = collect_grounding_pages(
        grounding_chunks,
        score_threshold=grounding_threshold,
        is_null=is_llm_null,
        question_text=question_text,
        answer_text=answer_text,
        intent=intent,
        allowed_doc_ids=_generation_doc_ids(generation_chunks),
        answer_type=answer_type,
        cited_page_keys=cited_page_keys,
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


def _should_retry_scoped_free_text_generation(
    question_text: str,
    answer_type: str,
    plan: Any,
    primary_attempt: dict[str, Any],
    reranked_chunks: list[tuple[dict, float]],
) -> bool:
    if str(answer_type or "").lower() != "free_text":
        return False
    if not primary_attempt["is_llm_null"] or not reranked_chunks:
        return False
    if not getattr(plan, "case_ids", ()):
        return False
    lowered_question = str(question_text or "").lower()
    return any(marker in lowered_question for marker in _SCOPED_INSUFFICIENCY_QUESTION_MARKERS)


def _chunks_for_scoped_free_text_retry(
    intent: GroundingIntent | None,
    reranked_chunks: list[tuple[dict, float]],
) -> list[tuple[dict, float]]:
    if intent is None or intent.kind != "generic" or len(intent.case_ids) != 1:
        return reranked_chunks

    preferred_doc_ids = set(_preferred_case_doc_ids(reranked_chunks, intent.case_ids))
    filtered_chunks = _filter_chunks_to_doc_ids(reranked_chunks, preferred_doc_ids)
    return filtered_chunks or reranked_chunks


def _should_retry_article_structured_generation(
    answer_type: str,
    plan: Any,
    intent: GroundingIntent | None,
    primary_attempt: dict[str, Any],
) -> bool:
    if str(answer_type or "").lower() not in _STRUCTURED_ANSWER_TYPES:
        return False
    if not primary_attempt["is_llm_null"]:
        return False
    if getattr(plan, "mode", "") != "article_lookup":
        return False
    return bool(intent and intent.kind == "article_ref")


def _widen_article_generation_intent(intent: GroundingIntent | None) -> GroundingIntent | None:
    if intent is None:
        return None
    widened_generation_top_k = max(4, int(intent.generation_top_k or 0))
    widened_grounding_top_k = max(2, int(intent.grounding_chunk_top_k or 0))
    return replace(
        intent,
        generation_top_k=widened_generation_top_k,
        grounding_chunk_top_k=widened_grounding_top_k,
    )


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
        plan = build_question_plan(question_text, answer_type)
        active_intent = detect_grounding_intent(question_text, answer_type)

        tokenizer = _get_tokenizer()
        timer = TelemetryTimer()

        resolution = try_resolve_question(question_item, plan)
        if resolution is not None:
            logger.info(
                f"[{question_id[:8]}] resolved deterministically via {getattr(resolution, 'method', plan.mode)} "
                f"pages={len(getattr(resolution, 'evidence_pages', []))} answer={str(getattr(resolution, 'answer', ''))[:50]}"
            )
            return _submission_from_resolution(
                question_id,
                resolution,
                timer,
                answer_type=answer_type,
                question_text=question_text,
            )

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
            answer_value = _finalize_answer_value(answer_value, answer_type, question_text=question_text)

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
            plan=plan,
            disable_unique_doc_preference=compare_bias_disabled,
        )
        input_tokens = int(attempt["input_tokens"])
        output_tokens = int(attempt["output_tokens"])

        if _should_retry_scoped_free_text_generation(question_text, answer_type, plan, attempt, reranked_chunks):
            logger.info(
                f"[{question_id[:8]}] free-text null fallback; retrying generation with scoped insufficiency allowed"
            )
            scoped_retry_chunks = _chunks_for_scoped_free_text_retry(active_intent, reranked_chunks)
            scoped_attempt = _run_generation_pass(
                question_text,
                answer_type,
                scoped_retry_chunks,
                tokenizer,
                timer,
                self.grounding_threshold,
                intent=active_intent,
                plan=plan,
                disable_unique_doc_preference=compare_bias_disabled,
                allow_scoped_insufficiency=True,
            )
            input_tokens += int(scoped_attempt["input_tokens"])
            output_tokens += int(scoped_attempt["output_tokens"])
            if _should_use_fallback_attempt(attempt, scoped_attempt):
                attempt = scoped_attempt

        if _should_retry_article_structured_generation(answer_type, plan, active_intent, attempt):
            logger.info(
                f"[{question_id[:8]}] article structured null fallback; retrying generation with wider article context"
            )
            widened_article_intent = _widen_article_generation_intent(active_intent)
            article_attempt = _run_generation_pass(
                question_text,
                answer_type,
                reranked_chunks,
                tokenizer,
                timer,
                self.grounding_threshold,
                intent=widened_article_intent,
                plan=plan,
                disable_unique_doc_preference=compare_bias_disabled,
            )
            input_tokens += int(article_attempt["input_tokens"])
            output_tokens += int(article_attempt["output_tokens"])
            if _should_use_fallback_attempt(attempt, article_attempt):
                attempt = article_attempt

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
                plan=plan,
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

        if is_llm_null and answer_type == "free_text":
            answer_value = _free_text_null_answer(question_text)
        answer_value = _finalize_answer_value(answer_value, answer_type, question_text=question_text)
        retrieval_refs = _override_structural_grounding_refs(
            plan,
            question_text=question_text,
            answer_type=answer_type,
            answer_text=answer_text,
            is_llm_null=is_llm_null,
            retrieval_refs=retrieval_refs,
        )

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




