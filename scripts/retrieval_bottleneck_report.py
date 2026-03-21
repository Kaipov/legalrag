"""
Locate where retrieval relevance is lost inside the current pipeline stages.

Stages:
  1. rrf_candidates      - hybrid RRF pool before intent bias (up to 30)
  2. intent_ranked       - after intent bias, before truncation (up to 30)
  3. retriever_output    - actual HybridRetriever output used downstream (typically top-10)
  4. generation_chunks   - chunks selected into the prompt context

Usage:
    python -m scripts.retrieval_bottleneck_report
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CHUNKS_JSONL, DATA_DIR, ENABLE_RERANKER, PROJECT_ROOT
from src.pipeline import _generation_top_k_for, _select_generation_chunks
from src.retrieve.grounding_policy import detect_grounding_intent
from src.retrieve.hybrid import HybridRetriever
from src.retrieve.question_plan import build_question_plan

DEFAULT_GOLD_PATH = PROJECT_ROOT / "golden_submission.json"
DEFAULT_QUESTIONS_PATH = DATA_DIR / "questions.json"
DEFAULT_DETAILS_LIMIT = 10
STAGE_ORDER = (
    "rrf_candidates",
    "intent_ranked",
    "retriever_output",
    "generation_chunks",
)


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_questions(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or DEFAULT_QUESTIONS_PATH
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _page_ref_set(answer_payload: dict[str, Any]) -> set[tuple[str, int]]:
    refs = answer_payload.get("telemetry", {}).get("retrieval", {}).get("retrieved_chunk_pages", [])
    if not isinstance(refs, list):
        return set()

    page_refs: set[tuple[str, int]] = set()
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        doc_id = str(ref.get("doc_id") or "").strip()
        if not doc_id:
            continue
        for page in ref.get("page_numbers", []):
            if isinstance(page, int) and page > 0:
                page_refs.add((doc_id, page))
    return page_refs


def _chunk_page_refs(chunk: dict[str, Any]) -> set[tuple[str, int]]:
    doc_id = str(chunk.get("doc_id") or "").strip()
    if not doc_id:
        return set()
    return {
        (doc_id, int(page))
        for page in chunk.get("page_numbers", [])
        if isinstance(page, int) and page > 0
    }


def _chunk_doc_id(chunk: dict[str, Any]) -> str:
    return str(chunk.get("doc_id") or "").strip()


def _stage_snapshot(chunks: list[tuple[dict[str, Any], float]]) -> dict[str, Any]:
    page_refs: set[tuple[str, int]] = set()
    doc_ids: set[str] = set()
    for chunk, _score in chunks:
        page_refs |= _chunk_page_refs(chunk)
        doc_id = _chunk_doc_id(chunk)
        if doc_id:
            doc_ids.add(doc_id)
    return {
        "size": len(chunks),
        "page_refs": page_refs,
        "doc_ids": doc_ids,
    }


def build_stage_row(
    question: dict[str, Any],
    gold_page_refs: set[tuple[str, int]],
    stage_snapshots: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    gold_doc_ids = {doc_id for doc_id, _page in gold_page_refs}
    stages: dict[str, dict[str, Any]] = {}

    for stage_name, snapshot in stage_snapshots.items():
        page_overlap = snapshot["page_refs"] & gold_page_refs
        doc_overlap = snapshot["doc_ids"] & gold_doc_ids
        stages[stage_name] = {
            "size": int(snapshot["size"]),
            "page_hit": bool(page_overlap),
            "page_recall": len(page_overlap) / len(gold_page_refs),
            "doc_hit": bool(doc_overlap),
            "doc_full_hit": bool(gold_doc_ids) and len(doc_overlap) == len(gold_doc_ids),
            "doc_coverage": len(doc_overlap) / len(gold_doc_ids),
        }

    return {
        "question_id": question["id"],
        "question": question.get("question", ""),
        "answer_type": question.get("answer_type", "unknown"),
        "bucket": detect_grounding_intent(question.get("question", ""), question.get("answer_type", "")).kind,
        "gold_page_count": len(gold_page_refs),
        "gold_doc_count": len(gold_doc_ids),
        "stages": stages,
    }


def summarize_stage_rows(rows: list[dict[str, Any]], stage_name: str) -> dict[str, Any]:
    if not rows:
        return {
            "question_count": 0,
            "mean_size": 0.0,
            "page_hit_rate": 0.0,
            "page_recall": 0.0,
            "doc_hit_rate": 0.0,
            "doc_full_hit_rate": 0.0,
            "doc_coverage": 0.0,
        }

    return {
        "question_count": len(rows),
        "mean_size": round(statistics.mean(row["stages"][stage_name]["size"] for row in rows), 4),
        "page_hit_rate": round(
            sum(1.0 for row in rows if row["stages"][stage_name]["page_hit"]) / len(rows),
            4,
        ),
        "page_recall": round(
            sum(row["stages"][stage_name]["page_recall"] for row in rows) / len(rows),
            4,
        ),
        "doc_hit_rate": round(
            sum(1.0 for row in rows if row["stages"][stage_name]["doc_hit"]) / len(rows),
            4,
        ),
        "doc_full_hit_rate": round(
            sum(1.0 for row in rows if row["stages"][stage_name]["doc_full_hit"]) / len(rows),
            4,
        ),
        "doc_coverage": round(
            sum(row["stages"][stage_name]["doc_coverage"] for row in rows) / len(rows),
            4,
        ),
    }


def summarize_transition(rows: list[dict[str, Any]], from_stage: str, to_stage: str) -> dict[str, Any]:
    if not rows:
        return {
            "prev_page_hit_count": 0,
            "next_page_hit_count": 0,
            "lost_page_hits": 0,
            "retained_page_hit_rate": 0.0,
            "mean_page_recall_delta": 0.0,
            "mean_size_delta": 0.0,
        }

    prev_page_hit_rows = [row for row in rows if row["stages"][from_stage]["page_hit"]]
    lost_page_hit_rows = [
        row
        for row in rows
        if row["stages"][from_stage]["page_hit"] and not row["stages"][to_stage]["page_hit"]
    ]

    return {
        "prev_page_hit_count": len(prev_page_hit_rows),
        "next_page_hit_count": sum(1 for row in rows if row["stages"][to_stage]["page_hit"]),
        "lost_page_hits": len(lost_page_hit_rows),
        "retained_page_hit_rate": round(
            0.0
            if not prev_page_hit_rows
            else 1.0 - (len(lost_page_hit_rows) / len(prev_page_hit_rows)),
            4,
        ),
        "mean_page_recall_delta": round(
            sum(
                row["stages"][to_stage]["page_recall"] - row["stages"][from_stage]["page_recall"]
                for row in rows
            ) / len(rows),
            4,
        ),
        "mean_size_delta": round(
            sum(
                row["stages"][to_stage]["size"] - row["stages"][from_stage]["size"]
                for row in rows
            ) / len(rows),
            4,
        ),
        "loss_rows": lost_page_hit_rows,
    }


def summarize_rows_by_bucket(rows: list[dict[str, Any]], stage_name: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["bucket"], []).append(row)
    return {
        bucket: summarize_stage_rows(group_rows, stage_name)
        for bucket, group_rows in sorted(grouped.items())
    }


def build_bottleneck_report(
    *,
    gold_path: Path,
    questions_path: Path,
    enable_reranker: bool = ENABLE_RERANKER,
    details_limit: int = DEFAULT_DETAILS_LIMIT,
    limit: int | None = None,
) -> dict[str, Any]:
    questions = load_questions(questions_path)
    gold_submission = load_json(gold_path)
    gold_answers = {answer["question_id"]: answer for answer in gold_submission.get("answers", [])}

    evidence_questions = []
    for question in questions:
        gold_payload = gold_answers.get(question["id"])
        if not gold_payload:
            continue
        gold_page_refs = _page_ref_set(gold_payload)
        if not gold_page_refs:
            continue
        evidence_questions.append((question, gold_page_refs))

    if limit is not None:
        evidence_questions = evidence_questions[: max(0, limit)]

    retriever = HybridRetriever(chunks_path=CHUNKS_JSONL, enable_reranker=enable_reranker)
    rows: list[dict[str, Any]] = []

    for question, gold_page_refs in evidence_questions:
        query = question.get("question", "")
        answer_type = question.get("answer_type", "free_text")
        intent = detect_grounding_intent(query, answer_type)
        plan = build_question_plan(query, answer_type)

        rrf_candidates = retriever._get_rrf_candidates(query)
        intent_ranked = retriever._apply_intent_bias(rrf_candidates, intent)
        retriever_output = retriever.retrieve(query, intent=intent)
        generation_top_k = _generation_top_k_for(answer_type, intent=intent)
        generation_chunks = _select_generation_chunks(
            retriever_output,
            generation_top_k,
            intent=intent,
            question_text=query,
            answer_type=answer_type,
            plan=plan,
            disable_unique_doc_preference=False,
        )

        stage_snapshots = {
            "rrf_candidates": _stage_snapshot(rrf_candidates),
            "intent_ranked": _stage_snapshot(intent_ranked),
            "retriever_output": _stage_snapshot(retriever_output),
            "generation_chunks": _stage_snapshot(generation_chunks),
        }
        rows.append(build_stage_row(question, gold_page_refs, stage_snapshots))

    stage_summaries = {
        stage_name: {
            "overall": summarize_stage_rows(rows, stage_name),
            "by_bucket": summarize_rows_by_bucket(rows, stage_name),
        }
        for stage_name in STAGE_ORDER
    }

    transition_summaries = {
        f"{from_stage}->{to_stage}": summarize_transition(rows, from_stage, to_stage)
        for from_stage, to_stage in zip(STAGE_ORDER, STAGE_ORDER[1:])
    }

    transition_losses: dict[str, list[dict[str, Any]]] = {}
    for transition_name, transition_summary in transition_summaries.items():
        transition_losses[transition_name] = transition_summary["loss_rows"][:details_limit]
        transition_summary.pop("loss_rows", None)

    return {
        "gold_path": str(gold_path),
        "questions_path": str(questions_path),
        "question_count": len(rows),
        "reranker_enabled": bool(enable_reranker),
        "stage_summaries": stage_summaries,
        "transition_summaries": transition_summaries,
        "transition_losses": transition_losses,
    }


def print_bottleneck_report(summary: dict[str, Any], details_limit: int = DEFAULT_DETAILS_LIMIT) -> None:
    print("\nRetrieval bottleneck report")
    print(f"Weak gold:      {summary['gold_path']}")
    print(f"Questions:      {summary['questions_path']}")
    print(f"Evidence items: {summary['question_count']}")
    print(f"Reranker:       {'on' if summary.get('reranker_enabled') else 'off'}")

    for stage_name in STAGE_ORDER:
        stage_summary = summary["stage_summaries"][stage_name]
        overall = stage_summary["overall"]
        print(f"\nStage: {stage_name}")
        print(f"  Mean size:          {overall['mean_size']}")
        print(f"  Page hit rate:      {overall['page_hit_rate']:.4f}")
        print(f"  Page recall:        {overall['page_recall']:.4f}")
        print(f"  Doc hit rate:       {overall['doc_hit_rate']:.4f}")
        print(f"  Doc full hit rate:  {overall['doc_full_hit_rate']:.4f}")
        print(f"  Doc coverage:       {overall['doc_coverage']:.4f}")
        print("  By bucket")
        for bucket, bucket_summary in stage_summary["by_bucket"].items():
            print(
                f"    {bucket:15s} "
                f"page_hit={bucket_summary['page_hit_rate']:.4f} "
                f"page_recall={bucket_summary['page_recall']:.4f} "
                f"doc_full={bucket_summary['doc_full_hit_rate']:.4f}"
            )

    print("\nStage transitions")
    for transition_name, transition_summary in summary["transition_summaries"].items():
        print(
            f"  {transition_name:34s} "
            f"lost_page_hits={transition_summary['lost_page_hits']:2d} "
            f"retained_hit_rate={transition_summary['retained_page_hit_rate']:.4f} "
            f"mean_page_recall_delta={transition_summary['mean_page_recall_delta']:+.4f} "
            f"mean_size_delta={transition_summary['mean_size_delta']:+.4f}"
        )

    print("\nRepresentative losses")
    for transition_name, rows in summary["transition_losses"].items():
        print(f"  {transition_name}")
        if not rows:
            print("    none")
            continue
        for row in rows[:details_limit]:
            print(
                f"    {row['question_id'][:8]} "
                f"[{row['bucket']}/{row['answer_type']}] "
                f"gold_docs={row['gold_doc_count']} gold_pages={row['gold_page_count']} "
                f"{row['question'][:100]}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Locate retrieval bottlenecks across internal pipeline stages")
    parser.add_argument(
        "--gold",
        default=str(DEFAULT_GOLD_PATH),
        help=f"Path to the weak-gold submission JSON (default: {DEFAULT_GOLD_PATH.name})",
    )
    parser.add_argument(
        "--questions",
        default=str(DEFAULT_QUESTIONS_PATH),
        help=f"Path to the questions JSON (default: {DEFAULT_QUESTIONS_PATH.name})",
    )
    parser.add_argument(
        "--details-limit",
        type=int,
        default=DEFAULT_DETAILS_LIMIT,
        help=f"How many loss rows to print per transition (default: {DEFAULT_DETAILS_LIMIT})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional cap on how many evidence-bearing questions to score",
    )
    reranker_group = parser.add_mutually_exclusive_group()
    reranker_group.add_argument(
        "--enable-reranker",
        dest="enable_reranker",
        action="store_true",
        default=None,
        help="Force-enable the configured reranker for retriever_output and generation_chunks",
    )
    reranker_group.add_argument(
        "--disable-reranker",
        dest="enable_reranker",
        action="store_false",
        help="Force-disable the reranker, regardless of config",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    gold_path = Path(args.gold)
    questions_path = Path(args.questions)
    if not gold_path.exists():
        parser.error(f"gold submission not found at {gold_path}")
    if not questions_path.exists():
        parser.error(f"questions file not found at {questions_path}")

    summary = build_bottleneck_report(
        gold_path=gold_path,
        questions_path=questions_path,
        enable_reranker=ENABLE_RERANKER if args.enable_reranker is None else bool(args.enable_reranker),
        details_limit=max(1, args.details_limit),
        limit=args.limit,
    )
    print_bottleneck_report(summary, details_limit=max(1, args.details_limit))


if __name__ == "__main__":
    main()
