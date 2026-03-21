"""
Evaluate chunk text retrieval quality against weak-gold public evidence pages.

Usage:
    python -m scripts.retrieval_scorecard
    python -m scripts.retrieval_scorecard --modes bm25,semantic,hybrid_rrf,current
    python -m scripts.retrieval_scorecard --top-ks 1,3,5,10,30
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
from src.retrieve.bm25 import BM25Searcher
from src.retrieve.grounding_policy import detect_grounding_intent
from src.retrieve.hybrid import HybridRetriever
from src.retrieve.semantic import SemanticSearcher

DEFAULT_GOLD_PATH = PROJECT_ROOT / "golden_submission.json"
DEFAULT_QUESTIONS_PATH = DATA_DIR / "questions.json"
DEFAULT_MODES = ("bm25", "semantic", "hybrid_rrf", "current")
DEFAULT_TOP_KS = (1, 3, 5, 10, 30)
DEFAULT_DETAILS_LIMIT = 8


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


def _read_chunks_by_id(path: Path | None = None) -> dict[str, dict[str, Any]]:
    path = path or CHUNKS_JSONL
    chunks_by_id: dict[str, dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            chunk = json.loads(line)
            chunk_id = str(chunk.get("chunk_id") or "").strip()
            if chunk_id:
                chunks_by_id[chunk_id] = chunk
    return chunks_by_id


def _parse_csv_choices(raw_value: str, *, valid_values: set[str]) -> tuple[str, ...]:
    values = []
    seen: set[str] = set()
    for part in str(raw_value or "").split(","):
        normalized = part.strip().lower()
        if not normalized or normalized in seen:
            continue
        if normalized not in valid_values:
            raise ValueError(f"Unsupported choice '{normalized}'. Expected one of: {', '.join(sorted(valid_values))}")
        seen.add(normalized)
        values.append(normalized)
    if not values:
        raise ValueError("At least one choice is required")
    return tuple(values)


def parse_top_ks(raw_value: str) -> tuple[int, ...]:
    values = []
    seen: set[int] = set()
    for part in str(raw_value or "").split(","):
        normalized = part.strip()
        if not normalized:
            continue
        value = int(normalized)
        if value <= 0 or value in seen:
            continue
        seen.add(value)
        values.append(value)
    if not values:
        raise ValueError("At least one positive top-k value is required")
    return tuple(sorted(values))


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


def build_retrieval_row(
    question: dict[str, Any],
    retrieved_chunks: list[tuple[dict[str, Any], float]],
    gold_page_refs: set[tuple[str, int]],
    top_ks: tuple[int, ...],
) -> dict[str, Any]:
    if not gold_page_refs:
        raise ValueError("gold_page_refs must not be empty")

    gold_doc_ids = {doc_id for doc_id, _page in gold_page_refs}
    first_page_hit_rank: int | None = None
    first_doc_hit_rank: int | None = None
    first_doc_full_hit_rank: int | None = None

    cumulative_page_refs: set[tuple[str, int]] = set()
    cumulative_doc_ids: set[str] = set()
    page_hit_at: dict[int, bool] = {}
    doc_hit_at: dict[int, bool] = {}
    doc_full_hit_at: dict[int, bool] = {}
    page_recall_at: dict[int, float] = {}
    doc_coverage_at: dict[int, float] = {}

    rank_to_snapshot: dict[int, tuple[set[tuple[str, int]], set[str]]] = {}
    max_k = max(top_ks)
    for rank, (chunk, _score) in enumerate(retrieved_chunks[:max_k], start=1):
        chunk_page_refs = _chunk_page_refs(chunk)
        chunk_doc_id = _chunk_doc_id(chunk)
        cumulative_page_refs |= chunk_page_refs
        if chunk_doc_id:
            cumulative_doc_ids.add(chunk_doc_id)

        if first_page_hit_rank is None and chunk_page_refs & gold_page_refs:
            first_page_hit_rank = rank
        if first_doc_hit_rank is None and chunk_doc_id and chunk_doc_id in gold_doc_ids:
            first_doc_hit_rank = rank
        if first_doc_full_hit_rank is None and gold_doc_ids and gold_doc_ids <= cumulative_doc_ids:
            first_doc_full_hit_rank = rank

        if rank in top_ks:
            rank_to_snapshot[rank] = (set(cumulative_page_refs), set(cumulative_doc_ids))

    # Ensure every requested top-k has a snapshot, even when fewer chunks were returned.
    last_snapshot = rank_to_snapshot.get(max(rank_to_snapshot.keys(), default=0), (set(), set()))
    for k in top_ks:
        if k not in rank_to_snapshot:
            rank_to_snapshot[k] = (set(last_snapshot[0]), set(last_snapshot[1]))

        retrieved_page_refs, retrieved_doc_ids = rank_to_snapshot[k]
        page_overlap = len(retrieved_page_refs & gold_page_refs)
        doc_overlap = len(retrieved_doc_ids & gold_doc_ids)
        page_hit_at[k] = page_overlap > 0
        doc_hit_at[k] = doc_overlap > 0
        doc_full_hit_at[k] = bool(gold_doc_ids) and doc_overlap == len(gold_doc_ids)
        page_recall_at[k] = page_overlap / len(gold_page_refs)
        doc_coverage_at[k] = doc_overlap / len(gold_doc_ids)

    return {
        "question_id": question["id"],
        "question": question.get("question", ""),
        "answer_type": question.get("answer_type", "unknown"),
        "bucket": detect_grounding_intent(question.get("question", ""), question.get("answer_type", "")).kind,
        "gold_page_count": len(gold_page_refs),
        "gold_doc_count": len(gold_doc_ids),
        "first_page_hit_rank": first_page_hit_rank,
        "first_doc_hit_rank": first_doc_hit_rank,
        "first_doc_full_hit_rank": first_doc_full_hit_rank,
        "page_mrr": 0.0 if first_page_hit_rank is None else 1.0 / first_page_hit_rank,
        "doc_mrr": 0.0 if first_doc_hit_rank is None else 1.0 / first_doc_hit_rank,
        "page_hit_at": page_hit_at,
        "doc_hit_at": doc_hit_at,
        "doc_full_hit_at": doc_full_hit_at,
        "page_recall_at": page_recall_at,
        "doc_coverage_at": doc_coverage_at,
    }


def summarize_retrieval_rows(
    rows: list[dict[str, Any]],
    top_ks: tuple[int, ...],
) -> dict[str, Any]:
    if not rows:
        return {
            "question_count": 0,
            "page_mrr": 0.0,
            "doc_mrr": 0.0,
            "mean_first_page_hit_rank": None,
            "mean_first_doc_hit_rank": None,
            "mean_first_doc_full_hit_rank": None,
            "page_hit_at": {k: 0.0 for k in top_ks},
            "doc_hit_at": {k: 0.0 for k in top_ks},
            "doc_full_hit_at": {k: 0.0 for k in top_ks},
            "page_recall_at": {k: 0.0 for k in top_ks},
            "doc_coverage_at": {k: 0.0 for k in top_ks},
            "misses_at_max_k": 0,
        }

    def _mean(values: list[int | float]) -> float | None:
        return round(statistics.mean(values), 4) if values else None

    max_k = max(top_ks)
    return {
        "question_count": len(rows),
        "page_mrr": round(sum(row["page_mrr"] for row in rows) / len(rows), 4),
        "doc_mrr": round(sum(row["doc_mrr"] for row in rows) / len(rows), 4),
        "mean_first_page_hit_rank": _mean(
            [row["first_page_hit_rank"] for row in rows if row["first_page_hit_rank"] is not None]
        ),
        "mean_first_doc_hit_rank": _mean(
            [row["first_doc_hit_rank"] for row in rows if row["first_doc_hit_rank"] is not None]
        ),
        "mean_first_doc_full_hit_rank": _mean(
            [row["first_doc_full_hit_rank"] for row in rows if row["first_doc_full_hit_rank"] is not None]
        ),
        "page_hit_at": {
            k: round(sum(1.0 for row in rows if row["page_hit_at"][k]) / len(rows), 4)
            for k in top_ks
        },
        "doc_hit_at": {
            k: round(sum(1.0 for row in rows if row["doc_hit_at"][k]) / len(rows), 4)
            for k in top_ks
        },
        "doc_full_hit_at": {
            k: round(sum(1.0 for row in rows if row["doc_full_hit_at"][k]) / len(rows), 4)
            for k in top_ks
        },
        "page_recall_at": {
            k: round(sum(row["page_recall_at"][k] for row in rows) / len(rows), 4)
            for k in top_ks
        },
        "doc_coverage_at": {
            k: round(sum(row["doc_coverage_at"][k] for row in rows) / len(rows), 4)
            for k in top_ks
        },
        "misses_at_max_k": sum(1 for row in rows if not row["page_hit_at"][max_k]),
    }


def summarize_rows_by_key(
    rows: list[dict[str, Any]],
    top_ks: tuple[int, ...],
    key: str,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(key) or "unknown"), []).append(row)
    return {
        group_value: summarize_retrieval_rows(group_rows, top_ks)
        for group_value, group_rows in sorted(grouped.items())
    }


def _top_miss_rows(rows: list[dict[str, Any]], max_k: int, limit: int) -> list[dict[str, Any]]:
    misses = [row for row in rows if not row["page_hit_at"][max_k]]
    misses.sort(
        key=lambda row: (
            row["bucket"],
            row["gold_doc_count"],
            row["gold_page_count"],
            row["question_id"],
        ),
        reverse=True,
    )
    return misses[:limit]


def _chunk_rows_from_ids(
    results: list[tuple[str, float]],
    chunks_by_id: dict[str, dict[str, Any]],
) -> list[tuple[dict[str, Any], float]]:
    rows: list[tuple[dict[str, Any], float]] = []
    for chunk_id, score in results:
        chunk = chunks_by_id.get(str(chunk_id))
        if chunk is not None:
            rows.append((chunk, float(score)))
    return rows


def build_retrieval_scorecard(
    *,
    gold_path: Path,
    questions_path: Path,
    modes: tuple[str, ...],
    top_ks: tuple[int, ...],
    details_limit: int = DEFAULT_DETAILS_LIMIT,
    limit: int | None = None,
) -> dict[str, Any]:
    questions = load_questions(questions_path)
    gold_submission = load_json(gold_path)
    gold_answers = {answer["question_id"]: answer for answer in gold_submission.get("answers", [])}
    chunks_by_id = _read_chunks_by_id()

    max_k = max(top_ks)
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

    bm25 = BM25Searcher() if "bm25" in modes else None
    semantic = SemanticSearcher() if "semantic" in modes else None
    hybrid = HybridRetriever(enable_reranker=ENABLE_RERANKER if "current" in modes else False) if (
        {"hybrid_rrf", "current"} & set(modes)
    ) else None

    mode_rows: dict[str, list[dict[str, Any]]] = {mode: [] for mode in modes}
    for question, gold_page_refs in evidence_questions:
        query = question.get("question", "")

        if bm25 is not None:
            bm25_chunks = _chunk_rows_from_ids(bm25.search(query, top_k=max_k), chunks_by_id)
            mode_rows["bm25"].append(build_retrieval_row(question, bm25_chunks, gold_page_refs, top_ks))

        if semantic is not None:
            semantic_chunks = _chunk_rows_from_ids(semantic.search(query, top_k=max_k), chunks_by_id)
            mode_rows["semantic"].append(build_retrieval_row(question, semantic_chunks, gold_page_refs, top_ks))

        if hybrid is not None and "hybrid_rrf" in modes:
            hybrid_rrf_chunks = hybrid.retrieve_without_rerank(query, top_k=max_k)
            mode_rows["hybrid_rrf"].append(build_retrieval_row(question, hybrid_rrf_chunks, gold_page_refs, top_ks))

        if hybrid is not None and "current" in modes:
            intent = detect_grounding_intent(query, question.get("answer_type", ""))
            current_chunks = hybrid.retrieve(query, rerank_top_k=max_k, intent=intent)
            mode_rows["current"].append(build_retrieval_row(question, current_chunks, gold_page_refs, top_ks))

    mode_summaries: dict[str, Any] = {}
    for mode, rows in mode_rows.items():
        mode_summaries[mode] = {
            "overall": summarize_retrieval_rows(rows, top_ks),
            "by_bucket": summarize_rows_by_key(rows, top_ks, "bucket"),
            "by_answer_type": summarize_rows_by_key(rows, top_ks, "answer_type"),
            "miss_rows": _top_miss_rows(rows, max_k, details_limit),
        }

    return {
        "gold_path": str(gold_path),
        "questions_path": str(questions_path),
        "question_count": len(evidence_questions),
        "top_ks": list(top_ks),
        "modes": list(modes),
        "reranker_enabled": ENABLE_RERANKER,
        "mode_summaries": mode_summaries,
    }


def print_retrieval_scorecard(summary: dict[str, Any], details_limit: int = DEFAULT_DETAILS_LIMIT) -> None:
    top_ks = tuple(int(value) for value in summary["top_ks"])
    max_k = max(top_ks)

    print("\nRetrieval scorecard")
    print(f"Weak gold:       {summary['gold_path']}")
    print(f"Questions:       {summary['questions_path']}")
    print(f"Evidence items:  {summary['question_count']}")
    print(f"Top-k values:    {', '.join(str(k) for k in top_ks)}")
    print(f"Modes:           {', '.join(summary['modes'])}")
    print(f"Reranker active: {'yes' if summary['reranker_enabled'] else 'no'}")

    for mode in summary["modes"]:
        mode_summary = summary["mode_summaries"][mode]
        overall = mode_summary["overall"]
        print(f"\nMode: {mode}")
        print(f"  Questions:                {overall['question_count']}")
        print(f"  Page MRR:                 {overall['page_mrr']:.4f}")
        print(f"  Doc MRR:                  {overall['doc_mrr']:.4f}")
        print(f"  Mean first page hit rank: {overall['mean_first_page_hit_rank']}")
        print(f"  Mean first doc hit rank:  {overall['mean_first_doc_hit_rank']}")
        print(f"  Misses at top-{max_k}:       {overall['misses_at_max_k']}")
        for k in top_ks:
            print(
                f"  @ {k:2d} "
                f"page_hit={overall['page_hit_at'][k]:.4f} "
                f"page_recall={overall['page_recall_at'][k]:.4f} "
                f"doc_hit={overall['doc_hit_at'][k]:.4f} "
                f"doc_full={overall['doc_full_hit_at'][k]:.4f}"
            )

        print("  By bucket")
        for bucket, bucket_summary in mode_summary["by_bucket"].items():
            print(
                f"    {bucket:15s} "
                f"page_hit@{max_k}={bucket_summary['page_hit_at'][max_k]:.4f} "
                f"page_recall@{max_k}={bucket_summary['page_recall_at'][max_k]:.4f} "
                f"doc_full@{max_k}={bucket_summary['doc_full_hit_at'][max_k]:.4f}"
            )

        print(f"  Top misses at top-{max_k}")
        miss_rows = mode_summary["miss_rows"]
        if not miss_rows:
            print("    none")
        else:
            for row in miss_rows[:details_limit]:
                print(
                    f"    {row['question_id'][:8]} "
                    f"[{row['bucket']}/{row['answer_type']}] "
                    f"gold_docs={row['gold_doc_count']} gold_pages={row['gold_page_count']} "
                    f"{row['question'][:100]}"
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate chunk text retrieval against weak-gold evidence pages")
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
        "--modes",
        default=",".join(DEFAULT_MODES),
        help=f"Comma-separated retrieval modes (default: {','.join(DEFAULT_MODES)})",
    )
    parser.add_argument(
        "--top-ks",
        default=",".join(str(k) for k in DEFAULT_TOP_KS),
        help=f"Comma-separated top-k cutoffs (default: {','.join(str(k) for k in DEFAULT_TOP_KS)})",
    )
    parser.add_argument(
        "--details-limit",
        type=int,
        default=DEFAULT_DETAILS_LIMIT,
        help=f"How many miss rows to print per mode (default: {DEFAULT_DETAILS_LIMIT})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional cap on how many evidence-bearing questions to score",
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

    try:
        modes = _parse_csv_choices(args.modes, valid_values=set(DEFAULT_MODES))
        top_ks = parse_top_ks(args.top_ks)
    except ValueError as exc:
        parser.error(str(exc))
        raise AssertionError("unreachable")

    summary = build_retrieval_scorecard(
        gold_path=gold_path,
        questions_path=questions_path,
        modes=modes,
        top_ks=top_ks,
        details_limit=max(1, args.details_limit),
        limit=args.limit,
    )
    print_retrieval_scorecard(summary, details_limit=max(1, args.details_limit))


if __name__ == "__main__":
    main()
