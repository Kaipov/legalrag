"""
Compare grounding density between two submissions.

Usage:
    python -m scripts.compare_submissions
    python -m scripts.compare_submissions --baseline path/to/old.json --candidate path/to/new.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DATA_DIR, PROJECT_ROOT

try:
    from src.retrieve.grounding_policy import detect_grounding_intent as _detect_grounding_intent
except ImportError:
    _detect_grounding_intent = None

DEFAULT_BASELINE_PATH = PROJECT_ROOT / "golden_submission.json"
DEFAULT_CANDIDATE_PATH = PROJECT_ROOT / "submission.json"


def load_json(path: Path) -> dict | list:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_questions(path: Path | None = None) -> list[dict]:
    path = path or DATA_DIR / "questions.json"
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _page_count(answer: dict) -> int:
    retrieval = answer.get("telemetry", {}).get("retrieval", {})
    refs = retrieval.get("retrieved_chunk_pages", [])
    if not isinstance(refs, list):
        return 0
    return sum(len(ref.get("page_numbers", [])) for ref in refs if isinstance(ref, dict))


def _is_compare_question(text: str) -> bool:
    compare_markers = (
        "both case",
        "both cases",
        "across all documents",
        "across case",
        "common to both",
        "in both cases",
        "appeared in both",
        "involve any of the same",
    )
    return any(marker in text for marker in compare_markers)


def _fallback_bucket(question_text: str, answer_type: str) -> str:
    text = question_text.lower()
    answer_type = answer_type.lower()

    if any(marker in text for marker in ("title page", "cover page", "header/caption", "header", "caption")):
        return "title_page"
    if any(marker in text for marker in ("date of issue", "issue date", "issued first", "earlier issue date")):
        return "date_of_issue"
    if any(marker in text for marker in ("last page", "conclusion", "it is hereby ordered that")):
        return "last_page"
    if "judge" in text and _is_compare_question(text):
        return "judge_compare"
    if answer_type in {"boolean", "name"} and any(
        marker in text for marker in ("party", "parties", "claimant", "defendant", "main party")
    ) and _is_compare_question(text):
        return "party_compare"
    return "generic"


def _bucket(question: dict) -> str:
    question_text = question.get("question", "")
    answer_type = question.get("answer_type", "")
    if _detect_grounding_intent is not None:
        return _detect_grounding_intent(question_text, answer_type).kind
    return _fallback_bucket(question_text, answer_type)


def _summarize(values: list[int]) -> tuple[float, float, int]:
    if not values:
        return 0.0, 0.0, 0
    return statistics.mean(values), statistics.median(values), sum(values)


def compare_submissions(baseline_path: Path, candidate_path: Path, questions_path: Path | None = None) -> int:
    questions = load_questions(questions_path)
    q_by_id = {question["id"]: question for question in questions}

    baseline = load_json(baseline_path)
    candidate = load_json(candidate_path)
    base_answers = {answer["question_id"]: answer for answer in baseline.get("answers", [])}
    cand_answers = {answer["question_id"]: answer for answer in candidate.get("answers", [])}

    common_ids = sorted(set(base_answers) & set(cand_answers) & set(q_by_id))
    if not common_ids:
        print("No overlapping question_ids found.")
        return 1

    overall_base: list[int] = []
    overall_cand: list[int] = []
    bucket_values: dict[str, dict[str, list[int]]] = {}
    largest_drops: list[tuple[int, str, int, int, str]] = []

    for question_id in common_ids:
        question = q_by_id[question_id]
        bucket = _bucket(question)
        base_pages = _page_count(base_answers[question_id])
        cand_pages = _page_count(cand_answers[question_id])
        overall_base.append(base_pages)
        overall_cand.append(cand_pages)
        bucket_values.setdefault(bucket, {"baseline": [], "candidate": []})["baseline"].append(base_pages)
        bucket_values[bucket]["candidate"].append(cand_pages)
        largest_drops.append((base_pages - cand_pages, question_id, base_pages, cand_pages, question["question"]))

    base_mean, base_median, base_total = _summarize(overall_base)
    cand_mean, cand_median, cand_total = _summarize(overall_cand)

    print("\nGrounding comparison")
    print(f"Questions compared: {len(common_ids)}")
    print(f"Baseline total pages:  {base_total}")
    print(f"Candidate total pages: {cand_total}")
    print(f"Delta total pages:     {cand_total - base_total:+d}")
    print(f"Baseline mean/median:  {base_mean:.2f} / {base_median:.1f}")
    print(f"Candidate mean/median: {cand_mean:.2f} / {cand_median:.1f}")

    print("\nBy intent")
    for bucket in sorted(bucket_values):
        bucket_base = bucket_values[bucket]["baseline"]
        bucket_cand = bucket_values[bucket]["candidate"]
        mean_base, median_base, total_base = _summarize(bucket_base)
        mean_cand, median_cand, total_cand = _summarize(bucket_cand)
        print(
            f"  {bucket:15s} baseline={total_base:4d} ({mean_base:.2f}/{median_base:.1f}) "
            f"candidate={total_cand:4d} ({mean_cand:.2f}/{median_cand:.1f}) delta={total_cand - total_base:+d}"
        )

    print("\nLargest page drops")
    for drop, question_id, base_pages, cand_pages, question_text in sorted(largest_drops, reverse=True)[:15]:
        if drop <= 0:
            continue
        print(f"  {question_id[:8]} {base_pages}->{cand_pages}: {question_text[:120]}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare grounding density between two submissions")
    parser.add_argument(
        "--baseline",
        default=str(DEFAULT_BASELINE_PATH),
        help=f"Path to the baseline submission JSON (default: {DEFAULT_BASELINE_PATH.name})",
    )
    parser.add_argument(
        "--candidate",
        default=str(DEFAULT_CANDIDATE_PATH),
        help=f"Path to the candidate submission JSON (default: {DEFAULT_CANDIDATE_PATH.name})",
    )
    parser.add_argument("--questions", default=str(DATA_DIR / "questions.json"), help="Path to the questions JSON")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    baseline_path = Path(args.baseline)
    candidate_path = Path(args.candidate)
    if not baseline_path.exists():
        parser.error(f"baseline submission not found at {baseline_path}")
    if not candidate_path.exists():
        parser.error(f"candidate submission not found at {candidate_path}")

    raise SystemExit(
        compare_submissions(
            baseline_path,
            candidate_path,
            Path(args.questions),
        )
    )


if __name__ == "__main__":
    main()
