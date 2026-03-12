"""
Compare answer drift between two submissions.

Usage:
    python -m scripts.compare_answers
    python -m scripts.compare_answers --baseline path/to/old.json --candidate path/to/new.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DATA_DIR, PROJECT_ROOT
from src.validation import is_null_like_answer

DEFAULT_BASELINE_PATH = PROJECT_ROOT / "golden_submission.json"
DEFAULT_CANDIDATE_PATH = PROJECT_ROOT / "submission.json"


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_questions(path: Path | None = None) -> list[dict]:
    path = path or DATA_DIR / "questions.json"
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _answer_signature(answer: Any) -> str:
    return json.dumps(answer, ensure_ascii=False, sort_keys=True)


def summarize_answer_changes(
    baseline_answers: dict[str, dict],
    candidate_answers: dict[str, dict],
    questions_by_id: dict[str, dict],
) -> dict[str, Any]:
    common_ids = sorted(set(baseline_answers) & set(candidate_answers) & set(questions_by_id))
    changed_rows: list[dict[str, Any]] = []
    by_type: dict[str, int] = {}
    baseline_null_like = 0
    candidate_null_like = 0
    null_regressions = 0

    for question_id in common_ids:
        question = questions_by_id[question_id]
        answer_type = question.get("answer_type", "unknown")
        baseline_answer = baseline_answers[question_id].get("answer")
        candidate_answer = candidate_answers[question_id].get("answer")

        baseline_is_null = is_null_like_answer(baseline_answer)
        candidate_is_null = is_null_like_answer(candidate_answer)
        if baseline_is_null:
            baseline_null_like += 1
        if candidate_is_null:
            candidate_null_like += 1
        if not baseline_is_null and candidate_is_null:
            null_regressions += 1

        if _answer_signature(baseline_answer) == _answer_signature(candidate_answer):
            continue

        by_type[answer_type] = by_type.get(answer_type, 0) + 1
        changed_rows.append(
            {
                "question_id": question_id,
                "answer_type": answer_type,
                "question": question.get("question", ""),
                "baseline_answer": baseline_answer,
                "candidate_answer": candidate_answer,
                "baseline_null_like": baseline_is_null,
                "candidate_null_like": candidate_is_null,
            }
        )

    return {
        "questions_compared": len(common_ids),
        "changed": len(changed_rows),
        "unchanged": len(common_ids) - len(changed_rows),
        "baseline_null_like": baseline_null_like,
        "candidate_null_like": candidate_null_like,
        "null_regressions": null_regressions,
        "by_type": dict(sorted(by_type.items())),
        "changed_rows": changed_rows,
    }


def compare_answers(baseline_path: Path, candidate_path: Path, questions_path: Path | None = None) -> int:
    questions = load_questions(questions_path)
    questions_by_id = {question["id"]: question for question in questions}
    baseline = load_json(baseline_path)
    candidate = load_json(candidate_path)
    baseline_answers = {answer["question_id"]: answer for answer in baseline.get("answers", [])}
    candidate_answers = {answer["question_id"]: answer for answer in candidate.get("answers", [])}

    summary = summarize_answer_changes(baseline_answers, candidate_answers, questions_by_id)
    print("\nAnswer comparison")
    print(f"Questions compared:     {summary['questions_compared']}")
    print(f"Unchanged answers:      {summary['unchanged']}")
    print(f"Changed answers:        {summary['changed']}")
    print(f"Baseline null-like:     {summary['baseline_null_like']}")
    print(f"Candidate null-like:    {summary['candidate_null_like']}")
    print(f"New null regressions:   {summary['null_regressions']}")

    print("\nChanged by answer type")
    if summary["by_type"]:
        for answer_type, count in summary["by_type"].items():
            print(f"  {answer_type}: {count}")
    else:
        print("  none")

    print("\nChanged answers")
    for row in summary["changed_rows"][:20]:
        print(f"  {row['question_id'][:8]} [{row['answer_type']}] {row['question'][:100]}")
        print(f"    baseline:  {_answer_signature(row['baseline_answer'])}")
        print(f"    candidate: {_answer_signature(row['candidate_answer'])}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare answer drift between two submissions")
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

    raise SystemExit(compare_answers(baseline_path, candidate_path, Path(args.questions)))


if __name__ == "__main__":
    main()
