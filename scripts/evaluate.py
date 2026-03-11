"""
Local evaluation: validate submission format and telemetry against the public dataset schema.

This script does not score answer correctness because public gold answers are not available.

Usage:
    python -m scripts.evaluate
    python -m scripts.evaluate --submission path/to/submission.json
    python -m scripts.evaluate --strict
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import PROJECT_ROOT, DATA_DIR
from src.validation import is_null_like_answer, validate_answer_value, validate_telemetry_payload


def load_submission(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_questions(path: Path | None = None) -> list[dict]:
    path = path or DATA_DIR / "questions.json"
    if not path.exists():
        path = DATA_DIR / "public_dataset.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate(submission_path: Path) -> int:
    """Run local validation and return a process-style exit code."""
    sub = load_submission(submission_path)

    try:
        questions = load_questions()
        q_by_id = {q["id"]: q for q in questions}
    except FileNotFoundError:
        q_by_id = {}

    answers = sub.get("answers", [])
    print(f"\n{'=' * 60}")
    print("SUBMISSION ANALYSIS")
    print(f"{'=' * 60}")
    print(f"Architecture: {sub.get('architecture_summary', 'N/A')[:120]}")
    print(f"Total answers: {len(answers)}")
    print(f"Questions loaded: {len(q_by_id)}")

    seen_ids = set()
    duplicate_ids = set()
    unexpected_ids = set()

    for ans in answers:
        qid = ans["question_id"]
        if qid in seen_ids:
            duplicate_ids.add(qid)
        seen_ids.add(qid)
        if q_by_id and qid not in q_by_id:
            unexpected_ids.add(qid)

    missing_ids = set(q_by_id) - seen_ids if q_by_id else set()

    type_counts: dict[str, int] = {}
    null_count = 0
    format_issues = 0
    telemetry_issues = 0
    ttft_values = []
    total_pages = 0
    pages_per_answer = []

    for ans in answers:
        qid = ans["question_id"]
        answer_val = ans.get("answer")
        tel = ans.get("telemetry", {})

        q = q_by_id.get(qid, {})
        answer_type = q.get("answer_type", "unknown")
        type_counts[answer_type] = type_counts.get(answer_type, 0) + 1

        if is_null_like_answer(answer_val):
            null_count += 1

        if q:
            answer_issues = validate_answer_value(answer_val, answer_type)
            if answer_issues:
                format_issues += 1

        telemetry_errors = validate_telemetry_payload(ans)
        if telemetry_errors:
            telemetry_issues += 1

        timing = tel.get("timing", {})
        ttft = timing.get("ttft_ms", 0)
        if isinstance(ttft, (int, float)) and ttft > 0:
            ttft_values.append(ttft)

        retrieval = tel.get("retrieval", {})
        chunks = retrieval.get("retrieved_chunk_pages", [])
        if isinstance(chunks, list):
            n_pages = sum(len(c.get("page_numbers", [])) for c in chunks if isinstance(c, dict))
            total_pages += n_pages
            pages_per_answer.append(n_pages)
        else:
            pages_per_answer.append(0)

    print("\n--- Submission Shape ---")
    print(f"  Missing answers: {len(missing_ids)}")
    print(f"  Duplicate answers: {len(duplicate_ids)}")
    print(f"  Unexpected question_ids: {len(unexpected_ids)}")

    print("\n--- Answer Types ---")
    for answer_type, count in sorted(type_counts.items()):
        print(f"  {answer_type}: {count}")

    print("\n--- Null Answers ---")
    print(f"  Null: {null_count} / {len(answers)} ({100 * null_count / max(len(answers), 1):.1f}%)")

    print("\n--- Answer Format ---")
    print(f"  Valid: {len(answers) - format_issues} / {len(answers)}")
    print(f"  Issues: {format_issues}")

    print("\n--- Telemetry ---")
    print(f"  Valid: {len(answers) - telemetry_issues} / {len(answers)}")
    print(f"  Issues: {telemetry_issues}")

    if ttft_values:
        median_ttft = statistics.median(ttft_values)
        mean_ttft = statistics.mean(ttft_values)
        print("\n--- TTFT ---")
        print(f"  Median: {median_ttft:.0f}ms")
        print(f"  Mean: {mean_ttft:.0f}ms")
        print(f"  Min: {min(ttft_values):.0f}ms")
        print(f"  Max: {max(ttft_values):.0f}ms")

        if median_ttft < 1000:
            factor = 1.05
        elif median_ttft < 2000:
            factor = 1.02
        elif median_ttft < 3000:
            factor = 1.00
        else:
            factor = 0.85
        print(f"  Estimated TTFT factor: {factor}")

    if pages_per_answer:
        print("\n--- Grounding ---")
        print(f"  Total pages cited: {total_pages}")
        print(f"  Mean pages/answer: {statistics.mean(pages_per_answer):.1f}")
        print(f"  Median pages/answer: {statistics.median(pages_per_answer):.0f}")
        print(f"  Answers with 0 pages: {pages_per_answer.count(0)}")

    total_issues = len(missing_ids) + len(duplicate_ids) + len(unexpected_ids) + format_issues + telemetry_issues
    print(f"\n{'=' * 60}")
    return 0 if total_issues == 0 else 1


def main():
    parser = argparse.ArgumentParser(description="Local submission evaluation")
    parser.add_argument("--submission", type=str, default=str(PROJECT_ROOT / "submission.json"))
    parser.add_argument("--strict", action="store_true", help="Exit with code 1 if validation issues are found")
    args = parser.parse_args()

    exit_code = evaluate(Path(args.submission))
    if args.strict:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
