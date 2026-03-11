"""
Local evaluation: compare submission answers against public dataset.

Note: We don't have gold answers for the public set, so this script
analyzes the submission for:
- Answer type correctness (does the format match?)
- Telemetry completeness (are all fields valid?)
- Grounding coverage (how many pages per answer?)
- Null detection stats

Usage:
    python -m scripts.evaluate
    python -m scripts.evaluate --submission path/to/submission.json
"""
import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import PROJECT_ROOT, DATA_DIR


def load_submission(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_questions(path: Path | None = None) -> list[dict]:
    path = path or DATA_DIR / "questions.json"
    if not path.exists():
        path = DATA_DIR / "public_dataset.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_telemetry(answer: dict) -> list[str]:
    """Check telemetry validity. Returns list of issues."""
    issues = []
    tel = answer.get("telemetry", {})

    # Timing
    timing = tel.get("timing", {})
    if not timing:
        issues.append("missing timing")
    else:
        ttft = timing.get("ttft_ms", -1)
        total = timing.get("total_time_ms", -1)
        if ttft < 0:
            issues.append("negative ttft_ms")
        if total < 0:
            issues.append("negative total_time_ms")
        if ttft > total and total > 0:
            issues.append(f"ttft_ms ({ttft}) > total_time_ms ({total})")

    # Usage
    usage = tel.get("usage", {})
    if not usage:
        issues.append("missing usage")
    else:
        if usage.get("input_tokens", 0) <= 0:
            issues.append("zero input_tokens")
        if usage.get("output_tokens", 0) <= 0:
            issues.append("zero output_tokens")

    # Retrieval
    retrieval = tel.get("retrieval", {})
    chunks = retrieval.get("retrieved_chunk_pages", [])
    if answer.get("answer") is not None and not chunks:
        issues.append("non-null answer but empty retrieved_chunk_pages")

    return issues


def evaluate(submission_path: Path):
    """Run local evaluation."""
    sub = load_submission(submission_path)

    try:
        questions = load_questions()
        q_by_id = {q["id"]: q for q in questions}
    except FileNotFoundError:
        q_by_id = {}

    answers = sub.get("answers", [])
    print(f"\n{'='*60}")
    print(f"SUBMISSION ANALYSIS")
    print(f"{'='*60}")
    print(f"Architecture: {sub.get('architecture_summary', 'N/A')[:80]}")
    print(f"Total answers: {len(answers)}")
    print(f"Questions loaded: {len(q_by_id)}")

    # Stats
    type_counts = {}
    null_count = 0
    telemetry_issues = 0
    ttft_values = []
    total_pages = 0
    pages_per_answer = []

    for ans in answers:
        qid = ans["question_id"]
        answer_val = ans.get("answer")
        tel = ans.get("telemetry", {})

        # Answer type
        q = q_by_id.get(qid, {})
        at = q.get("answer_type", "unknown")
        type_counts[at] = type_counts.get(at, 0) + 1

        # Null check
        if answer_val is None:
            null_count += 1

        # Telemetry validation
        issues = validate_telemetry(ans)
        if issues:
            telemetry_issues += 1

        # TTFT
        timing = tel.get("timing", {})
        ttft = timing.get("ttft_ms", 0)
        if ttft > 0:
            ttft_values.append(ttft)

        # Grounding
        retrieval = tel.get("retrieval", {})
        chunks = retrieval.get("retrieved_chunk_pages", [])
        n_pages = sum(len(c.get("page_numbers", [])) for c in chunks)
        total_pages += n_pages
        pages_per_answer.append(n_pages)

    # Print results
    print(f"\n--- Answer Types ---")
    for at, count in sorted(type_counts.items()):
        print(f"  {at}: {count}")

    print(f"\n--- Null Answers ---")
    print(f"  Null: {null_count} / {len(answers)} ({100*null_count/max(len(answers),1):.1f}%)")

    print(f"\n--- Telemetry ---")
    print(f"  Valid: {len(answers) - telemetry_issues} / {len(answers)}")
    print(f"  Issues: {telemetry_issues} (each costs 0.1 multiplier)")

    if ttft_values:
        import statistics
        median_ttft = statistics.median(ttft_values)
        mean_ttft = statistics.mean(ttft_values)
        print(f"\n--- TTFT ---")
        print(f"  Median: {median_ttft:.0f}ms")
        print(f"  Mean: {mean_ttft:.0f}ms")
        print(f"  Min: {min(ttft_values):.0f}ms")
        print(f"  Max: {max(ttft_values):.0f}ms")

        # TTFT factor estimation
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
        import statistics
        print(f"\n--- Grounding ---")
        print(f"  Total pages cited: {total_pages}")
        print(f"  Mean pages/answer: {statistics.mean(pages_per_answer):.1f}")
        print(f"  Median pages/answer: {statistics.median(pages_per_answer):.0f}")
        print(f"  Answers with 0 pages: {pages_per_answer.count(0)}")

    print(f"\n{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Local submission evaluation")
    parser.add_argument("--submission", type=str, default=str(PROJECT_ROOT / "submission.json"))
    args = parser.parse_args()

    evaluate(Path(args.submission))


if __name__ == "__main__":
    main()
