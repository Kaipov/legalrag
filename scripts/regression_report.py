"""
Regression harness for comparing a candidate submission against a golden baseline.

Usage:
    python -m scripts.regression_report
    python -m scripts.regression_report --strict
    python -m scripts.regression_report --baseline path/to/golden.json --candidate path/to/submission.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.compare_submissions import _bucket as grounding_bucket
from src.config import DATA_DIR, PROJECT_ROOT
from src.validation import is_null_like_answer

DEFAULT_BASELINE_PATH = PROJECT_ROOT / "golden_submission.json"
DEFAULT_CANDIDATE_PATH = PROJECT_ROOT / "submission.json"
DEFAULT_QUESTIONS_PATH = DATA_DIR / "questions.json"

STRUCTURED_TYPES = {"number", "boolean", "name", "names", "date"}
GROUNDING_BAD_THRESHOLD = 0.5
FREE_TEXT_STRONG_CHAR_THRESHOLD = 0.55
FREE_TEXT_STRONG_TOKEN_THRESHOLD = 0.45
FREE_TEXT_WEAK_CHAR_THRESHOLD = 0.35
FREE_TEXT_WEAK_TOKEN_THRESHOLD = 0.25

# These defaults are intentionally loose enough to match the current public-set baseline-v6 run
# while still catching the large regressions we saw in the v4 grounding experiments.
DEFAULT_GUARDRAILS = {
    "min_grounding_fbeta": 0.57,
    "max_structured_mismatches": 10,
    "max_free_text_weak": 5,
    "max_free_text_null_risk": 1,
    "max_answer_exact_grounding_bad": 16,
}


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_questions(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or DEFAULT_QUESTIONS_PATH
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalized_string(value: Any) -> str | None:
    if value is None:
        return None
    return " ".join(str(value).strip().split()).casefold()


def _normalized_answer(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return _normalized_string(value)
    if isinstance(value, list):
        return [_normalized_answer(item) for item in value]
    return value


def _answer_signature(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _normalized_signature(value: Any) -> str:
    return _answer_signature(_normalized_answer(value))


def _iterable_values(value: Any) -> Iterable[Any]:
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        return value
    return ()


def _normalized_set(value: Any) -> set[str]:
    return {
        item
        for item in (_normalized_string(entry) for entry in _iterable_values(value))
        if item
    }


def _structured_score(answer_type: str, baseline_answer: Any, candidate_answer: Any) -> float:
    answer_type = str(answer_type or "").lower()
    if baseline_answer is None and candidate_answer is None:
        return 1.0
    if baseline_answer is None or candidate_answer is None:
        return 0.0

    if answer_type == "number":
        try:
            baseline_num = float(baseline_answer)
            candidate_num = float(candidate_answer)
        except (TypeError, ValueError):
            return 0.0
        if baseline_num == 0:
            return 1.0 if candidate_num == 0 else 0.0
        return 1.0 if abs(candidate_num - baseline_num) / abs(baseline_num) <= 0.01 else 0.0

    if answer_type == "boolean":
        return 1.0 if baseline_answer is candidate_answer else 0.0

    if answer_type in {"name", "date"}:
        return 1.0 if _normalized_string(baseline_answer) == _normalized_string(candidate_answer) else 0.0

    if answer_type == "names":
        baseline_set = _normalized_set(baseline_answer)
        candidate_set = _normalized_set(candidate_answer)
        if not baseline_set and not candidate_set:
            return 1.0
        union = baseline_set | candidate_set
        return len(baseline_set & candidate_set) / len(union) if union else 1.0

    return 0.0


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


def _fbeta(predicted_pages: set[tuple[str, int]], golden_pages: set[tuple[str, int]], beta: float = 2.5) -> float:
    if not predicted_pages and not golden_pages:
        return 1.0
    if not predicted_pages or not golden_pages:
        return 0.0

    intersection = predicted_pages & golden_pages
    if not intersection:
        return 0.0

    precision = len(intersection) / len(predicted_pages)
    recall = len(intersection) / len(golden_pages)
    beta_sq = beta ** 2
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


def _jaccard(predicted_pages: set[tuple[str, int]], golden_pages: set[tuple[str, int]]) -> float:
    if not predicted_pages and not golden_pages:
        return 1.0
    union = predicted_pages | golden_pages
    if not union:
        return 1.0
    return len(predicted_pages & golden_pages) / len(union)


def _char_similarity(baseline_text: str, candidate_text: str) -> float:
    return SequenceMatcher(None, baseline_text or "", candidate_text or "").ratio()


def _token_set(text: str) -> set[str]:
    normalized = "".join(char.lower() if char.isalnum() else " " for char in text or "")
    return {token for token in normalized.split() if token}


def _token_jaccard(baseline_text: str, candidate_text: str) -> float:
    baseline_tokens = _token_set(baseline_text)
    candidate_tokens = _token_set(candidate_text)
    if not baseline_tokens and not candidate_tokens:
        return 1.0
    union = baseline_tokens | candidate_tokens
    if not union:
        return 1.0
    return len(baseline_tokens & candidate_tokens) / len(union)


def summarize_structured_drift(
    baseline_answers: dict[str, dict[str, Any]],
    candidate_answers: dict[str, dict[str, Any]],
    questions_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    common_ids = sorted(set(baseline_answers) & set(candidate_answers) & set(questions_by_id))
    by_type: dict[str, dict[str, float]] = {}
    mismatches: list[dict[str, Any]] = []
    total_score = 0.0
    question_count = 0

    for question_id in common_ids:
        question = questions_by_id[question_id]
        answer_type = str(question.get("answer_type", "")).lower()
        if answer_type not in STRUCTURED_TYPES:
            continue

        baseline_answer = baseline_answers[question_id].get("answer")
        candidate_answer = candidate_answers[question_id].get("answer")
        score = _structured_score(answer_type, baseline_answer, candidate_answer)
        question_count += 1
        total_score += score

        bucket = by_type.setdefault(answer_type, {"count": 0.0, "score": 0.0, "mismatches": 0.0})
        bucket["count"] += 1
        bucket["score"] += score
        if score < 1.0:
            bucket["mismatches"] += 1
            mismatches.append(
                {
                    "question_id": question_id,
                    "answer_type": answer_type,
                    "question": question.get("question", ""),
                    "score": score,
                    "baseline_answer": baseline_answer,
                    "candidate_answer": candidate_answer,
                }
            )

    by_type_summary = {
        answer_type: {
            "count": int(values["count"]),
            "score": round(values["score"] / values["count"], 4) if values["count"] else 0.0,
            "mismatches": int(values["mismatches"]),
        }
        for answer_type, values in sorted(by_type.items())
    }

    return {
        "question_count": question_count,
        "score": round(total_score / question_count, 4) if question_count else 0.0,
        "mismatches": len(mismatches),
        "by_type": by_type_summary,
        "rows": mismatches,
    }


def summarize_grounding_regressions(
    baseline_answers: dict[str, dict[str, Any]],
    candidate_answers: dict[str, dict[str, Any]],
    questions_by_id: dict[str, dict[str, Any]],
    grounding_bad_threshold: float = GROUNDING_BAD_THRESHOLD,
) -> dict[str, Any]:
    common_ids = sorted(set(baseline_answers) & set(candidate_answers) & set(questions_by_id))
    rows: list[dict[str, Any]] = []
    bucket_stats: dict[str, dict[str, float]] = {}

    for question_id in common_ids:
        question = questions_by_id[question_id]
        baseline_payload = baseline_answers[question_id]
        candidate_payload = candidate_answers[question_id]
        golden_pages = _page_ref_set(baseline_payload)
        candidate_pages = _page_ref_set(candidate_payload)
        answer_exact = _normalized_signature(baseline_payload.get("answer")) == _normalized_signature(
            candidate_payload.get("answer")
        )

        row = {
            "question_id": question_id,
            "answer_type": question.get("answer_type", ""),
            "question": question.get("question", ""),
            "bucket": grounding_bucket(question),
            "answer_exact": answer_exact,
            "golden_pages": len(golden_pages),
            "candidate_pages": len(candidate_pages),
            "fbeta": round(_fbeta(candidate_pages, golden_pages), 4),
            "jaccard": round(_jaccard(candidate_pages, golden_pages), 4),
        }
        rows.append(row)

        stats = bucket_stats.setdefault(
            row["bucket"],
            {"count": 0.0, "fbeta": 0.0, "jaccard": 0.0, "answer_exact_grounding_bad": 0.0},
        )
        stats["count"] += 1
        stats["fbeta"] += row["fbeta"]
        stats["jaccard"] += row["jaccard"]
        if answer_exact and row["fbeta"] < grounding_bad_threshold:
            stats["answer_exact_grounding_bad"] += 1

    bucket_summary = {
        bucket: {
            "count": int(values["count"]),
            "fbeta": round(values["fbeta"] / values["count"], 4) if values["count"] else 0.0,
            "jaccard": round(values["jaccard"] / values["count"], 4) if values["count"] else 0.0,
            "answer_exact_grounding_bad": int(values["answer_exact_grounding_bad"]),
        }
        for bucket, values in sorted(bucket_stats.items())
    }

    answer_exact_grounding_bad_rows = [
        row for row in rows if row["answer_exact"] and row["fbeta"] < grounding_bad_threshold
    ]

    return {
        "question_count": len(rows),
        "macro_fbeta": round(sum(row["fbeta"] for row in rows) / len(rows), 4) if rows else 0.0,
        "macro_jaccard": round(sum(row["jaccard"] for row in rows) / len(rows), 4) if rows else 0.0,
        "exact_page_set_matches": sum(1 for row in rows if row["jaccard"] == 1.0),
        "answer_exact_grounding_bad": len(answer_exact_grounding_bad_rows),
        "bucket_summary": bucket_summary,
        "rows": rows,
        "answer_exact_grounding_bad_rows": answer_exact_grounding_bad_rows,
    }


def summarize_free_text_proxy(
    baseline_answers: dict[str, dict[str, Any]],
    candidate_answers: dict[str, dict[str, Any]],
    questions_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    common_ids = sorted(set(baseline_answers) & set(candidate_answers) & set(questions_by_id))
    null_risk_rows: list[dict[str, Any]] = []
    weak_rows: list[dict[str, Any]] = []
    substantive_rows: list[dict[str, Any]] = []
    gold_null_count = 0
    safe_null_like_count = 0

    for question_id in common_ids:
        question = questions_by_id[question_id]
        answer_type = str(question.get("answer_type", "")).lower()
        if answer_type != "free_text":
            continue

        baseline_answer = baseline_answers[question_id].get("answer")
        candidate_answer = candidate_answers[question_id].get("answer")
        baseline_is_null_like = is_null_like_answer(baseline_answer)
        candidate_is_null_like = is_null_like_answer(candidate_answer)

        if baseline_is_null_like:
            gold_null_count += 1
            if candidate_is_null_like:
                safe_null_like_count += 1
                continue
            null_risk_rows.append(
                {
                    "question_id": question_id,
                    "question": question.get("question", ""),
                    "baseline_answer": baseline_answer,
                    "candidate_answer": candidate_answer,
                }
            )
            continue

        char_similarity = round(_char_similarity(str(baseline_answer or ""), str(candidate_answer or "")), 4)
        token_jaccard = round(_token_jaccard(str(baseline_answer or ""), str(candidate_answer or "")), 4)
        row = {
            "question_id": question_id,
            "question": question.get("question", ""),
            "char_similarity": char_similarity,
            "token_jaccard": token_jaccard,
            "baseline_answer": baseline_answer,
            "candidate_answer": candidate_answer,
        }
        substantive_rows.append(row)
        is_weak = (
            char_similarity < FREE_TEXT_WEAK_CHAR_THRESHOLD
            and token_jaccard < FREE_TEXT_WEAK_TOKEN_THRESHOLD
        )
        if is_weak:
            weak_rows.append(row)

    strong_count = sum(
        1
        for row in substantive_rows
        if row["char_similarity"] >= FREE_TEXT_STRONG_CHAR_THRESHOLD
        or row["token_jaccard"] >= FREE_TEXT_STRONG_TOKEN_THRESHOLD
    )
    weak_count = len(weak_rows)
    mid_count = max(0, len(substantive_rows) - strong_count - weak_count)

    return {
        "question_count": sum(
            1
            for question_id in common_ids
            if str(questions_by_id[question_id].get("answer_type", "")).lower() == "free_text"
        ),
        "gold_null_count": gold_null_count,
        "gold_null_safe_null_like_count": safe_null_like_count,
        "null_risk_count": len(null_risk_rows),
        "substantive_count": len(substantive_rows),
        "substantive_strong": strong_count,
        "substantive_mid": mid_count,
        "substantive_weak": weak_count,
        "null_risk_rows": null_risk_rows,
        "weak_rows": weak_rows,
    }


def build_regression_report(
    baseline_path: Path,
    candidate_path: Path,
    questions_path: Path | None = None,
    grounding_bad_threshold: float = GROUNDING_BAD_THRESHOLD,
) -> dict[str, Any]:
    questions = load_questions(questions_path)
    questions_by_id = {question["id"]: question for question in questions}
    baseline = load_json(baseline_path)
    candidate = load_json(candidate_path)
    baseline_answers = {answer["question_id"]: answer for answer in baseline.get("answers", [])}
    candidate_answers = {answer["question_id"]: answer for answer in candidate.get("answers", [])}

    structured = summarize_structured_drift(baseline_answers, candidate_answers, questions_by_id)
    grounding = summarize_grounding_regressions(
        baseline_answers,
        candidate_answers,
        questions_by_id,
        grounding_bad_threshold=grounding_bad_threshold,
    )
    free_text = summarize_free_text_proxy(baseline_answers, candidate_answers, questions_by_id)

    return {
        "baseline_path": str(baseline_path),
        "candidate_path": str(candidate_path),
        "questions_path": str(questions_path or DEFAULT_QUESTIONS_PATH),
        "structured": structured,
        "grounding": grounding,
        "free_text": free_text,
    }


def evaluate_guardrails(summary: dict[str, Any], thresholds: dict[str, float | int]) -> list[str]:
    violations: list[str] = []
    if summary["grounding"]["macro_fbeta"] < float(thresholds["min_grounding_fbeta"]):
        violations.append(
            f"grounding macro F-beta {summary['grounding']['macro_fbeta']:.4f} "
            f"< {float(thresholds['min_grounding_fbeta']):.4f}"
        )
    if summary["structured"]["mismatches"] > int(thresholds["max_structured_mismatches"]):
        violations.append(
            f"structured mismatches {summary['structured']['mismatches']} "
            f"> {int(thresholds['max_structured_mismatches'])}"
        )
    if summary["free_text"]["substantive_weak"] > int(thresholds["max_free_text_weak"]):
        violations.append(
            f"free-text weak cases {summary['free_text']['substantive_weak']} "
            f"> {int(thresholds['max_free_text_weak'])}"
        )
    if summary["free_text"]["null_risk_count"] > int(thresholds["max_free_text_null_risk"]):
        violations.append(
            f"free-text null-risk cases {summary['free_text']['null_risk_count']} "
            f"> {int(thresholds['max_free_text_null_risk'])}"
        )
    if summary["grounding"]["answer_exact_grounding_bad"] > int(thresholds["max_answer_exact_grounding_bad"]):
        violations.append(
            f"answer-exact but grounding-bad cases {summary['grounding']['answer_exact_grounding_bad']} "
            f"> {int(thresholds['max_answer_exact_grounding_bad'])}"
        )
    return violations


def _print_rows(title: str, rows: list[dict[str, Any]], limit: int = 8) -> None:
    print(f"\n{title}")
    if not rows:
        print("  none")
        return
    for row in rows[:limit]:
        print(f"  {row['question_id'][:8]} [{row.get('answer_type', 'free_text')}] {row['question'][:100]}")


def print_report(summary: dict[str, Any], thresholds: dict[str, float | int], strict: bool = False) -> None:
    structured = summary["structured"]
    grounding = summary["grounding"]
    free_text = summary["free_text"]

    print("\nRegression report")
    print(f"Baseline:  {summary['baseline_path']}")
    print(f"Candidate: {summary['candidate_path']}")
    print(f"Questions: {summary['questions_path']}")

    print("\nStructured")
    print(f"  Questions:   {structured['question_count']}")
    print(f"  Score:       {structured['score']:.4f}")
    print(f"  Mismatches:  {structured['mismatches']}")
    for answer_type, values in structured["by_type"].items():
        print(
            f"  {answer_type:8s} score={values['score']:.4f} "
            f"mismatches={values['mismatches']:2d}/{values['count']}"
        )

    print("\nGrounding")
    print(f"  Questions:                   {grounding['question_count']}")
    print(f"  Macro F-beta:                {grounding['macro_fbeta']:.4f}")
    print(f"  Macro Jaccard:               {grounding['macro_jaccard']:.4f}")
    print(f"  Exact page-set matches:      {grounding['exact_page_set_matches']}")
    print(f"  Answer exact, grounding bad: {grounding['answer_exact_grounding_bad']}")
    for bucket, values in grounding["bucket_summary"].items():
        print(
            f"  {bucket:15s} fbeta={values['fbeta']:.4f} "
            f"exact_bad={values['answer_exact_grounding_bad']:2d}/{values['count']}"
        )

    print("\nFree-text proxy")
    print(f"  Questions:             {free_text['question_count']}")
    print(f"  Gold null-like:        {free_text['gold_null_count']}")
    print(f"  Safe null-like match:  {free_text['gold_null_safe_null_like_count']}")
    print(f"  Null-risk cases:       {free_text['null_risk_count']}")
    print(f"  Substantive strong:    {free_text['substantive_strong']}")
    print(f"  Substantive mid:       {free_text['substantive_mid']}")
    print(f"  Substantive weak:      {free_text['substantive_weak']}")

    violations = evaluate_guardrails(summary, thresholds)
    print("\nGuardrails")
    print(f"  Min grounding F-beta:              {float(thresholds['min_grounding_fbeta']):.4f}")
    print(f"  Max structured mismatches:         {int(thresholds['max_structured_mismatches'])}")
    print(f"  Max free-text weak cases:          {int(thresholds['max_free_text_weak'])}")
    print(f"  Max free-text null-risk cases:     {int(thresholds['max_free_text_null_risk'])}")
    print(f"  Max answer exact, grounding bad:   {int(thresholds['max_answer_exact_grounding_bad'])}")
    if violations:
        for violation in violations:
            print(f"  FAIL: {violation}")
    else:
        print("  PASS: all guardrails satisfied")

    _print_rows("Structured mismatches", structured["rows"])
    _print_rows("Answer exact but grounding bad", grounding["answer_exact_grounding_bad_rows"])
    _print_rows("Free-text weak cases", free_text["weak_rows"])
    _print_rows("Free-text null-risk cases", free_text["null_risk_rows"])

    if strict and violations:
        print("\nStrict mode failed.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Regression harness against the golden public-set submission")
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
    parser.add_argument(
        "--questions",
        default=str(DEFAULT_QUESTIONS_PATH),
        help=f"Path to the questions JSON (default: {DEFAULT_QUESTIONS_PATH.name})",
    )
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when any guardrail is violated")
    parser.add_argument(
        "--min-grounding-fbeta",
        type=float,
        default=DEFAULT_GUARDRAILS["min_grounding_fbeta"],
        help="Minimum acceptable macro grounding F-beta in strict mode",
    )
    parser.add_argument(
        "--max-structured-mismatches",
        type=int,
        default=DEFAULT_GUARDRAILS["max_structured_mismatches"],
        help="Maximum acceptable structured mismatches in strict mode",
    )
    parser.add_argument(
        "--max-free-text-weak",
        type=int,
        default=DEFAULT_GUARDRAILS["max_free_text_weak"],
        help="Maximum acceptable weak substantive free-text proxy cases in strict mode",
    )
    parser.add_argument(
        "--max-free-text-null-risk",
        type=int,
        default=DEFAULT_GUARDRAILS["max_free_text_null_risk"],
        help="Maximum acceptable risky free-text null cases in strict mode",
    )
    parser.add_argument(
        "--max-answer-exact-grounding-bad",
        type=int,
        default=DEFAULT_GUARDRAILS["max_answer_exact_grounding_bad"],
        help="Maximum acceptable answer-correct but grounding-bad cases in strict mode",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    baseline_path = Path(args.baseline)
    candidate_path = Path(args.candidate)
    questions_path = Path(args.questions)
    if not baseline_path.exists():
        parser.error(f"baseline submission not found at {baseline_path}")
    if not candidate_path.exists():
        parser.error(f"candidate submission not found at {candidate_path}")
    if not questions_path.exists():
        parser.error(f"questions file not found at {questions_path}")

    summary = build_regression_report(
        baseline_path,
        candidate_path,
        questions_path,
    )
    thresholds = {
        "min_grounding_fbeta": args.min_grounding_fbeta,
        "max_structured_mismatches": args.max_structured_mismatches,
        "max_free_text_weak": args.max_free_text_weak,
        "max_free_text_null_risk": args.max_free_text_null_risk,
        "max_answer_exact_grounding_bad": args.max_answer_exact_grounding_bad,
    }
    print_report(summary, thresholds, strict=args.strict)

    if args.strict and evaluate_guardrails(summary, thresholds):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
