"""
Repair suspicious structured answers in an existing submission without rerunning the model.

This currently focuses on `name` and `names` answers, applying the same cleanup heuristics
used at generation time.

Usage:
    python -m scripts.repair_submission --submission submission.json
    python -m scripts.repair_submission --submission submission.json --output submission.repaired.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DATA_DIR, PROJECT_ROOT
from src.generate.parse import parse_answer


def load_json(path: Path) -> dict | list:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _answer_to_raw_text(answer) -> str:
    if answer is None:
        return ""
    if isinstance(answer, list):
        return "; ".join(str(item) for item in answer)
    return str(answer)


def repair_submission(submission_path: Path, questions_path: Path, output_path: Path) -> tuple[int, int]:
    submission = load_json(submission_path)
    questions = load_json(questions_path)
    question_by_id = {question["id"]: question for question in questions}

    answers = submission.get("answers", [])
    changed = 0
    skipped = 0

    for answer_payload in answers:
        question = question_by_id.get(answer_payload.get("question_id"))
        if question is None:
            skipped += 1
            continue

        answer_type = str(question.get("answer_type", "")).lower()
        if answer_type not in {"name", "names"}:
            continue

        original_answer = answer_payload.get("answer")
        if original_answer is None:
            skipped += 1
            continue

        repaired_answer = parse_answer(
            _answer_to_raw_text(original_answer),
            answer_type,
            question_text=question.get("question", ""),
        )
        if repaired_answer != original_answer:
            answer_payload["answer"] = repaired_answer
            changed += 1

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(submission, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    return changed, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair suspicious name/names answers in a submission JSON")
    parser.add_argument("--submission", type=str, default=str(PROJECT_ROOT / "submission.json"))
    parser.add_argument("--questions", type=str, default=str(DATA_DIR / "questions.json"))
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "submission.repaired.json"))
    args = parser.parse_args()

    changed, skipped = repair_submission(
        Path(args.submission),
        Path(args.questions),
        Path(args.output),
    )
    print(f"Repaired {changed} answers")
    print(f"Skipped {skipped} answers")
    print(f"Wrote repaired submission to {args.output}")


if __name__ == "__main__":
    main()
