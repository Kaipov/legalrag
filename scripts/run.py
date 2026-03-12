"""
Main execution script: download data, answer all questions, submit.

Usage:
    python -m scripts.run                     # full pipeline (download + answer + submit)
    python -m scripts.run --no-download       # skip download (use existing data)
    python -m scripts.run --no-submit         # answer but don't submit
    python -m scripts.run --dry-run           # answer first 5 questions only
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter_kit"))

from src.config import (
    DATA_DIR,
    DOCUMENTS_DIR,
    ENABLE_RERANKER,
    EVAL_API_KEY,
    GENERATION_MODEL,
    GENERATION_TOP_K,
    PROJECT_ROOT,
    ensure_dirs,
)
from src.pipeline import RAGPipeline

from arlc import EvaluationClient, SubmissionBuilder


def download_data(client: EvaluationClient) -> list[dict]:
    """Download questions and documents from API."""
    questions_path = DATA_DIR / "questions.json"
    print("Downloading questions...")
    questions = client.download_questions(target_path=str(questions_path))
    print(f"Downloaded {len(questions)} questions")

    print("Downloading documents...")
    client.download_documents(str(DOCUMENTS_DIR))
    print("Documents downloaded and extracted")

    return questions


def load_questions(path: Path | None = None) -> list[dict]:
    """Load questions from local JSON file."""
    path = path or DATA_DIR / "questions.json"
    if not path.exists():
        raise FileNotFoundError(f"No questions file found at {path}")

    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


_EXCLUDE_DIRS = {"__pycache__", "data", "index", "storage", ".venv", "venv", "env", ".git", "node_modules"}
_EXCLUDE_FILES = {".env", "submission.json", "golden_submission.json", "questions.json", "code_archive.zip"}


def create_code_archive(archive_path: Path) -> Path:
    """Create code archive for submission."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_resolved = archive_path.resolve()

    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        for file_path in PROJECT_ROOT.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.resolve() == archive_resolved:
                continue
            rel_parts = set(file_path.relative_to(PROJECT_ROOT).parts)
            if rel_parts & _EXCLUDE_DIRS:
                continue
            if file_path.name in _EXCLUDE_FILES:
                continue
            if file_path.stat().st_size > 10_000_000:
                continue
            archive.write(file_path, file_path.relative_to(PROJECT_ROOT))

    print(f"Code archive created: {archive_path} ({archive_path.stat().st_size / 1024:.0f} KB)")
    return archive_path


def build_architecture_summary() -> str:
    """Describe the currently active pipeline for the submission metadata."""
    retrieval_summary = "BM25+bge-m3 hybrid search with RRF fusion"
    if ENABLE_RERANKER:
        retrieval_summary += ", bge-reranker-v2-m3 cross-encoder rerank"
    else:
        retrieval_summary += ", reranker disabled"

    return (
        "Hybrid RAG: pdfplumber+PaddleOCR extraction, structure-aware chunking, "
        f"{retrieval_summary}, "
        f"{GENERATION_MODEL} streaming generation over adaptive top-k chunks (up to {GENERATION_TOP_K}), "
        "answer-aware grounding from cited source blocks, "
        "3-tier null detection."
    )


def main():
    parser = argparse.ArgumentParser(description="ARLC RAG pipeline - answer and submit")
    parser.add_argument("--no-download", action="store_true", help="Skip data download")
    parser.add_argument("--no-submit", action="store_true", help="Don't submit to API")
    parser.add_argument("--dry-run", action="store_true", help="Answer only first 5 questions")
    parser.add_argument("--questions", type=str, help="Path to questions JSON file")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    ensure_dirs()

    client = None
    if not args.no_download and not args.dry_run:
        if not EVAL_API_KEY:
            print("WARNING: EVAL_API_KEY not set. Skipping download.")
        else:
            try:
                client = EvaluationClient.from_env()
                questions = download_data(client)
            except requests.HTTPError as exc:
                status_code = getattr(getattr(exc, "response", None), "status_code", None)
                if status_code == 401:
                    print("WARNING: Evaluation API rejected EVAL_API_KEY (401). Falling back to local data.")
                else:
                    print(f"WARNING: Download failed with HTTP {status_code}. Falling back to local data.")
                logging.warning("Download failed: %s", exc)
                client = None
            except Exception as exc:
                print(f"WARNING: Download failed ({exc}). Falling back to local data.")
                logging.warning("Download failed: %s", exc)
                client = None

    if args.questions:
        questions = load_questions(Path(args.questions))
    else:
        try:
            questions = load_questions()
        except FileNotFoundError:
            print("ERROR: No questions file found at data/questions.json. Run without --no-download or provide --questions path.")
            sys.exit(1)

    if args.dry_run:
        questions = questions[:5]
        print(f"\n[DRY RUN] Answering {len(questions)} questions only")

    print(f"\nLoaded {len(questions)} questions")

    print("\nInitializing RAG pipeline...")
    pipeline = RAGPipeline()

    print(f"\nAnswering {len(questions)} questions...\n")
    builder = SubmissionBuilder(architecture_summary=build_architecture_summary())

    for i, question_item in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {question_item['id'][:12]}... ({question_item.get('answer_type', '?')})")
        started_at = time.perf_counter()

        try:
            answer = pipeline.answer_question(question_item)
            builder.add_answer(answer)
        except Exception as exc:
            logging.error(f"Error on question {question_item['id']}: {exc}")
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            from arlc import SubmissionAnswer, Telemetry, TimingMetrics, UsageMetrics

            builder.add_answer(SubmissionAnswer(
                question_id=question_item["id"],
                answer=None,
                telemetry=Telemetry(
                    timing=TimingMetrics(ttft_ms=0, tpot_ms=0, total_time_ms=elapsed_ms),
                    retrieval=[],
                    usage=UsageMetrics(input_tokens=0, output_tokens=0),
                    model_name=GENERATION_MODEL,
                ),
            ))

    submission_path = PROJECT_ROOT / "submission.json"
    builder.save(str(submission_path))
    print(f"\nSubmission saved to {submission_path}")

    if not args.no_submit and not args.dry_run:
        if client is None:
            if EVAL_API_KEY:
                client = EvaluationClient.from_env()
            else:
                print("WARNING: EVAL_API_KEY not set. Cannot submit.")
                return

        archive_path = PROJECT_ROOT / "code_archive.zip"
        create_code_archive(archive_path)

        print("\nSubmitting...")
        try:
            response = client.submit_submission(str(submission_path), str(archive_path))
            print(f"Submission response: {json.dumps(response, indent=2)}")
        except Exception as exc:
            print(f"Submission failed: {exc}")
    else:
        print("\nSkipping submission (--no-submit or --dry-run)")


if __name__ == "__main__":
    main()
