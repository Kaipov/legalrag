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
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter_kit"))

from src.config import (
    ensure_dirs, DOCUMENTS_DIR, DATA_DIR,
    EVAL_API_KEY, PROJECT_ROOT,
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
        # Try public dataset
        alt_path = DATA_DIR / "public_dataset.json"
        if alt_path.exists():
            path = alt_path
        else:
            raise FileNotFoundError(f"No questions file found at {path} or {alt_path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Directories/files to skip when archiving
_EXCLUDE_DIRS = {"__pycache__", "data", "index", "storage", ".venv", "venv", "env", ".git", "node_modules"}
_EXCLUDE_FILES = {".env", "submission.json", "questions.json", "code_archive.zip"}


def create_code_archive(archive_path: Path) -> Path:
    """Create code archive for submission."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_resolved = archive_path.resolve()

    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as zf:
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
            # Skip large files
            if file_path.stat().st_size > 10_000_000:  # 10MB
                continue
            zf.write(file_path, file_path.relative_to(PROJECT_ROOT))

    print(f"Code archive created: {archive_path} ({archive_path.stat().st_size / 1024:.0f} KB)")
    return archive_path


def main():
    parser = argparse.ArgumentParser(description="ARLC RAG pipeline — answer and submit")
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

    # --- Download ---
    client = None
    if not args.no_download and not args.dry_run:
        if not EVAL_API_KEY:
            print("WARNING: EVAL_API_KEY not set. Skipping download.")
        else:
            client = EvaluationClient.from_env()
            questions = download_data(client)

    # --- Load questions ---
    if args.questions:
        questions = load_questions(Path(args.questions))
    else:
        try:
            questions = load_questions()
        except FileNotFoundError:
            print("ERROR: No questions file found. Run with --no-download=false or provide --questions path.")
            sys.exit(1)

    if args.dry_run:
        questions = questions[:5]
        print(f"\n[DRY RUN] Answering {len(questions)} questions only")

    print(f"\nLoaded {len(questions)} questions")

    # --- Initialize pipeline ---
    print("\nInitializing RAG pipeline...")
    pipeline = RAGPipeline()

    # --- Answer all questions ---
    print(f"\nAnswering {len(questions)} questions...\n")
    builder = SubmissionBuilder(
        architecture_summary=(
            "Hybrid RAG: pdfplumber+PaddleOCR extraction, structure-aware chunking, "
            "BM25+bge-m3 hybrid search with RRF fusion, bge-reranker-v2-m3 cross-encoder, "
            "gpt-4o streaming generation with type-specific prompts, 3-tier null detection."
        ),
    )

    for i, question_item in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {question_item['id'][:12]}... ({question_item.get('answer_type', '?')})")

        try:
            answer = pipeline.answer_question(question_item)
            builder.add_answer(answer)
        except Exception as e:
            logging.error(f"Error on question {question_item['id']}: {e}")
            # Add a fallback answer to avoid missing questions
            from arlc import SubmissionAnswer, Telemetry, TimingMetrics, UsageMetrics
            builder.add_answer(SubmissionAnswer(
                question_id=question_item["id"],
                answer=None,
                telemetry=Telemetry(
                    timing=TimingMetrics(ttft_ms=0, tpot_ms=0, total_time_ms=0),
                    retrieval=[],
                    usage=UsageMetrics(input_tokens=0, output_tokens=0),
                    model_name="gpt-4o",
                ),
            ))

    # --- Save ---
    submission_path = PROJECT_ROOT / "submission.json"
    builder.save(str(submission_path))
    print(f"\nSubmission saved to {submission_path}")

    # --- Submit ---
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
        except Exception as e:
            print(f"Submission failed: {e}")
    else:
        print("\nSkipping submission (--no-submit or --dry-run)")


if __name__ == "__main__":
    main()
