"""
Offline preprocessing: extract text from PDFs, chunk, build indices.

Usage:
    python -m scripts.preprocess              # full pipeline
    python -m scripts.preprocess --extract     # only extract text
    python -m scripts.preprocess --chunk       # only chunk (requires extract first)
    python -m scripts.preprocess --index       # only build indices (requires chunk first)
"""
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ensure_dirs, DOCUMENTS_DIR, PAGES_JSONL, CHUNKS_JSONL
from src.preprocess.extract import extract_all_documents
from src.preprocess.chunk import chunk_all_documents
from src.preprocess.build_index import build_all_indices


def main():
    parser = argparse.ArgumentParser(description="ARLC document preprocessing pipeline")
    parser.add_argument("--extract", action="store_true", help="Only extract text from PDFs")
    parser.add_argument("--chunk", action="store_true", help="Only chunk extracted pages")
    parser.add_argument("--index", action="store_true", help="Only build search indices")
    parser.add_argument("--docs-dir", type=str, help="Path to documents directory")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    ensure_dirs()

    docs_dir = Path(args.docs_dir) if args.docs_dir else DOCUMENTS_DIR

    # If no specific step requested, run all
    run_all = not (args.extract or args.chunk or args.index)

    if run_all or args.extract:
        print(f"\n{'='*60}")
        print(f"STEP 1: Extracting text from PDFs in {docs_dir}")
        print(f"{'='*60}")
        extract_all_documents(docs_dir=docs_dir)

    if run_all or args.chunk:
        if not PAGES_JSONL.exists():
            print(f"ERROR: {PAGES_JSONL} not found. Run --extract first.")
            sys.exit(1)
        print(f"\n{'='*60}")
        print(f"STEP 2: Chunking documents")
        print(f"{'='*60}")
        chunk_all_documents()

    if run_all or args.index:
        if not CHUNKS_JSONL.exists():
            print(f"ERROR: {CHUNKS_JSONL} not found. Run --chunk first.")
            sys.exit(1)
        print(f"\n{'='*60}")
        print(f"STEP 3: Building search indices (BM25 + FAISS)")
        print(f"{'='*60}")
        build_all_indices()

    print(f"\n{'='*60}")
    print("Preprocessing complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
