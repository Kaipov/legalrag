"""
Step 1: Extract text from PDFs page-by-page.

Strategy:
  - Primary: pdfplumber (fast, reliable for digital/text-based PDFs)
  - Fallback: PaddleOCR-VL-1.5 (SOTA VLM for scanned/complex pages)

Output: index/pages.jsonl — one line per page:
  {"doc_id": "sha256hash", "page_num": 1, "text": "...", "method": "pdfplumber|paddleocr"}
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

import pdfplumber
from tqdm import tqdm

from src.config import DOCUMENTS_DIR, PAGES_JSONL, INDEX_DIR, OCR_MIN_CHARS

logger = logging.getLogger(__name__)

# Lazy-load PaddleOCR to avoid import errors when not installed
_paddle_ocr = None


def _get_paddle_ocr():
    """Lazy-initialize PaddleOCR-VL-1.5 model (loads on first call)."""
    global _paddle_ocr
    if _paddle_ocr is None:
        try:
            from paddleocr import PaddleOCRVL
            _paddle_ocr = PaddleOCRVL()
            logger.info("PaddleOCR-VL-1.5 loaded successfully")
        except ImportError:
            logger.warning(
                "PaddleOCR not installed. Scanned pages will have empty text. "
                "Install with: pip install paddlepaddle-gpu==3.2.1 && "
                "pip install -U 'paddleocr[doc-parser]'"
            )
            return None
    return _paddle_ocr


def _extract_page_pdfplumber(page: pdfplumber.page.Page) -> str:
    """Extract text from a single pdfplumber page."""
    text = page.extract_text() or ""
    return text.strip()


def _extract_page_ocr(pdf_path: Path, page_num: int) -> str:
    """
    Extract text from a page using PaddleOCR-VL-1.5.
    page_num is 0-based here (for pdf2image).
    Returns extracted text or empty string if OCR not available.
    """
    ocr = _get_paddle_ocr()
    if ocr is None:
        return ""

    try:
        from pdf2image import convert_from_path
        # Convert single page to image (1-indexed for pdf2image)
        images = convert_from_path(
            str(pdf_path),
            first_page=page_num + 1,
            last_page=page_num + 1,
            dpi=200,
        )
        if not images:
            return ""

        # Save temp image and run OCR
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
            images[0].save(tmp_path, "PNG")

        try:
            output = ocr.predict(tmp_path)
            # PaddleOCR returns list of results; extract text
            texts = []
            for res in output:
                if hasattr(res, 'text'):
                    texts.append(res.text)
                elif hasattr(res, 'save_to_markdown'):
                    # Try to get markdown representation
                    import io
                    md = res.to_markdown() if hasattr(res, 'to_markdown') else str(res)
                    texts.append(md)
                else:
                    texts.append(str(res))
            return "\n".join(texts).strip()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    except Exception as e:
        logger.warning(f"OCR failed for {pdf_path.name} page {page_num + 1}: {e}")
        return ""


def extract_single_pdf(pdf_path: Path) -> Iterator[dict]:
    """
    Extract text from a single PDF, page by page.
    Yields dict per page: {doc_id, page_num (1-based), text, method}
    """
    doc_id = pdf_path.stem  # SHA hash filename without .pdf

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                page_num = page_idx + 1  # 1-based

                # Try pdfplumber first
                text = _extract_page_pdfplumber(page)
                method = "pdfplumber"

                # Fallback to OCR if text is too short
                if len(text) < OCR_MIN_CHARS:
                    ocr_text = _extract_page_ocr(pdf_path, page_idx)
                    if len(ocr_text) > len(text):
                        text = ocr_text
                        method = "paddleocr"

                yield {
                    "doc_id": doc_id,
                    "page_num": page_num,
                    "text": text,
                    "method": method,
                }
    except Exception as e:
        logger.error(f"Failed to process {pdf_path.name}: {e}")


def extract_all_documents(
    docs_dir: Path | str | None = None,
    output_path: Path | str | None = None,
) -> Path:
    """
    Extract text from all PDFs in docs_dir.
    Saves to output_path (default: index/pages.jsonl).
    Returns the output path.
    """
    docs_dir = Path(docs_dir) if docs_dir else DOCUMENTS_DIR
    output_path = Path(output_path) if output_path else PAGES_JSONL

    output_path.parent.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(docs_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {docs_dir}")
        return output_path

    logger.info(f"Extracting text from {len(pdf_files)} PDFs...")

    total_pages = 0
    ocr_pages = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for pdf_path in tqdm(pdf_files, desc="Extracting PDFs"):
            for page_data in extract_single_pdf(pdf_path):
                f.write(json.dumps(page_data, ensure_ascii=False) + "\n")
                total_pages += 1
                if page_data["method"] == "paddleocr":
                    ocr_pages += 1

    logger.info(
        f"Extracted {total_pages} pages from {len(pdf_files)} PDFs "
        f"({ocr_pages} pages needed OCR). Saved to {output_path}"
    )
    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    extract_all_documents()
