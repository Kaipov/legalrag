"""
Step 3: Build BM25 and FAISS indices from chunks and pages.

BM25: keyword-based retrieval (rank_bm25 library)
FAISS: dense vector retrieval (Gemini embeddings via API)

All indices are saved to index/ directory for online query use.
"""
from __future__ import annotations

import json
import logging
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.config import (
    BM25_INDEX,
    CHUNKS_JSONL,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    FAISS_IDS,
    FAISS_INDEX,
    PAGE_BM25_INDEX,
    PAGE_FAISS_IDS,
    PAGE_FAISS_INDEX,
    PAGES_JSONL,
)
from src.embeddings import GeminiApiError, get_embedding_client
from src.preprocess.chunk import _detect_doc_title
from src.retrieve.lexical import build_bm25_document_tokens

logger = logging.getLogger(__name__)


# --- Shared helpers ---


def _read_jsonl_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
    return records


# --- Chunk BM25 Index ---


def build_bm25_index(
    chunks_path: Path | str | None = None,
    output_path: Path | str | None = None,
) -> Path:
    """
    Build BM25 index from chunks.jsonl.
    Saves pickle with (bm25_model, chunk_ids) to output_path.
    """
    from rank_bm25 import BM25Okapi

    chunks_path = Path(chunks_path) if chunks_path else CHUNKS_JSONL
    output_path = Path(output_path) if output_path else BM25_INDEX

    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = _read_jsonl_records(chunks_path)
    chunk_ids = [str(record["chunk_id"]) for record in records]
    tokenized_corpus = [build_bm25_document_tokens(record) for record in tqdm(records, desc="Tokenizing chunks for BM25")]

    logger.info("Building BM25 index over %s chunks...", len(chunk_ids))
    bm25 = BM25Okapi(tokenized_corpus)

    with open(output_path, "wb") as handle:
        pickle.dump({"bm25": bm25, "chunk_ids": chunk_ids}, handle)

    logger.info("BM25 index saved to %s", output_path)
    return output_path


# --- Page indices ---


def _build_page_section_path(page_num: int, last_page: int) -> str:
    labels: list[str] = []
    if page_num == 1:
        labels.extend(["Title Page", "First Page"])
    if page_num == last_page:
        labels.extend(["Last Page", "Final Page"])
    labels.append(f"Page {page_num}")

    deduped_labels: list[str] = []
    for label in labels:
        if label not in deduped_labels:
            deduped_labels.append(label)
    return " > ".join(deduped_labels)



def _load_page_records(pages_path: Path | str | None = None) -> list[dict]:
    pages_path = Path(pages_path) if pages_path else PAGES_JSONL
    grouped_rows: dict[str, list[dict]] = defaultdict(list)

    for record in _read_jsonl_records(pages_path):
        doc_id = str(record.get("doc_id") or "").strip()
        page_num = int(record.get("page_num") or 0)
        if not doc_id or page_num <= 0:
            continue
        grouped_rows[doc_id].append(record)

    page_records: list[dict] = []
    for doc_id in sorted(grouped_rows.keys()):
        rows = sorted(grouped_rows[doc_id], key=lambda row: int(row.get("page_num") or 0))
        title_sample = "\n\n".join(str(row.get("text") or "") for row in rows[:2])
        doc_title = _detect_doc_title(title_sample)
        last_page = max(int(row.get("page_num") or 0) for row in rows)

        for row in rows:
            page_num = int(row.get("page_num") or 0)
            if page_num <= 0:
                continue
            page_records.append(
                {
                    "page_id": f"{doc_id}:{page_num}",
                    "doc_id": doc_id,
                    "page_num": page_num,
                    "doc_title": doc_title,
                    "section_path": _build_page_section_path(page_num, last_page),
                    "text": str(row.get("text") or ""),
                }
            )

    return page_records



def build_page_bm25_index(
    pages_path: Path | str | None = None,
    output_path: Path | str | None = None,
) -> Path:
    """Build BM25 index from pages.jsonl."""
    from rank_bm25 import BM25Okapi

    output_path = Path(output_path) if output_path else PAGE_BM25_INDEX
    output_path.parent.mkdir(parents=True, exist_ok=True)

    page_records = _load_page_records(pages_path)
    page_ids = [str(record["page_id"]) for record in page_records]
    tokenized_corpus = [build_bm25_document_tokens(record) for record in tqdm(page_records, desc="Tokenizing pages for BM25")]

    logger.info("Building page BM25 index over %s pages...", len(page_ids))
    bm25 = BM25Okapi(tokenized_corpus)

    with open(output_path, "wb") as handle:
        pickle.dump({"bm25": bm25, "page_ids": page_ids}, handle)

    logger.info("Page BM25 index saved to %s", output_path)
    return output_path


# --- Shared FAISS Index helpers ---


def _is_batch_size_error(exc: Exception) -> bool:
    if not isinstance(exc, GeminiApiError):
        return False

    message = f"{exc} {exc.response_text}".lower()
    if exc.status_code == 413:
        return True
    return "payload" in message or "too large" in message or "request size" in message



def _build_embedding_title(record: dict) -> str | None:
    doc_title = str(record.get("doc_title") or "").strip()
    section_path = str(record.get("section_path") or "").strip()
    if doc_title and section_path and section_path != doc_title:
        return f"{doc_title} - {section_path}"
    return section_path or doc_title or None



def _encode_with_backoff(
    client,
    texts: list[str],
    batch_size: int,
    *,
    titles: list[str | None] | None = None,
) -> tuple[np.ndarray, int]:
    """Retry with smaller API batches if a request payload is too large."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32), max(1, batch_size)

    if titles is not None and len(titles) != len(texts):
        raise ValueError("titles length must match texts length")

    effective_batch_size = max(1, min(batch_size, len(texts)))
    outputs: list[np.ndarray] = []
    start = 0

    while start < len(texts):
        current_batch_size = min(effective_batch_size, len(texts) - start)
        while True:
            batch_texts = texts[start:start + current_batch_size]
            batch_titles = titles[start:start + current_batch_size] if titles is not None else None
            try:
                outputs.append(client.embed_documents(batch_texts, titles=batch_titles))
                start += current_batch_size
                break
            except Exception as exc:
                if not _is_batch_size_error(exc) or current_batch_size == 1:
                    raise

                new_batch_size = max(1, current_batch_size // 2)
                logger.warning(
                    "Embedding batch failed with batch_size=%s; retrying with batch_size=%s",
                    current_batch_size,
                    new_batch_size,
                )
                current_batch_size = new_batch_size
                effective_batch_size = min(effective_batch_size, new_batch_size)

    return np.vstack(outputs).astype(np.float32), effective_batch_size



def _build_faiss_index_from_records(
    records: list[dict],
    *,
    record_id_field: str,
    index_path: Path,
    ids_path: Path,
    batch_size: int,
    desc: str,
) -> Path:
    import faiss

    if not records:
        raise ValueError(f"No records found for {desc}")

    client = get_embedding_client()
    logger.info(
        "Embedding %s %s with %s using batch_size=%s...",
        len(records),
        desc,
        EMBEDDING_MODEL,
        batch_size,
    )

    chunk_ids: list[str] = []
    current_ids: list[str] = []
    current_texts: list[str] = []
    current_titles: list[str | None] = []
    current_batch_size = max(1, batch_size)
    index = None
    dim = 0

    with tqdm(total=len(records), desc=f"Embedding {desc}") as progress:
        for record in records:
            current_ids.append(str(record[record_id_field]))
            current_texts.append(str(record.get("text") or ""))
            current_titles.append(_build_embedding_title(record))

            if len(current_texts) < current_batch_size:
                continue

            batch_embeddings, current_batch_size = _encode_with_backoff(
                client,
                current_texts,
                current_batch_size,
                titles=current_titles,
            )
            if index is None:
                dim = batch_embeddings.shape[1]
                index = faiss.IndexFlatIP(dim)
            index.add(batch_embeddings)
            chunk_ids.extend(current_ids)
            progress.update(len(current_ids))
            current_ids = []
            current_texts = []
            current_titles = []

        if current_texts:
            batch_embeddings, current_batch_size = _encode_with_backoff(
                client,
                current_texts,
                current_batch_size,
                titles=current_titles,
            )
            if index is None:
                dim = batch_embeddings.shape[1]
                index = faiss.IndexFlatIP(dim)
            index.add(batch_embeddings)
            chunk_ids.extend(current_ids)
            progress.update(len(current_ids))

    if index is None:
        raise ValueError(f"Could not build FAISS index for {desc}")

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    with open(ids_path, "w", encoding="utf-8") as handle:
        json.dump(chunk_ids, handle)

    logger.info(
        "%s FAISS index saved to %s (%s vectors, dim=%s)",
        desc.capitalize(),
        index_path,
        len(chunk_ids),
        dim,
    )
    return index_path



def build_faiss_index(
    chunks_path: Path | str | None = None,
    index_path: Path | str | None = None,
    ids_path: Path | str | None = None,
    batch_size: int | None = None,
) -> Path:
    """
    Build FAISS index from chunks.jsonl using dense embeddings.
    Saves FAISS index and chunk_id mapping.
    """
    chunks_path = Path(chunks_path) if chunks_path else CHUNKS_JSONL
    index_path = Path(index_path) if index_path else FAISS_INDEX
    ids_path = Path(ids_path) if ids_path else FAISS_IDS
    batch_size = batch_size or EMBEDDING_BATCH_SIZE

    records = _read_jsonl_records(chunks_path)
    return _build_faiss_index_from_records(
        records,
        record_id_field="chunk_id",
        index_path=index_path,
        ids_path=ids_path,
        batch_size=batch_size,
        desc="chunks",
    )



def build_page_faiss_index(
    pages_path: Path | str | None = None,
    index_path: Path | str | None = None,
    ids_path: Path | str | None = None,
    batch_size: int | None = None,
) -> Path:
    """Build FAISS index from pages.jsonl using dense embeddings."""
    index_path = Path(index_path) if index_path else PAGE_FAISS_INDEX
    ids_path = Path(ids_path) if ids_path else PAGE_FAISS_IDS
    batch_size = batch_size or EMBEDDING_BATCH_SIZE

    records = _load_page_records(pages_path)
    return _build_faiss_index_from_records(
        records,
        record_id_field="page_id",
        index_path=index_path,
        ids_path=ids_path,
        batch_size=batch_size,
        desc="pages",
    )



def build_all_indices(
    chunks_path: Path | str | None = None,
    pages_path: Path | str | None = None,
) -> None:
    """Build chunk and page BM25 + FAISS indices."""
    chunks_path = chunks_path or CHUNKS_JSONL
    pages_path = pages_path or PAGES_JSONL

    build_bm25_index(chunks_path)
    build_faiss_index(chunks_path)
    build_page_bm25_index(pages_path)
    build_page_faiss_index(pages_path)

    logger.info("All chunk and page indices built successfully!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_all_indices()
