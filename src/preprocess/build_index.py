"""
Step 3: Build BM25 and FAISS indices from chunks.

BM25: keyword-based retrieval (rank_bm25 library)
FAISS: dense vector retrieval (Gemini embeddings via API)

Both indices are saved to index/ directory for online query use.
"""
from __future__ import annotations

import json
import logging
import pickle
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
)
from src.embeddings import GeminiApiError, get_embedding_client
from src.retrieve.lexical import build_bm25_document_tokens

logger = logging.getLogger(__name__)


# --- BM25 Index ---


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

    chunk_ids = []
    tokenized_corpus = []

    with open(chunks_path, "r", encoding="utf-8") as handle:
        for line in tqdm(handle, desc="Tokenizing for BM25"):
            chunk = json.loads(line)
            chunk_ids.append(chunk["chunk_id"])
            tokenized_corpus.append(build_bm25_document_tokens(chunk))

    logger.info("Building BM25 index over %s chunks...", len(chunk_ids))
    bm25 = BM25Okapi(tokenized_corpus)

    with open(output_path, "wb") as handle:
        pickle.dump({"bm25": bm25, "chunk_ids": chunk_ids}, handle)

    logger.info("BM25 index saved to %s", output_path)
    return output_path


# --- FAISS Index ---

def _count_chunks(chunks_path: Path) -> int:
    with open(chunks_path, "r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _is_batch_size_error(exc: Exception) -> bool:
    if not isinstance(exc, GeminiApiError):
        return False

    message = f"{exc} {exc.response_text}".lower()
    if exc.status_code == 413:
        return True
    return "payload" in message or "too large" in message or "request size" in message


def _build_embedding_title(chunk: dict) -> str | None:
    doc_title = str(chunk.get("doc_title") or "").strip()
    section_path = str(chunk.get("section_path") or "").strip()
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
    import faiss

    chunks_path = Path(chunks_path) if chunks_path else CHUNKS_JSONL
    index_path = Path(index_path) if index_path else FAISS_INDEX
    ids_path = Path(ids_path) if ids_path else FAISS_IDS
    batch_size = batch_size or EMBEDDING_BATCH_SIZE

    index_path.parent.mkdir(parents=True, exist_ok=True)

    total_chunks = _count_chunks(chunks_path)
    if total_chunks == 0:
        raise ValueError(f"No chunks found in {chunks_path}")

    client = get_embedding_client()
    logger.info(
        "Embedding %s chunks with %s using batch_size=%s...",
        total_chunks,
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

    with open(chunks_path, "r", encoding="utf-8") as handle, tqdm(
        total=total_chunks,
        desc="Embedding chunks",
    ) as progress:
        for line in handle:
            chunk = json.loads(line)
            current_ids.append(chunk["chunk_id"])
            current_texts.append(chunk["text"])
            current_titles.append(_build_embedding_title(chunk))

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
        raise ValueError(f"Could not build FAISS index from {chunks_path}")

    faiss.write_index(index, str(index_path))
    with open(ids_path, "w", encoding="utf-8") as handle:
        json.dump(chunk_ids, handle)

    logger.info(
        "FAISS index saved to %s (%s vectors, dim=%s)",
        index_path,
        len(chunk_ids),
        dim,
    )
    return index_path


def build_all_indices(chunks_path: Path | str | None = None) -> None:
    """Build both BM25 and FAISS indices."""
    chunks_path = chunks_path or CHUNKS_JSONL

    build_bm25_index(chunks_path)
    build_faiss_index(chunks_path)

    logger.info("All indices built successfully!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_all_indices()
