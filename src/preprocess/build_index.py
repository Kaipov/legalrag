"""
Step 3: Build BM25 and FAISS indices from chunks.

BM25: keyword-based retrieval (rank_bm25 library)
FAISS: dense vector retrieval (bge-m3 embeddings)

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
    DEVICE,
    EMBEDDING_MODEL,
    FAISS_IDS,
    FAISS_INDEX,
    MAX_CHUNK_TOKENS,
)

logger = logging.getLogger(__name__)


# --- BM25 Index ---

def _tokenize_for_bm25(text: str) -> list[str]:
    """Simple tokenization for BM25: lowercase, split on non-alphanumeric."""
    import re

    tokens = re.findall(r"[a-z0-9]+", text.lower())
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "of", "in", "to", "for",
        "with", "on", "at", "by", "from", "as", "into", "through", "during",
        "and", "or", "but", "not", "no", "nor", "if", "then", "than",
        "that", "this", "these", "those", "it", "its",
    }
    return [token for token in tokens if len(token) > 1 and token not in stopwords]


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
            tokenized_corpus.append(_tokenize_for_bm25(chunk["text"]))

    logger.info(f"Building BM25 index over {len(chunk_ids)} chunks...")
    bm25 = BM25Okapi(tokenized_corpus)

    with open(output_path, "wb") as handle:
        pickle.dump({"bm25": bm25, "chunk_ids": chunk_ids}, handle)

    logger.info(f"BM25 index saved to {output_path}")
    return output_path


# --- FAISS Index ---

def _load_embedding_model():
    """Load the embedding model with a sane truncation limit for preprocessing."""
    from sentence_transformers import SentenceTransformer

    logger.info(f"Loading embedding model {EMBEDDING_MODEL} on {DEVICE}...")
    model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)

    max_seq_length = getattr(model, "max_seq_length", None)
    if isinstance(max_seq_length, int) and max_seq_length > 0:
        model.max_seq_length = min(max_seq_length, MAX_CHUNK_TOKENS)
        logger.info(f"Embedding max_seq_length set to {model.max_seq_length}")

    logger.info(
        f"Embedding model loaded. Dimension: {model.get_sentence_embedding_dimension()}"
    )
    return model


def _count_chunks(chunks_path: Path) -> int:
    with open(chunks_path, "r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _is_memory_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "not enough memory" in message


def _clear_torch_cache() -> None:
    if DEVICE != "cuda":
        return

    try:
        import torch
    except ImportError:  # pragma: no cover - depends on local environment
        return

    torch.cuda.empty_cache()


def _encode_with_backoff(model, texts: list[str], batch_size: int) -> tuple[np.ndarray, int]:
    """Retry embedding with smaller batch sizes when the runtime runs out of memory."""
    effective_batch_size = max(1, min(batch_size, len(texts)))

    while True:
        try:
            embeddings = model.encode(
                texts,
                batch_size=effective_batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            return embeddings.astype(np.float32), effective_batch_size
        except RuntimeError as exc:
            if not _is_memory_error(exc) or effective_batch_size == 1:
                raise

            new_batch_size = max(1, effective_batch_size // 2)
            logger.warning(
                "Embedding batch failed with batch_size=%s; retrying with batch_size=%s",
                effective_batch_size,
                new_batch_size,
            )
            effective_batch_size = new_batch_size
            _clear_torch_cache()


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
    batch_size = batch_size or (8 if DEVICE == "cpu" else 32)

    index_path.parent.mkdir(parents=True, exist_ok=True)

    total_chunks = _count_chunks(chunks_path)
    if total_chunks == 0:
        raise ValueError(f"No chunks found in {chunks_path}")

    logger.info(
        f"Embedding {total_chunks} chunks with {EMBEDDING_MODEL} using batch_size={batch_size}..."
    )

    model = _load_embedding_model()
    chunk_ids: list[str] = []
    current_ids: list[str] = []
    current_texts: list[str] = []
    current_batch_size = max(1, batch_size)
    index = None
    dim = 0

    try:
        with open(chunks_path, "r", encoding="utf-8") as handle, tqdm(
            total=total_chunks,
            desc="Embedding chunks",
        ) as progress:
            for line in handle:
                chunk = json.loads(line)
                current_ids.append(chunk["chunk_id"])
                current_texts.append(chunk["text"])

                if len(current_texts) < current_batch_size:
                    continue

                batch_embeddings, current_batch_size = _encode_with_backoff(
                    model,
                    current_texts,
                    current_batch_size,
                )
                if index is None:
                    dim = batch_embeddings.shape[1]
                    index = faiss.IndexFlatIP(dim)
                index.add(batch_embeddings)
                chunk_ids.extend(current_ids)
                progress.update(len(current_ids))
                current_ids = []
                current_texts = []

            if current_texts:
                batch_embeddings, current_batch_size = _encode_with_backoff(
                    model,
                    current_texts,
                    current_batch_size,
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
            f"FAISS index saved to {index_path} ({len(chunk_ids)} vectors, dim={dim})"
        )
        return index_path
    finally:
        del model
        _clear_torch_cache()


def build_all_indices(chunks_path: Path | str | None = None) -> None:
    """Build both BM25 and FAISS indices."""
    chunks_path = chunks_path or CHUNKS_JSONL

    build_bm25_index(chunks_path)
    build_faiss_index(chunks_path)

    logger.info("All indices built successfully!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_all_indices()

