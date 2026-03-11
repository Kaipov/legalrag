"""
Step 3: Build BM25 and FAISS indices from chunks.

BM25: keyword-based retrieval (rank_bm25 library)
FAISS: dense vector retrieval (bge-m3 embeddings on GPU)

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
    CHUNKS_JSONL, BM25_INDEX, FAISS_INDEX, FAISS_IDS,
    EMBEDDING_MODEL, DEVICE, INDEX_DIR,
)

logger = logging.getLogger(__name__)


# --- BM25 Index ---

def _tokenize_for_bm25(text: str) -> list[str]:
    """Simple tokenization for BM25: lowercase, split on non-alphanumeric."""
    import re
    # Lowercase and split on non-alphanumeric characters
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    # Remove very short tokens and common stopwords
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "of", "in", "to", "for",
        "with", "on", "at", "by", "from", "as", "into", "through", "during",
        "and", "or", "but", "not", "no", "nor", "if", "then", "than",
        "that", "this", "these", "those", "it", "its",
    }
    return [t for t in tokens if len(t) > 1 and t not in stopwords]


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

    # Load chunks
    chunk_ids = []
    tokenized_corpus = []

    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Tokenizing for BM25"):
            chunk = json.loads(line)
            chunk_ids.append(chunk["chunk_id"])
            tokenized_corpus.append(_tokenize_for_bm25(chunk["text"]))

    logger.info(f"Building BM25 index over {len(chunk_ids)} chunks...")
    bm25 = BM25Okapi(tokenized_corpus)

    # Save BM25 model + chunk_ids
    with open(output_path, "wb") as f:
        pickle.dump({"bm25": bm25, "chunk_ids": chunk_ids}, f)

    logger.info(f"BM25 index saved to {output_path}")
    return output_path


# --- FAISS Index ---

def _load_embedding_model():
    """Load the bge-m3 embedding model on GPU."""
    from sentence_transformers import SentenceTransformer
    logger.info(f"Loading embedding model {EMBEDDING_MODEL} on {DEVICE}...")
    model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
    logger.info(f"Embedding model loaded. Dimension: {model.get_sentence_embedding_dimension()}")
    return model


def build_faiss_index(
    chunks_path: Path | str | None = None,
    index_path: Path | str | None = None,
    ids_path: Path | str | None = None,
    batch_size: int = 64,
) -> Path:
    """
    Build FAISS index from chunks.jsonl using bge-m3 embeddings.
    Saves FAISS index and chunk_id mapping.
    """
    import faiss

    chunks_path = Path(chunks_path) if chunks_path else CHUNKS_JSONL
    index_path = Path(index_path) if index_path else FAISS_INDEX
    ids_path = Path(ids_path) if ids_path else FAISS_IDS

    index_path.parent.mkdir(parents=True, exist_ok=True)

    # Load chunks
    chunk_ids = []
    texts = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            chunk_ids.append(chunk["chunk_id"])
            texts.append(chunk["text"])

    logger.info(f"Embedding {len(texts)} chunks with {EMBEDDING_MODEL}...")

    # Load model and encode
    model = _load_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # Normalize for cosine similarity via inner product
    )

    # Build FAISS index (inner product = cosine on normalized vectors)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    # Save
    faiss.write_index(index, str(index_path))
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(chunk_ids, f)

    logger.info(
        f"FAISS index saved to {index_path} ({len(chunk_ids)} vectors, dim={dim})"
    )

    # Free GPU memory
    del model
    if DEVICE == "cuda":
        import torch
        torch.cuda.empty_cache()

    return index_path


def build_all_indices(
    chunks_path: Path | str | None = None,
) -> None:
    """Build both BM25 and FAISS indices."""
    chunks_path = chunks_path or CHUNKS_JSONL

    build_bm25_index(chunks_path)
    build_faiss_index(chunks_path)

    logger.info("All indices built successfully!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_all_indices()
