"""
Semantic (dense vector) search using FAISS + bge-m3 embeddings.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from src.config import FAISS_INDEX, FAISS_IDS, EMBEDDING_MODEL, DEVICE

logger = logging.getLogger(__name__)


class SemanticSearcher:
    """FAISS-based semantic search with local bge-m3 embeddings."""

    def __init__(
        self,
        index_path: Path | str | None = None,
        ids_path: Path | str | None = None,
    ):
        import faiss

        index_path = Path(index_path) if index_path else FAISS_INDEX
        ids_path = Path(ids_path) if ids_path else FAISS_IDS

        self.index = faiss.read_index(str(index_path))

        with open(ids_path, "r", encoding="utf-8") as f:
            self.chunk_ids = json.load(f)

        # Load embedding model
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)

        logger.info(
            f"Semantic searcher loaded: {len(self.chunk_ids)} chunks, "
            f"model={EMBEDDING_MODEL}, device={DEVICE}"
        )

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query."""
        embedding = self.model.encode(
            [query], normalize_embeddings=True
        )
        return embedding.astype(np.float32)

    def search(self, query: str, top_k: int = 30) -> list[tuple[str, float]]:
        """
        Search FAISS index.
        Returns list of (chunk_id, score) sorted by score descending.
        """
        query_embedding = self.embed_query(query)
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunk_ids):
                results.append((self.chunk_ids[idx], float(score)))

        return results
