"""
Semantic (dense vector) search using FAISS + Gemini embeddings.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from src.config import EMBEDDING_MODEL, FAISS_IDS, FAISS_INDEX
from src.embeddings import get_embedding_client

logger = logging.getLogger(__name__)


class SemanticSearcher:
    """FAISS-based semantic search with Gemini embedding vectors."""

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

        self.client = get_embedding_client()

        logger.info(
            "Semantic searcher loaded: %s chunks, model=%s",
            len(self.chunk_ids),
            EMBEDDING_MODEL,
        )

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query."""
        return self.client.embed_query(query).astype(np.float32)

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