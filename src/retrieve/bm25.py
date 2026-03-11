"""
BM25 keyword search over pre-built index.
"""
from __future__ import annotations

import pickle
import logging
from pathlib import Path

from src.config import BM25_INDEX

logger = logging.getLogger(__name__)


class BM25Searcher:
    """Wrapper around rank_bm25 for keyword-based retrieval."""

    def __init__(self, index_path: Path | str | None = None):
        index_path = Path(index_path) if index_path else BM25_INDEX

        with open(index_path, "rb") as f:
            data = pickle.load(f)

        self.bm25 = data["bm25"]
        self.chunk_ids = data["chunk_ids"]
        logger.info(f"BM25 index loaded: {len(self.chunk_ids)} chunks")

    def search(self, query: str, top_k: int = 30) -> list[tuple[str, float]]:
        """
        Search BM25 index.
        Returns list of (chunk_id, score) sorted by score descending.
        """
        import re
        # Tokenize query same as index
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "shall", "can", "of", "in", "to", "for",
            "with", "on", "at", "by", "from", "as", "into", "through", "during",
            "and", "or", "but", "not", "no", "nor", "if", "then", "than",
            "that", "this", "these", "those", "it", "its",
        }
        tokens = [t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 1 and t not in stopwords]

        scores = self.bm25.get_scores(tokens)

        # Get top-k indices
        top_indices = scores.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0:
                results.append((self.chunk_ids[idx], score))

        return results
