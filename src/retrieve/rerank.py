"""
Cross-encoder reranking using bge-reranker-v2-m3 on GPU.

Takes (query, chunk_text) pairs and scores their relevance.
Much more accurate than bi-encoder (FAISS) or BM25 because it
sees both texts simultaneously through cross-attention.
"""
from __future__ import annotations

import logging

from src.config import RERANKER_MODEL, DEVICE

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Cross-encoder reranker using bge-reranker-v2-m3."""

    def __init__(self, model_name: str | None = None):
        from sentence_transformers import CrossEncoder

        model_name = model_name or RERANKER_MODEL
        logger.info(f"Loading reranker {model_name} on {DEVICE}...")

        self.model = CrossEncoder(model_name, device=DEVICE)
        logger.info("Reranker loaded successfully")

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int = 10,
    ) -> list[tuple[dict, float]]:
        """
        Rerank chunks by relevance to query.

        Args:
            query: Question text
            chunks: List of chunk dicts (must have 'text' key)
            top_k: Number of top results to return

        Returns:
            List of (chunk_dict, reranker_score) sorted by score descending.
        """
        if not chunks:
            return []

        # Build pairs for cross-encoder
        pairs = [(query, chunk["text"]) for chunk in chunks]

        # Score all pairs
        scores = self.model.predict(pairs)

        # Combine with chunks and sort
        scored = list(zip(chunks, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        # Return top-k
        return [(chunk, float(score)) for chunk, score in scored[:top_k]]
