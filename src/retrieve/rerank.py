"""
Cross-encoder reranking using bge-reranker-v2-m3 on GPU.

Takes (query, chunk_text) pairs and scores their relevance.
Much more accurate than bi-encoder (FAISS) or BM25 because it
sees both texts simultaneously through cross-attention.
"""
from __future__ import annotations

import logging

from src.config import (
    DEVICE,
    RERANKER_BATCH_SIZE,
    RERANKER_MAX_LENGTH,
    RERANKER_MODEL,
    RERANKER_USE_FP16,
)

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Cross-encoder reranker using bge-reranker-v2-m3."""

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int | None = None,
        max_length: int | None = None,
        use_fp16: bool | None = None,
    ):
        from sentence_transformers import CrossEncoder

        model_name = model_name or RERANKER_MODEL
        self.batch_size = batch_size if batch_size is not None else RERANKER_BATCH_SIZE
        self.max_length = max_length if max_length is not None else RERANKER_MAX_LENGTH
        self.use_fp16 = RERANKER_USE_FP16 if use_fp16 is None else use_fp16

        model_kwargs: dict = {}
        dtype_name = "default"
        if DEVICE == "cuda" and self.use_fp16:
            import torch

            model_kwargs["torch_dtype"] = torch.float16
            dtype_name = "float16"
        elif self.use_fp16:
            logger.info("Reranker fp16 requested but device=%s; using default dtype", DEVICE)

        logger.info(
            "Loading reranker %s on %s (max_length=%s, batch_size=%s, dtype=%s)...",
            model_name,
            DEVICE,
            self.max_length,
            self.batch_size,
            dtype_name,
        )

        self.model = CrossEncoder(
            model_name,
            device=DEVICE,
            max_length=self.max_length,
            model_kwargs=model_kwargs,
        )
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

        pairs = [(query, chunk["text"]) for chunk in chunks]
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        scored = list(zip(chunks, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(chunk, float(score)) for chunk, score in scored[:top_k]]
