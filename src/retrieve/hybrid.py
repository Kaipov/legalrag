"""
Hybrid retrieval: BM25 + Semantic search -> RRF fusion -> optional cross-encoder rerank.

This is the main retrieval orchestrator.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from src.config import (
    BM25_TOP_K,
    CHUNKS_JSONL,
    ENABLE_RERANKER,
    RERANK_CANDIDATES,
    RERANK_TOP_K,
    RRF_K,
    SEMANTIC_TOP_K,
)
from src.retrieve.bm25 import BM25Searcher
from src.retrieve.rerank import CrossEncoderReranker
from src.retrieve.semantic import SemanticSearcher

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Full retrieval pipeline:
    1. BM25 search (keyword, top-k)
    2. Semantic search (dense, top-k)
    3. RRF fusion
    4. Optional cross-encoder rerank
    """

    def __init__(
        self,
        chunks_path: Path | str | None = None,
        enable_reranker: bool | None = None,
    ):
        chunks_path = Path(chunks_path) if chunks_path else CHUNKS_JSONL
        self.enable_reranker = ENABLE_RERANKER if enable_reranker is None else enable_reranker

        self.chunks_by_id: dict[str, dict] = {}
        with open(chunks_path, "r", encoding="utf-8") as handle:
            for line in handle:
                chunk = json.loads(line)
                self.chunks_by_id[chunk["chunk_id"]] = chunk

        logger.info(f"Loaded {len(self.chunks_by_id)} chunks")

        self.bm25 = BM25Searcher()
        self.semantic = SemanticSearcher()
        self.reranker = CrossEncoderReranker() if self.enable_reranker else None

        if self.enable_reranker:
            logger.info("HybridRetriever initialized with reranker enabled")
        else:
            logger.info("HybridRetriever initialized with reranker disabled")

    def retrieve(
        self,
        query: str,
        rerank_top_k: int | None = None,
    ) -> list[tuple[dict, float]]:
        """
        Full retrieval pipeline for a query.

        Returns list of (chunk_dict, score) sorted by relevance.
        Score is the cross-encoder logit when reranker is enabled, otherwise the RRF score.
        """
        rerank_top_k = rerank_top_k or RERANK_TOP_K
        candidates = self._get_rrf_candidates(query)

        if not candidates:
            logger.warning(f"No candidates found for query: {query[:80]}...")
            return []

        if not self.enable_reranker or self.reranker is None:
            return candidates[:rerank_top_k]

        reranked = self.reranker.rerank(
            query,
            [chunk for chunk, _score in candidates],
            top_k=rerank_top_k,
        )
        return reranked

    def retrieve_without_rerank(
        self,
        query: str,
        top_k: int = 15,
    ) -> list[tuple[dict, float]]:
        """Retrieval without reranking (faster, for testing or fallback)."""
        return self._get_rrf_candidates(query)[:top_k]

    def _get_rrf_candidates(self, query: str) -> list[tuple[dict, float]]:
        """Return top fusion candidates as (chunk_dict, rrf_score)."""
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        semantic_results = self.semantic.search(query, top_k=SEMANTIC_TOP_K)
        rrf_ranked = self._rrf_fusion(bm25_results, semantic_results)

        candidates = []
        for chunk_id, rrf_score in rrf_ranked[:RERANK_CANDIDATES]:
            chunk = self.chunks_by_id.get(chunk_id)
            if chunk is not None:
                candidates.append((chunk, float(rrf_score)))

        return candidates

    def _rrf_fusion(
        self,
        bm25_results: list[tuple[str, float]],
        semantic_results: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        """
        Reciprocal Rank Fusion to merge BM25 and semantic results.
        RRF_score(doc) = sum(1 / (k + rank_in_list))
        """
        scores: dict[str, float] = {}

        for rank, (chunk_id, _) in enumerate(bm25_results):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)

        for rank, (chunk_id, _) in enumerate(semantic_results):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)

        return sorted(scores.items(), key=lambda item: item[1], reverse=True)
