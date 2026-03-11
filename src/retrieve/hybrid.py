"""
Hybrid retrieval: BM25 + Semantic search → RRF fusion → Cross-encoder rerank.

This is the main retrieval orchestrator.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from src.config import (
    BM25_TOP_K, SEMANTIC_TOP_K, RRF_K,
    RERANK_CANDIDATES, RERANK_TOP_K, CHUNKS_JSONL,
)
from src.retrieve.bm25 import BM25Searcher
from src.retrieve.semantic import SemanticSearcher
from src.retrieve.rerank import CrossEncoderReranker

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Full retrieval pipeline:
    1. BM25 search (keyword, top-30)
    2. Semantic search (bge-m3, top-30)
    3. RRF fusion (merge, top-30)
    4. Cross-encoder rerank (bge-reranker, top-10)
    """

    def __init__(self, chunks_path: Path | str | None = None):
        chunks_path = Path(chunks_path) if chunks_path else CHUNKS_JSONL

        # Load chunk data for text lookup
        self.chunks_by_id: dict[str, dict] = {}
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                chunk = json.loads(line)
                self.chunks_by_id[chunk["chunk_id"]] = chunk

        logger.info(f"Loaded {len(self.chunks_by_id)} chunks")

        # Initialize searchers
        self.bm25 = BM25Searcher()
        self.semantic = SemanticSearcher()
        self.reranker = CrossEncoderReranker()

        logger.info("HybridRetriever initialized")

    def retrieve(
        self,
        query: str,
        rerank_top_k: int | None = None,
    ) -> list[tuple[dict, float]]:
        """
        Full retrieval pipeline for a query.

        Returns list of (chunk_dict, reranker_score) sorted by relevance.
        Each chunk_dict has: chunk_id, doc_id, page_numbers, section_path, doc_title, text
        """
        rerank_top_k = rerank_top_k or RERANK_TOP_K

        # Step 1: BM25 search
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)

        # Step 2: Semantic search
        semantic_results = self.semantic.search(query, top_k=SEMANTIC_TOP_K)

        # Step 3: RRF fusion
        rrf_ranked = self._rrf_fusion(bm25_results, semantic_results)

        # Take top candidates for reranking
        candidates = []
        for chunk_id, rrf_score in rrf_ranked[:RERANK_CANDIDATES]:
            if chunk_id in self.chunks_by_id:
                candidates.append(self.chunks_by_id[chunk_id])

        if not candidates:
            logger.warning(f"No candidates found for query: {query[:80]}...")
            return []

        # Step 4: Cross-encoder rerank
        reranked = self.reranker.rerank(query, candidates, top_k=rerank_top_k)

        return reranked

    def retrieve_without_rerank(
        self,
        query: str,
        top_k: int = 15,
    ) -> list[tuple[dict, float]]:
        """
        Retrieval without reranking (faster, for testing or fallback).
        """
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        semantic_results = self.semantic.search(query, top_k=SEMANTIC_TOP_K)
        rrf_ranked = self._rrf_fusion(bm25_results, semantic_results)

        results = []
        for chunk_id, rrf_score in rrf_ranked[:top_k]:
            if chunk_id in self.chunks_by_id:
                results.append((self.chunks_by_id[chunk_id], rrf_score))

        return results

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
            scores[chunk_id] = scores.get(chunk_id, 0) + 1.0 / (RRF_K + rank + 1)

        for rank, (chunk_id, _) in enumerate(semantic_results):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1.0 / (RRF_K + rank + 1)

        # Sort by combined RRF score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked
