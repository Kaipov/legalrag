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
    RERANKER_ENABLED_INTENTS,
    RERANK_TOP_K,
    RRF_K,
    SEMANTIC_TOP_K,
)
from src.retrieve.bm25 import BM25Searcher
from src.retrieve.grounding_policy import GroundingIntent, score_chunk_for_intent
from src.retrieve.rerank import build_reranker
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
        reranker_enabled_intents: tuple[str, ...] | None = None,
    ):
        chunks_path = Path(chunks_path) if chunks_path else CHUNKS_JSONL
        self.enable_reranker = ENABLE_RERANKER if enable_reranker is None else enable_reranker
        self.reranker_enabled_intents = (
            tuple(intent.strip().lower() for intent in reranker_enabled_intents if intent and intent.strip())
            if reranker_enabled_intents is not None
            else RERANKER_ENABLED_INTENTS
        )

        self.chunks_by_id: dict[str, dict] = {}
        self.doc_max_page_by_id: dict[str, int] = {}
        with open(chunks_path, "r", encoding="utf-8") as handle:
            for line in handle:
                chunk = json.loads(line)
                self.chunks_by_id[chunk["chunk_id"]] = chunk
                doc_id = str(chunk.get("doc_id") or "").strip()
                if doc_id:
                    max_page = max(
                        (int(page) for page in chunk.get("page_numbers", []) if isinstance(page, int)),
                        default=0,
                    )
                    if max_page > 0:
                        self.doc_max_page_by_id[doc_id] = max(max_page, self.doc_max_page_by_id.get(doc_id, 0))

        logger.info(f"Loaded {len(self.chunks_by_id)} chunks")

        self.bm25 = BM25Searcher()
        self.semantic = SemanticSearcher()
        self.reranker = build_reranker() if self.enable_reranker else None

        if self.enable_reranker and self.reranker is not None:
            logger.info(
                "HybridRetriever initialized with reranker enabled for intents=%s",
                ",".join(self.reranker_enabled_intents) or "(none)",
            )
        elif self.enable_reranker:
            logger.info("HybridRetriever initialized with reranker requested but unavailable; using fallback retrieval")
        else:
            logger.info("HybridRetriever initialized with reranker disabled")

    def retrieve(
        self,
        query: str,
        rerank_top_k: int | None = None,
        intent: GroundingIntent | None = None,
    ) -> list[tuple[dict, float]]:
        """
        Full retrieval pipeline for a query.

        Returns list of (chunk_dict, score) sorted by relevance.
        Score is the cross-encoder logit when reranker is enabled, otherwise the RRF score.
        """
        rerank_top_k = rerank_top_k or RERANK_TOP_K
        candidates = self._get_rrf_candidates(query)
        candidates = self._apply_intent_bias(candidates, intent)

        if not candidates:
            logger.warning(f"No candidates found for query: {query[:80]}...")
            return []

        if not self._should_rerank(intent):
            return candidates[:rerank_top_k]

        try:
            reranked = self.reranker.rerank(
                query,
                [chunk for chunk, _score in candidates],
                top_k=rerank_top_k,
            )
        except Exception as exc:
            logger.warning("Reranker failed for query '%s...': %s. Falling back to baseline retrieval.", query[:80], exc)
            return candidates[:rerank_top_k]
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

    def _apply_intent_bias(
        self,
        candidates: list[tuple[dict, float]],
        intent: GroundingIntent | None,
    ) -> list[tuple[dict, float]]:
        if intent is None or intent.kind == "generic":
            return candidates

        ranked = []
        for index, (chunk, score) in enumerate(candidates):
            doc_id = str(chunk.get("doc_id") or "").strip()
            doc_max_page = self.doc_max_page_by_id.get(doc_id)
            intent_bias = score_chunk_for_intent(chunk, intent, doc_max_page)
            ranked.append((intent_bias, score, index, chunk, score))

        ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
        return [(chunk, score) for _bias, _base_score, _index, chunk, score in ranked]

    def _should_rerank(self, intent: GroundingIntent | None) -> bool:
        if not self.enable_reranker or self.reranker is None:
            return False

        if not self.reranker_enabled_intents:
            return False

        allowlist = set(self.reranker_enabled_intents)
        if "all" in allowlist:
            return True

        intent_kind = str(getattr(intent, "kind", "") or "").lower()
        return bool(intent_kind and intent_kind in allowlist)

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
