"""Page-level retrieval for grounding rescue."""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np

from src.config import (
    PAGE_BM25_INDEX,
    PAGE_BM25_TOP_K,
    PAGE_FAISS_IDS,
    PAGE_FAISS_INDEX,
    PAGE_GROUNDING_TOP_K,
    PAGE_SEMANTIC_TOP_K,
    RRF_K,
)
from src.embeddings import get_embedding_client
from src.retrieve.lexical import tokenize_legal_text

logger = logging.getLogger(__name__)



def make_page_id(doc_id: str, page_num: int) -> str:
    return f"{doc_id}:{int(page_num)}"



def parse_page_id(page_id: str) -> tuple[str, int]:
    doc_id, page_num = str(page_id).rsplit(":", 1)
    return doc_id, int(page_num)


class PageBM25Searcher:
    """BM25 keyword search over page-level index."""

    def __init__(self, index_path: Path | str | None = None):
        index_path = Path(index_path) if index_path else PAGE_BM25_INDEX
        with open(index_path, "rb") as handle:
            data = pickle.load(handle)

        self.bm25 = data["bm25"]
        self.page_ids = data["page_ids"]
        logger.info("Page BM25 index loaded: %s pages", len(self.page_ids))

    def search(
        self,
        query: str,
        top_k: int = PAGE_BM25_TOP_K,
        allowed_doc_ids: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        tokens = tokenize_legal_text(query)
        if not tokens or top_k <= 0:
            return []

        allowed = {str(doc_id).strip() for doc_id in (allowed_doc_ids or set()) if str(doc_id).strip()}
        scores = self.bm25.get_scores(tokens)
        ranked_indices = scores.argsort()[::-1]

        results: list[tuple[str, float]] = []
        for idx in ranked_indices:
            score = float(scores[idx])
            if score <= 0:
                break

            page_id = str(self.page_ids[idx])
            doc_id, _page_num = parse_page_id(page_id)
            if allowed and doc_id not in allowed:
                continue

            results.append((page_id, score))
            if len(results) >= top_k:
                break

        return results


class PageSemanticSearcher:
    """FAISS dense search over page embeddings."""

    def __init__(
        self,
        index_path: Path | str | None = None,
        ids_path: Path | str | None = None,
    ):
        import faiss

        index_path = Path(index_path) if index_path else PAGE_FAISS_INDEX
        ids_path = Path(ids_path) if ids_path else PAGE_FAISS_IDS

        self.index = faiss.read_index(str(index_path))
        with open(ids_path, "r", encoding="utf-8") as handle:
            self.page_ids = json.load(handle)

        self.client = get_embedding_client()
        logger.info("Page semantic index loaded: %s pages", len(self.page_ids))

    def embed_query(self, query: str) -> np.ndarray:
        return self.client.embed_query(query).astype(np.float32)

    def search(
        self,
        query: str,
        top_k: int = PAGE_SEMANTIC_TOP_K,
        allowed_doc_ids: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        if top_k <= 0:
            return []

        allowed = {str(doc_id).strip() for doc_id in (allowed_doc_ids or set()) if str(doc_id).strip()}
        query_embedding = self.embed_query(query)
        candidate_k = len(self.page_ids) if allowed else min(len(self.page_ids), max(top_k, PAGE_SEMANTIC_TOP_K))
        scores, indices = self.index.search(query_embedding, candidate_k)

        results: list[tuple[str, float]] = []
        seen_page_ids: set[str] = set()
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.page_ids):
                continue
            page_id = str(self.page_ids[idx])
            if page_id in seen_page_ids:
                continue
            doc_id, _page_num = parse_page_id(page_id)
            if allowed and doc_id not in allowed:
                continue

            results.append((page_id, float(score)))
            seen_page_ids.add(page_id)
            if len(results) >= top_k:
                break

        return results


class PageRetriever:
    """Hybrid page-level retrieval for grounding rescue."""

    def __init__(
        self,
        *,
        bm25_searcher: PageBM25Searcher | None = None,
        semantic_searcher: PageSemanticSearcher | None = None,
    ):
        self.bm25 = bm25_searcher or PageBM25Searcher()
        self.semantic = semantic_searcher or PageSemanticSearcher()

    def _rrf_fusion(
        self,
        bm25_results: list[tuple[str, float]],
        semantic_results: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        scores: dict[str, float] = {}

        for rank, (page_id, _score) in enumerate(bm25_results):
            scores[page_id] = scores.get(page_id, 0.0) + 1.0 / (RRF_K + rank + 1)

        for rank, (page_id, _score) in enumerate(semantic_results):
            scores[page_id] = scores.get(page_id, 0.0) + 1.0 / (RRF_K + rank + 1)

        return sorted(scores.items(), key=lambda item: item[1], reverse=True)

    def search(
        self,
        query: str,
        *,
        top_k: int = PAGE_GROUNDING_TOP_K,
        allowed_doc_ids: set[str] | None = None,
    ) -> list[tuple[dict, float]]:
        if top_k <= 0:
            return []

        bm25_results = self.bm25.search(query, top_k=max(top_k, PAGE_BM25_TOP_K), allowed_doc_ids=allowed_doc_ids)
        semantic_results = self.semantic.search(
            query,
            top_k=max(top_k, PAGE_SEMANTIC_TOP_K),
            allowed_doc_ids=allowed_doc_ids,
        )
        fused = self._rrf_fusion(bm25_results, semantic_results)

        results: list[tuple[dict, float]] = []
        for page_id, score in fused[:top_k]:
            doc_id, page_num = parse_page_id(page_id)
            results.append(({"doc_id": doc_id, "page_num": page_num}, float(score)))
        return results
