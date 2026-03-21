"""
Reranker providers for late-stage chunk reranking.

Supports:
- local cross-encoder rerank via sentence-transformers
- Voyage API rerank via HTTPS
"""
from __future__ import annotations

import logging
from typing import Any

import requests

from src.config import (
    DEVICE,
    RERANKER_BATCH_SIZE,
    RERANKER_MAX_LENGTH,
    RERANKER_MODEL,
    RERANKER_PROVIDER,
    RERANKER_TIMEOUT_SECONDS,
    RERANKER_USE_FP16,
    VOYAGE_API_BASE,
    VOYAGE_API_KEY,
    VOYAGE_RERANKER_MODEL,
)

logger = logging.getLogger(__name__)


def _chunk_rerank_text(chunk: dict[str, Any]) -> str:
    return str(chunk.get("text") or "")


class CrossEncoderReranker:
    """Cross-encoder reranker using a local sentence-transformers model."""

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

        model_kwargs: dict[str, Any] = {}
        dtype_name = "default"
        if DEVICE == "cuda" and self.use_fp16:
            import torch

            model_kwargs["torch_dtype"] = torch.float16
            dtype_name = "float16"
        elif self.use_fp16:
            logger.info("Reranker fp16 requested but device=%s; using default dtype", DEVICE)

        logger.info(
            "Loading local reranker %s on %s (max_length=%s, batch_size=%s, dtype=%s)...",
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
        logger.info("Local reranker loaded successfully")

    def rerank(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        top_k: int = 10,
    ) -> list[tuple[dict[str, Any], float]]:
        if not chunks:
            return []

        pairs = [(query, _chunk_rerank_text(chunk)) for chunk in chunks]
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        scored = list(zip(chunks, scores))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [(chunk, float(score)) for chunk, score in scored[:top_k]]


class VoyageAPIReranker:
    """Hosted Voyage reranker with the same interface as the local reranker."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        model_name: str | None = None,
        timeout_seconds: int | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.api_key = str(api_key or VOYAGE_API_KEY).strip()
        if not self.api_key:
            raise ValueError("VOYAGE_API_KEY is required for the Voyage reranker")

        self.api_base = str(api_base or VOYAGE_API_BASE).rstrip("/")
        self.model_name = model_name or VOYAGE_RERANKER_MODEL
        self.timeout_seconds = timeout_seconds or RERANKER_TIMEOUT_SECONDS
        self.session = session or requests.Session()
        self.endpoint = f"{self.api_base}/rerank"
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

        logger.info(
            "Voyage reranker configured (model=%s, timeout=%ss, endpoint=%s)",
            self.model_name,
            self.timeout_seconds,
            self.endpoint,
        )

    def rerank(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        top_k: int = 10,
    ) -> list[tuple[dict[str, Any], float]]:
        if not chunks:
            return []

        payload = {
            "model": self.model_name,
            "query": query,
            "documents": [_chunk_rerank_text(chunk) for chunk in chunks],
            "top_k": max(1, min(top_k, len(chunks))),
        }
        response = self.session.post(
            self.endpoint,
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()

        results = body.get("data")
        if not isinstance(results, list):
            results = body.get("results")
        if not isinstance(results, list):
            raise ValueError("Voyage reranker response is missing a results list")

        scored: list[tuple[dict[str, Any], float]] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            index = item.get("index")
            score = item.get("relevance_score", item.get("score"))
            if not isinstance(index, int) or index < 0 or index >= len(chunks):
                continue
            if not isinstance(score, (int, float)):
                continue
            scored.append((chunks[index], float(score)))

        if not scored:
            raise ValueError("Voyage reranker returned no valid results")
        return scored


def build_reranker() -> CrossEncoderReranker | VoyageAPIReranker | None:
    """Instantiate the configured reranker provider, or return None on safe fallback."""
    provider = str(RERANKER_PROVIDER or "local").lower()
    try:
        if provider == "voyage":
            return VoyageAPIReranker()
        if provider == "local":
            return CrossEncoderReranker()
        logger.warning("Unknown reranker provider '%s'; reranking disabled", provider)
        return None
    except Exception as exc:  # pragma: no cover - provider failures depend on local env/network
        logger.warning("Failed to initialize reranker provider '%s': %s", provider, exc)
        return None
