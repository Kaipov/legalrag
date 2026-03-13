from __future__ import annotations

from src.config import EMBEDDING_PROVIDER
from src.embeddings.gemini import GeminiApiError, GeminiEmbeddingClient

_embedding_client = None


def get_embedding_client():
    global _embedding_client
    if _embedding_client is None:
        if EMBEDDING_PROVIDER == "gemini":
            _embedding_client = GeminiEmbeddingClient()
        else:  # pragma: no cover - currently only Gemini is supported
            raise ValueError(f"Unsupported embedding provider: {EMBEDDING_PROVIDER}")
    return _embedding_client


__all__ = ["GeminiApiError", "GeminiEmbeddingClient", "get_embedding_client"]