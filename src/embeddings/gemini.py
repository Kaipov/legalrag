from __future__ import annotations

import json
import logging
import time

import numpy as np
import requests

from src.config import (
    EMBEDDING_API_BASE,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DOCUMENT_TASK_TYPE,
    EMBEDDING_MAX_RETRIES,
    EMBEDDING_MODEL,
    EMBEDDING_OUTPUT_DIMENSION,
    EMBEDDING_QUERY_TASK_TYPE,
    EMBEDDING_REQUEST_TIMEOUT_SECONDS,
    get_embedding_api_key,
)

logger = logging.getLogger(__name__)
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class GeminiApiError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None, response_text: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text


class GeminiEmbeddingClient:
    """Gemini embedding client shared by chunking, indexing, and semantic retrieval."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_name: str | None = None,
        api_base: str | None = None,
        batch_size: int | None = None,
        timeout_seconds: int | None = None,
        max_retries: int | None = None,
        query_task_type: str | None = None,
        document_task_type: str | None = None,
        output_dimensionality: int | None = EMBEDDING_OUTPUT_DIMENSION,
        session: requests.Session | None = None,
    ) -> None:
        self.api_key = (api_key or get_embedding_api_key()).strip()
        if not self.api_key:
            raise ValueError("No Gemini embedding API key configured. Set GEMINI_API_KEY.")

        self.model_name = (model_name or EMBEDDING_MODEL).strip()
        self.api_base = (api_base or EMBEDDING_API_BASE).rstrip("/")
        self.batch_size = max(1, batch_size if batch_size is not None else EMBEDDING_BATCH_SIZE)
        self.timeout_seconds = max(
            1,
            timeout_seconds if timeout_seconds is not None else EMBEDDING_REQUEST_TIMEOUT_SECONDS,
        )
        self.max_retries = max(1, max_retries if max_retries is not None else EMBEDDING_MAX_RETRIES)
        self.query_task_type = (query_task_type or EMBEDDING_QUERY_TASK_TYPE).strip()
        self.document_task_type = (document_task_type or EMBEDDING_DOCUMENT_TASK_TYPE).strip()
        self.output_dimensionality = output_dimensionality
        self.session = session or requests.Session()

    def _url(self, action: str) -> str:
        return f"{self.api_base}/{self.model_name}:{action}?key={self.api_key}"

    @staticmethod
    def _extract_error_message(response_text: str) -> str:
        try:
            payload = json.loads(response_text)
        except json.JSONDecodeError:
            return response_text.strip() or "unknown error"

        error = payload.get("error") or {}
        return str(error.get("message") or response_text or "unknown error").strip()

    def _post_json(self, action: str, payload: dict) -> dict:
        url = self._url(action)
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.post(
                    url,
                    json=payload,
                    timeout=self.timeout_seconds,
                )
            except requests.RequestException as exc:
                last_error = exc
                if attempt == self.max_retries:
                    raise GeminiApiError(
                        f"Gemini {action} request failed after {attempt} attempts: {exc}"
                    ) from exc
                time.sleep(min(8, 2 ** (attempt - 1)))
                continue

            if response.status_code < 400:
                return response.json()

            message = self._extract_error_message(response.text)
            if response.status_code in _RETRYABLE_STATUS_CODES and attempt < self.max_retries:
                logger.warning(
                    "Gemini %s request failed with HTTP %s on attempt %s/%s: %s",
                    action,
                    response.status_code,
                    attempt,
                    self.max_retries,
                    message,
                )
                time.sleep(min(8, 2 ** (attempt - 1)))
                continue

            raise GeminiApiError(
                f"Gemini {action} request failed ({response.status_code}): {message}",
                status_code=response.status_code,
                response_text=response.text,
            )

        raise GeminiApiError(f"Gemini {action} request failed: {last_error}")

    @staticmethod
    def _build_content(text: str) -> dict:
        return {"parts": [{"text": text}]}

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (embeddings / norms).astype(np.float32)

    def count_tokens(self, text: str) -> int:
        text = text.strip()
        if not text:
            return 0

        payload = {"contents": [self._build_content(text)]}
        response = self._post_json("countTokens", payload)
        return int(response.get("totalTokens") or 0)

    def embed_texts(
        self,
        texts: list[str],
        *,
        task_type: str,
        titles: list[str | None] | None = None,
    ) -> np.ndarray:
        if not texts:
            width = self.output_dimensionality or 0
            return np.zeros((0, width), dtype=np.float32)

        if titles is not None and len(titles) != len(texts):
            raise ValueError("titles length must match texts length")

        requests_payload = []
        for idx, text in enumerate(texts):
            request = {
                "model": self.model_name,
                "content": self._build_content(text),
                "taskType": task_type,
            }
            title = titles[idx] if titles is not None else None
            if title:
                request["title"] = title
            if self.output_dimensionality is not None:
                request["outputDimensionality"] = self.output_dimensionality
            requests_payload.append(request)

        response = self._post_json("batchEmbedContents", {"requests": requests_payload})
        embeddings_payload = response.get("embeddings") or []
        if len(embeddings_payload) != len(texts):
            raise GeminiApiError(
                "Gemini batchEmbedContents returned an unexpected number of embeddings",
                response_text=json.dumps(response),
            )

        vectors = np.asarray(
            [embedding.get("values") or [] for embedding in embeddings_payload],
            dtype=np.float32,
        )
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise GeminiApiError(
                "Gemini batchEmbedContents returned malformed embedding vectors",
                response_text=json.dumps(response),
            )

        return self._normalize_embeddings(vectors)

    def embed_documents(
        self,
        texts: list[str],
        *,
        titles: list[str | None] | None = None,
    ) -> np.ndarray:
        return self.embed_texts(
            texts,
            task_type=self.document_task_type,
            titles=titles,
        )

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self.embed_texts(texts, task_type=self.query_task_type)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_queries([query])