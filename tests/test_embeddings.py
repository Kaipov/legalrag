from __future__ import annotations

import numpy as np

from src.embeddings.gemini import GeminiEmbeddingClient


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self) -> dict:
        return self._payload


class _FakeSession:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self.responses = responses
        self.calls: list[dict] = []

    def post(self, url: str, json: dict, timeout: int):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        if not self.responses:
            raise AssertionError("no fake responses left")
        return self.responses.pop(0)


def test_count_tokens_uses_embedding_model_endpoint() -> None:
    session = _FakeSession([_FakeResponse({"totalTokens": 7})])
    client = GeminiEmbeddingClient(
        api_key="test-key",
        model_name="models/gemini-embedding-2-preview",
        session=session,
    )

    total_tokens = client.count_tokens("alpha beta gamma")

    assert total_tokens == 7
    assert session.calls[0]["url"].endswith("models/gemini-embedding-2-preview:countTokens?key=test-key")
    assert session.calls[0]["json"] == {
        "contents": [{"parts": [{"text": "alpha beta gamma"}]}]
    }


def test_embed_documents_normalizes_vectors_and_preserves_titles() -> None:
    session = _FakeSession([
        _FakeResponse(
            {
                "embeddings": [
                    {"values": [3.0, 4.0]},
                    {"values": [0.0, 5.0]},
                ]
            }
        )
    ])
    client = GeminiEmbeddingClient(
        api_key="test-key",
        model_name="models/gemini-embedding-2-preview",
        output_dimensionality=2,
        session=session,
    )

    vectors = client.embed_documents(
        ["alpha text", "beta text"],
        titles=["Title A", "Title B"],
    )

    assert vectors.shape == (2, 2)
    assert np.allclose(vectors[0], np.array([0.6, 0.8], dtype=np.float32))
    assert np.allclose(vectors[1], np.array([0.0, 1.0], dtype=np.float32))

    payload = session.calls[0]["json"]
    assert payload["requests"][0]["taskType"] == "RETRIEVAL_DOCUMENT"
    assert payload["requests"][0]["title"] == "Title A"
    assert payload["requests"][0]["outputDimensionality"] == 2


def test_embed_query_uses_retrieval_query_task_type() -> None:
    session = _FakeSession([_FakeResponse({"embeddings": [{"values": [1.0, 0.0]}]})])
    client = GeminiEmbeddingClient(
        api_key="test-key",
        model_name="models/gemini-embedding-2-preview",
        session=session,
    )

    vectors = client.embed_query("who issued the order?")

    assert vectors.shape == (1, 2)
    assert session.calls[0]["json"]["requests"][0]["taskType"] == "RETRIEVAL_QUERY"