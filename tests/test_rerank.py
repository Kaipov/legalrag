from __future__ import annotations

from src.retrieve.rerank import VoyageAPIReranker


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeSession:
    def __init__(self, payload: dict):
        self.headers: dict[str, str] = {}
        self.payload = payload
        self.calls: list[tuple[str, dict, int]] = []

    def post(self, url: str, json: dict, timeout: int):
        self.calls.append((url, json, timeout))
        return _FakeResponse(self.payload)


def test_voyage_api_reranker_reranks_by_returned_indexes() -> None:
    session = _FakeSession(
        {
            "data": [
                {"index": 1, "relevance_score": 0.91},
                {"index": 0, "relevance_score": 0.42},
            ]
        }
    )
    reranker = VoyageAPIReranker(
        api_key="voyage-test-key",
        session=session,
        timeout_seconds=7,
        model_name="rerank-2.5",
        api_base="https://api.voyageai.com/v1",
    )

    chunks = [
        {"chunk_id": "a", "text": "alpha"},
        {"chunk_id": "b", "text": "beta"},
    ]
    results = reranker.rerank("test query", chunks, top_k=2)

    assert [chunk["chunk_id"] for chunk, _score in results] == ["b", "a"]
    assert results[0][1] == 0.91
    assert session.headers["Authorization"] == "Bearer voyage-test-key"
    assert session.calls == [
        (
            "https://api.voyageai.com/v1/rerank",
            {
                "model": "rerank-2.5",
                "query": "test query",
                "documents": ["alpha", "beta"],
                "top_k": 2,
            },
            7,
        )
    ]
