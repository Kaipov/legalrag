from __future__ import annotations

import json
from pathlib import Path

from src.retrieve import hybrid as hybrid_mod


class _FakeBM25Searcher:
    def search(self, query: str, top_k: int = 30):
        return [("a", 10.0), ("b", 9.0)]


class _FakeSemanticSearcher:
    def search(self, query: str, top_k: int = 30):
        return [("b", 8.0), ("c", 7.0)]


def _write_chunks(path: Path) -> None:
    chunks = [
        {"chunk_id": "a", "doc_id": "doc-a", "page_numbers": [1], "section_path": "A", "doc_title": "A", "text": "alpha"},
        {"chunk_id": "b", "doc_id": "doc-b", "page_numbers": [2], "section_path": "B", "doc_title": "B", "text": "beta"},
        {"chunk_id": "c", "doc_id": "doc-c", "page_numbers": [3], "section_path": "C", "doc_title": "C", "text": "gamma"},
    ]
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk) + "\n")


def test_hybrid_retriever_skips_reranker_when_disabled(tmp_path, monkeypatch) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(chunks_path)

    monkeypatch.setattr(hybrid_mod, "BM25Searcher", lambda: _FakeBM25Searcher())
    monkeypatch.setattr(hybrid_mod, "SemanticSearcher", lambda: _FakeSemanticSearcher())

    def _unexpected_reranker():
        raise AssertionError("reranker should not be initialized when disabled")

    monkeypatch.setattr(hybrid_mod, "CrossEncoderReranker", _unexpected_reranker)

    retriever = hybrid_mod.HybridRetriever(chunks_path=chunks_path, enable_reranker=False)
    results = retriever.retrieve("test query", rerank_top_k=2)

    assert [chunk["chunk_id"] for chunk, _score in results] == ["b", "a"]
    assert results[0][1] > results[1][1] > 0


def test_hybrid_retriever_uses_reranker_when_enabled(tmp_path, monkeypatch) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(chunks_path)

    monkeypatch.setattr(hybrid_mod, "BM25Searcher", lambda: _FakeBM25Searcher())
    monkeypatch.setattr(hybrid_mod, "SemanticSearcher", lambda: _FakeSemanticSearcher())

    class _FakeReranker:
        def __init__(self) -> None:
            self.calls = []

        def rerank(self, query: str, chunks: list[dict], top_k: int = 10):
            self.calls.append((query, [chunk["chunk_id"] for chunk in chunks], top_k))
            return [(chunks[-1], 0.9)]

    fake_reranker = _FakeReranker()
    monkeypatch.setattr(hybrid_mod, "CrossEncoderReranker", lambda: fake_reranker)

    retriever = hybrid_mod.HybridRetriever(chunks_path=chunks_path, enable_reranker=True)
    results = retriever.retrieve("test query", rerank_top_k=1)

    assert fake_reranker.calls == [("test query", ["b", "a", "c"], 1)]
    assert results == [({"chunk_id": "c", "doc_id": "doc-c", "page_numbers": [3], "section_path": "C", "doc_title": "C", "text": "gamma"}, 0.9)]
