from __future__ import annotations

import json
from pathlib import Path

from src.retrieve import hybrid as hybrid_mod
from src.retrieve.grounding_policy import GroundingIntent


class _FakeBM25Searcher:
    def search(self, query: str, top_k: int = 30):
        return [("a", 10.0), ("b", 9.0)]


class _FakeSemanticSearcher:
    def search(self, query: str, top_k: int = 30):
        return [("b", 8.0), ("c", 7.0)]


def _write_chunks(path: Path) -> None:
    chunks = [
        {"chunk_id": "a", "doc_id": "doc-a", "page_numbers": [3], "section_path": "A", "doc_title": "A", "text": "alpha"},
        {"chunk_id": "b", "doc_id": "doc-b", "page_numbers": [1], "section_path": "B", "doc_title": "B", "text": "beta caption"},
        {"chunk_id": "c", "doc_id": "doc-c", "page_numbers": [4], "section_path": "C", "doc_title": "C", "text": "gamma"},
    ]
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk) + "\n")


def test_hybrid_retriever_skips_reranker_when_disabled(tmp_path, monkeypatch) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(chunks_path)

    monkeypatch.setattr(hybrid_mod, "BM25Searcher", lambda: _FakeBM25Searcher())
    monkeypatch.setattr(hybrid_mod, "SemanticSearcher", lambda: _FakeSemanticSearcher())
    monkeypatch.setattr(hybrid_mod, "build_reranker", lambda: (_ for _ in ()).throw(AssertionError("should not build")))

    retriever = hybrid_mod.HybridRetriever(chunks_path=chunks_path, enable_reranker=False)
    results = retriever.retrieve("test query", rerank_top_k=2)

    assert [chunk["chunk_id"] for chunk, _score in results] == ["b", "a"]
    assert results[0][1] > results[1][1] > 0


def test_hybrid_retriever_applies_intent_bias_without_reranker(tmp_path, monkeypatch) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(chunks_path)

    monkeypatch.setattr(hybrid_mod, "BM25Searcher", lambda: _FakeBM25Searcher())
    monkeypatch.setattr(hybrid_mod, "SemanticSearcher", lambda: _FakeSemanticSearcher())
    monkeypatch.setattr(hybrid_mod, "build_reranker", lambda: None)

    retriever = hybrid_mod.HybridRetriever(chunks_path=chunks_path, enable_reranker=False)
    intent = GroundingIntent(kind="title_page", page_focus="first", keyphrases=("caption",))
    results = retriever.retrieve("From the title page...", rerank_top_k=1, intent=intent)

    assert [chunk["chunk_id"] for chunk, _score in results] == ["b"]


def test_hybrid_retriever_uses_intent_specific_retrieval_depth(tmp_path, monkeypatch) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as handle:
        for index in range(25):
            chunk = {
                "chunk_id": f"chunk-{index}",
                "doc_id": f"doc-{index}",
                "page_numbers": [index + 1],
                "section_path": f"Section {index}",
                "doc_title": f"Doc {index}",
                "text": f"text {index}",
            }
            handle.write(json.dumps(chunk) + "\n")

    class _DeepFakeBM25Searcher:
        def search(self, query: str, top_k: int = 30):
            return [(f"chunk-{index}", float(30 - index)) for index in range(25)]

    class _DeepFakeSemanticSearcher:
        def search(self, query: str, top_k: int = 30):
            return [(f"chunk-{index}", float(25 - index)) for index in range(25)]

    monkeypatch.setattr(hybrid_mod, "BM25Searcher", lambda: _DeepFakeBM25Searcher())
    monkeypatch.setattr(hybrid_mod, "SemanticSearcher", lambda: _DeepFakeSemanticSearcher())
    monkeypatch.setattr(hybrid_mod, "build_reranker", lambda: None)

    retriever = hybrid_mod.HybridRetriever(chunks_path=chunks_path, enable_reranker=False)
    generic_results = retriever.retrieve("test query", intent=GroundingIntent(kind="generic"))
    deep_results = retriever.retrieve(
        "test query",
        intent=GroundingIntent(kind="date_of_issue", retrieval_top_k=20),
    )

    assert len(generic_results) == hybrid_mod.RERANK_TOP_K
    assert len(deep_results) == 20
    assert [chunk["chunk_id"] for chunk, _score in deep_results[:3]] == ["chunk-0", "chunk-1", "chunk-2"]


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
    monkeypatch.setattr(hybrid_mod, "build_reranker", lambda: fake_reranker)

    retriever = hybrid_mod.HybridRetriever(
        chunks_path=chunks_path,
        enable_reranker=True,
        reranker_enabled_intents=("all",),
    )
    results = retriever.retrieve("test query", rerank_top_k=1, intent=GroundingIntent(kind="generic"))

    assert fake_reranker.calls == [("test query", ["b", "a", "c"], 1)]
    assert results == [({"chunk_id": "c", "doc_id": "doc-c", "page_numbers": [4], "section_path": "C", "doc_title": "C", "text": "gamma"}, 0.9)]


def test_hybrid_retriever_skips_reranker_for_disallowed_intent(tmp_path, monkeypatch) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(chunks_path)

    monkeypatch.setattr(hybrid_mod, "BM25Searcher", lambda: _FakeBM25Searcher())
    monkeypatch.setattr(hybrid_mod, "SemanticSearcher", lambda: _FakeSemanticSearcher())

    class _FakeReranker:
        def rerank(self, query: str, chunks: list[dict], top_k: int = 10):
            raise AssertionError("reranker should not be used for disallowed intents")

    monkeypatch.setattr(hybrid_mod, "build_reranker", lambda: _FakeReranker())

    retriever = hybrid_mod.HybridRetriever(
        chunks_path=chunks_path,
        enable_reranker=True,
        reranker_enabled_intents=("article_ref",),
    )
    results = retriever.retrieve("test query", rerank_top_k=2, intent=GroundingIntent(kind="generic"))

    assert [chunk["chunk_id"] for chunk, _score in results] == ["b", "a"]


def test_hybrid_retriever_falls_back_when_reranker_errors(tmp_path, monkeypatch) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(chunks_path)

    monkeypatch.setattr(hybrid_mod, "BM25Searcher", lambda: _FakeBM25Searcher())
    monkeypatch.setattr(hybrid_mod, "SemanticSearcher", lambda: _FakeSemanticSearcher())

    class _BrokenReranker:
        def rerank(self, query: str, chunks: list[dict], top_k: int = 10):
            raise RuntimeError("voyage timeout")

    monkeypatch.setattr(hybrid_mod, "build_reranker", lambda: _BrokenReranker())

    retriever = hybrid_mod.HybridRetriever(
        chunks_path=chunks_path,
        enable_reranker=True,
        reranker_enabled_intents=("article_ref",),
    )
    results = retriever.retrieve("test query", rerank_top_k=2, intent=GroundingIntent(kind="article_ref"))

    assert [chunk["chunk_id"] for chunk, _score in results] == ["b", "a"]
