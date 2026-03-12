from __future__ import annotations

import json
from pathlib import Path

from src.preprocess.build_index import build_page_bm25_index
from src.retrieve.page_search import PageBM25Searcher, PageRetriever



def _write_pages(path: Path, pages: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for page in pages:
            handle.write(json.dumps(page) + "\n")



def test_build_page_bm25_index_prefers_title_page_with_doc_restriction(tmp_path) -> None:
    pages_path = tmp_path / "pages.jsonl"
    index_path = tmp_path / "page_bm25.pkl"
    _write_pages(
        pages_path,
        [
            {"doc_id": "doc-a", "page_num": 1, "text": "OPERATING LAW\nDIFC Law No. 7 of 2018\nTitle page text."},
            {"doc_id": "doc-a", "page_num": 2, "text": "Body provisions about licence duration."},
            {"doc_id": "doc-b", "page_num": 1, "text": "EMPLOYMENT LAW\nDefinitions page."},
        ],
    )

    build_page_bm25_index(pages_path, index_path)
    searcher = PageBM25Searcher(index_path=index_path)
    results = searcher.search(
        "What is stated on the title page of the Operating Law?",
        top_k=2,
        allowed_doc_ids={"doc-a"},
    )

    assert results[0][0] == "doc-a:1"
    assert all(page_id.startswith("doc-a:") for page_id, _score in results)



def test_page_retriever_rrf_merges_bm25_and_semantic_with_doc_filter() -> None:
    class FakeBM25Searcher:
        def search(self, query: str, top_k: int = 20, allowed_doc_ids: set[str] | None = None):
            assert allowed_doc_ids == {"doc-a"}
            return [("doc-a:1", 12.0), ("doc-a:2", 9.0)]

    class FakeSemanticSearcher:
        def search(self, query: str, top_k: int = 20, allowed_doc_ids: set[str] | None = None):
            assert allowed_doc_ids == {"doc-a"}
            return [("doc-a:2", 0.9), ("doc-a:1", 0.7)]

    retriever = PageRetriever(
        bm25_searcher=FakeBM25Searcher(),
        semantic_searcher=FakeSemanticSearcher(),
    )
    results = retriever.search("Operating Law title page", top_k=2, allowed_doc_ids={"doc-a"})

    assert results == [
        ({"doc_id": "doc-a", "page_num": 1}, results[0][1]),
        ({"doc_id": "doc-a", "page_num": 2}, results[1][1]),
    ]
