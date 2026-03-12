from __future__ import annotations

import json
from pathlib import Path

from src.preprocess.build_index import build_bm25_index
from src.retrieve.bm25 import BM25Searcher
from src.retrieve.lexical import build_bm25_document_tokens, summarize_token_counts, tokenize_legal_text


def _write_chunks(path: Path, chunks: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk) + "\n")


def test_tokenize_legal_text_normalizes_case_and_structural_citations() -> None:
    tokens = tokenize_legal_text("Appeal from ENF-316-2023/2 under Article 14(2)(b).")

    assert "enf316" in tokens
    assert "enf3162023" in tokens
    assert "enf3162023_2" in tokens
    assert "article14" in tokens
    assert "article14_2" in tokens
    assert "article14_2_b" in tokens


def test_build_bm25_document_tokens_boosts_title_and_section_fields() -> None:
    chunk = {
        "doc_title": "Operating Law",
        "section_path": "Part 3 > Article 9",
        "text": "Licence",
    }

    counts = summarize_token_counts(build_bm25_document_tokens(chunk))

    assert counts["operating"] == 3
    assert counts["law"] == 3
    assert counts["part3"] == 2
    assert counts["article9"] == 2
    assert counts["licence"] == 1



def test_bm25_search_uses_title_and_section_tokens(tmp_path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    index_path = tmp_path / "bm25.pkl"
    _write_chunks(
        chunks_path,
        [
            {
                "chunk_id": "operating-law",
                "doc_id": "doc-1",
                "page_numbers": [7],
                "doc_title": "Operating Law",
                "section_path": "Part 3 > 9. Licence",
                "text": "A licence takes effect for twelve months from the date specified on the Licence.",
            },
            {
                "chunk_id": "other-law",
                "doc_id": "doc-2",
                "page_numbers": [3],
                "doc_title": "Employment Law",
                "section_path": "Definitions",
                "text": "A licence takes effect for twelve months from the date specified on the Licence.",
            },
            {
                "chunk_id": "unrelated",
                "doc_id": "doc-3",
                "page_numbers": [9],
                "doc_title": "Trust Law",
                "section_path": "Schedule 1",
                "text": "Trustees must keep records for at least six years.",
            },
        ],
    )

    build_bm25_index(chunks_path, index_path)
    searcher = BM25Searcher(index_path=index_path)
    results = searcher.search("According to Article 9 of the Operating Law, how long does a Licence take effect for?", top_k=2)

    assert results[0][0] == "operating-law"



def test_bm25_search_matches_case_citation_variants(tmp_path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    index_path = tmp_path / "bm25.pkl"
    _write_chunks(
        chunks_path,
        [
            {
                "chunk_id": "appeal",
                "doc_id": "doc-1",
                "page_numbers": [2],
                "doc_title": "CA 009/2024 Judgment",
                "section_path": "Page 2",
                "text": "This appeal arose from claim number ENF-316-2023/2.",
            },
            {
                "chunk_id": "other",
                "doc_id": "doc-2",
                "page_numbers": [4],
                "doc_title": "CA 005/2025 Judgment",
                "section_path": "Page 4",
                "text": "This appeal arose from another enforcement claim.",
            },
            {
                "chunk_id": "unrelated",
                "doc_id": "doc-3",
                "page_numbers": [8],
                "doc_title": "CFI 067/2025 Judgment",
                "section_path": "Page 8",
                "text": "The court dismissed the application with costs.",
            },
        ],
    )

    build_bm25_index(chunks_path, index_path)
    searcher = BM25Searcher(index_path=index_path)
    results = searcher.search("Which appeal arose from ENF 316/2023?", top_k=2)

    assert results[0][0] == "appeal"

