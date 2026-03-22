from __future__ import annotations

import numpy as np
from pathlib import Path

from src.embeddings import GeminiApiError
from src.preprocess import chunk as chunk_mod
from src.preprocess import extract as extract_mod
from src.preprocess.build_index import _encode_with_backoff


def _fake_count_tokens(text: str) -> int:
    return max(1, len(text.split())) if text.strip() else 0


def test_split_text_to_fit_handles_single_oversized_paragraph(monkeypatch) -> None:
    monkeypatch.setattr(chunk_mod, "_count_tokens", _fake_count_tokens)
    text = " ".join(f"word{i}" for i in range(300))

    parts = chunk_mod._split_text_to_fit(text, 50)

    assert len(parts) > 1
    assert all(chunk_mod._count_tokens(part) <= 50 for part in parts)


def test_chunk_document_splits_oversized_single_page(monkeypatch) -> None:
    monkeypatch.setattr(chunk_mod, "_count_tokens", _fake_count_tokens)
    monkeypatch.setattr(chunk_mod, "MAX_CHUNK_TOKENS", 50)
    monkeypatch.setattr(chunk_mod, "MIN_CHUNK_TOKENS", 10)
    huge_page = " ".join(f"word{i}" for i in range(300))

    chunks = chunk_mod.chunk_document(
        "doc",
        [{"page_num": 1, "text": huge_page}],
    )

    assert len(chunks) > 1
    assert all(chunk_mod._count_tokens(chunk["text"]) <= 50 for chunk in chunks)
    assert all(chunk["page_numbers"] == [1] for chunk in chunks)


def test_chunk_document_maps_split_preamble_to_narrower_pages(monkeypatch) -> None:
    monkeypatch.setattr(chunk_mod, "_count_tokens", _fake_count_tokens)
    monkeypatch.setattr(chunk_mod, "MAX_CHUNK_TOKENS", 20)
    monkeypatch.setattr(chunk_mod, "MIN_CHUNK_TOKENS", 1)
    page_one = " ".join(f"intro{i}" for i in range(20))
    page_two = " ".join(f"more{i}" for i in range(20)) + "\n\nArticle 1\n" + " ".join(
        f"body{i}" for i in range(10)
    )

    chunks = chunk_mod.chunk_document(
        "doc",
        [
            {"page_num": 1, "text": page_one},
            {"page_num": 2, "text": page_two},
        ],
    )
    preamble_chunks = [chunk for chunk in chunks if "Preamble" in chunk["section_path"]]

    assert len(preamble_chunks) > 1
    assert any(chunk["page_numbers"] == [1] for chunk in preamble_chunks)
    assert any(2 in chunk["page_numbers"] for chunk in preamble_chunks)
    assert not all(chunk["page_numbers"] == [1, 2] for chunk in preamble_chunks)


def test_chunk_document_maps_split_segments_to_narrower_pages(monkeypatch) -> None:
    monkeypatch.setattr(chunk_mod, "_count_tokens", _fake_count_tokens)
    monkeypatch.setattr(chunk_mod, "MAX_CHUNK_TOKENS", 20)
    monkeypatch.setattr(chunk_mod, "MIN_CHUNK_TOKENS", 1)
    pages = [
        {"page_num": 1, "text": "Article 1\n" + " ".join(f"page1_{i}" for i in range(18))},
        {"page_num": 2, "text": " ".join(f"page2_{i}" for i in range(18))},
        {"page_num": 3, "text": " ".join(f"page3_{i}" for i in range(18))},
    ]

    chunks = chunk_mod.chunk_document("doc", pages)
    article_chunks = [chunk for chunk in chunks if "Article 1" in chunk["section_path"]]

    assert len(article_chunks) > 1
    assert any(chunk["page_numbers"] == [1] for chunk in article_chunks)
    assert any(chunk["page_numbers"] == [2] for chunk in article_chunks)
    assert any(chunk["page_numbers"] == [3] for chunk in article_chunks)
    assert not all(chunk["page_numbers"] == [1, 2, 3] for chunk in article_chunks)


def test_find_boundaries_detects_numbered_statutory_headings() -> None:
    text = (
        "GENERAL PARTNERSHIP LAW\n"
        "PART 2: FORMATION AND REGISTRATION\n"
        "13. Recognised Partnership\n"
        "(1) The Partners of a General Partnership formed outside of the DIFC...\n"
        "14. Maintenance of Accounting Records\n"
        "(1) A Recognised Partnership shall keep Accounting Records...\n"
        "15. Licence\n"
        "A General Partnership shall hold a Commercial Licence.\n"
    )

    boundaries = chunk_mod._find_boundaries(text, doc_title="GENERAL PARTNERSHIP LAW")
    labels = [label for _pos, _kind, label in boundaries]

    assert labels == [
        "13. Recognised Partnership",
        "14. Maintenance of Accounting Records",
        "15. Licence",
    ]


def test_find_boundaries_ignores_inline_article_references() -> None:
    text = (
        "OPERATING LAW\n"
        "PART 3: CONDUCT OF BUSINESS IN THE DIFC\n"
        "15. Licence\n"
        "A General Partnership shall hold a Commercial Licence pursuant to\n"
        "Article 9 of the Operating Law.\n"
        "16. Conduct of Business in the DIFC\n"
        "A General Partnership shall comply with the requirements of this Law.\n"
    )

    boundaries = chunk_mod._find_boundaries(text, doc_title="OPERATING LAW")
    labels = [label for _pos, _kind, label in boundaries]

    assert labels == [
        "15. Licence",
        "16. Conduct of Business in the DIFC",
    ]


def test_find_boundaries_ignores_table_of_contents_lines() -> None:
    text = (
        "TRUST LAW\n"
        "CONTENTS\n"
        "1. Title ................................................................ 1\n"
        "2. Legislative authority .................................................. 1\n"
        "PART 1: GENERAL\n"
        "1. Title\n"
        "This Law repeals and replaces the Trust Law 2005.\n"
    )

    boundaries = chunk_mod._find_boundaries(text, doc_title="TRUST LAW")
    labels = [label for _pos, _kind, label in boundaries]

    assert labels == ["1. Title"]


def test_find_boundaries_accepts_indented_article_heading_with_punctuation() -> None:
    text = (
        "DIGITAL ASSETS LAW\n"
        "  Article 15.\n"
        "General obligations of issuers\n"
        "Text body.\n"
        "  Article 16:\n"
        "Compliance requirements\n"
        "More text.\n"
    )

    boundaries = chunk_mod._find_boundaries(text, doc_title="DIGITAL ASSETS LAW")
    labels = [label for _pos, _kind, label in boundaries]

    assert labels == ["Article 15.", "Article 16:"]


class _FakeEmbeddingClient:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def embed_documents(self, texts, *, titles=None):
        self.calls.append(len(texts))
        if len(texts) > 2:
            raise GeminiApiError("payload too large", status_code=413, response_text="payload too large")
        return np.ones((len(texts), 3), dtype=np.float32)


def test_encode_with_backoff_reduces_batch_size() -> None:
    client = _FakeEmbeddingClient()

    embeddings, used_batch_size = _encode_with_backoff(client, ["a", "b", "c", "d"], 8)

    assert used_batch_size == 2
    assert client.calls == [4, 2, 2]
    assert embeddings.shape == (4, 3)
    assert embeddings.dtype == np.float32


class _FakePdf:
    def __init__(self, pages):
        self.pages = pages

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_extract_single_pdf_uses_vision_fallback_when_text_is_empty(monkeypatch) -> None:
    monkeypatch.setattr(
        extract_mod.pdfplumber,
        "open",
        lambda *_args, **_kwargs: _FakePdf([object()]),
    )
    monkeypatch.setattr(extract_mod, "_extract_page_pdfplumber", lambda _page: "")
    monkeypatch.setattr(extract_mod, "_extract_page_ocr", lambda _pdf_path, _page_num: "")
    monkeypatch.setattr(extract_mod, "_extract_page_vision", lambda _pdf_path, _page_num: "Recovered scanned text")

    rows = list(extract_mod.extract_single_pdf(Path("sample.pdf")))

    assert rows == [
        {
            "doc_id": "sample",
            "page_num": 1,
            "text": "Recovered scanned text",
            "method": "vision_llm",
        }
    ]


def test_extract_single_pdf_skips_vision_when_pdfplumber_has_text(monkeypatch) -> None:
    monkeypatch.setattr(
        extract_mod.pdfplumber,
        "open",
        lambda *_args, **_kwargs: _FakePdf([object()]),
    )
    monkeypatch.setattr(extract_mod, "_extract_page_pdfplumber", lambda _page: "Already extracted text")
    monkeypatch.setattr(extract_mod, "_extract_page_ocr", lambda _pdf_path, _page_num: "")

    def _unexpected_vision(*_args, **_kwargs):
        raise AssertionError("vision fallback should not run when pdfplumber already extracted text")

    monkeypatch.setattr(extract_mod, "_extract_page_vision", _unexpected_vision)

    rows = list(extract_mod.extract_single_pdf(Path("sample.pdf")))

    assert rows[0]["method"] == "pdfplumber"
    assert rows[0]["text"] == "Already extracted text"
