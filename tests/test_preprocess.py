from __future__ import annotations

import numpy as np

from src.preprocess import chunk as chunk_mod
from src.preprocess.build_index import _encode_with_backoff


def test_split_text_to_fit_handles_single_oversized_paragraph() -> None:
    text = " ".join(f"word{i}" for i in range(300))

    parts = chunk_mod._split_text_to_fit(text, 50)

    assert len(parts) > 1
    assert all(chunk_mod._approx_tokens(part) <= 50 for part in parts)


def test_chunk_document_splits_oversized_single_page(monkeypatch) -> None:
    monkeypatch.setattr(chunk_mod, "MAX_CHUNK_TOKENS", 50)
    monkeypatch.setattr(chunk_mod, "MIN_CHUNK_TOKENS", 10)
    huge_page = " ".join(f"word{i}" for i in range(300))

    chunks = chunk_mod.chunk_document(
        "doc",
        [{"page_num": 1, "text": huge_page}],
    )

    assert len(chunks) > 1
    assert all(chunk_mod._approx_tokens(chunk["text"]) <= 50 for chunk in chunks)
    assert all(chunk["page_numbers"] == [1] for chunk in chunks)


def test_chunk_document_maps_split_preamble_to_narrower_pages(monkeypatch) -> None:
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


class _FakeModel:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def encode(self, texts, batch_size: int, **kwargs):
        self.calls.append(batch_size)
        if batch_size > 2:
            raise RuntimeError("DefaultCPUAllocator: not enough memory")
        return np.ones((len(texts), 3), dtype=np.float32)


def test_encode_with_backoff_reduces_batch_size() -> None:
    model = _FakeModel()

    embeddings, used_batch_size = _encode_with_backoff(model, ["a", "b", "c", "d"], 8)

    assert used_batch_size == 2
    assert model.calls == [4, 2]
    assert embeddings.shape == (4, 3)
    assert embeddings.dtype == np.float32