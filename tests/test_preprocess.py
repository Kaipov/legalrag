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

