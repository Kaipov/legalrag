from __future__ import annotations

import sys
from types import ModuleType

from src.retrieve import rerank as rerank_mod


class _FakeCrossEncoder:
    last_instance = None

    def __init__(self, model_name, device=None, max_length=None, model_kwargs=None):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.model_kwargs = model_kwargs or {}
        self.predict_calls: list[dict] = []
        _FakeCrossEncoder.last_instance = self

    def predict(self, pairs, batch_size=32, show_progress_bar=None):
        self.predict_calls.append(
            {
                "pairs": pairs,
                "batch_size": batch_size,
                "show_progress_bar": show_progress_bar,
            }
        )
        return [0.2, 0.9, 0.5]


def test_cross_encoder_reranker_uses_gpu_friendly_defaults(monkeypatch) -> None:
    fake_module = ModuleType("sentence_transformers")
    fake_module.CrossEncoder = _FakeCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    monkeypatch.setattr(rerank_mod, "DEVICE", "cuda")
    monkeypatch.setattr(rerank_mod, "RERANKER_BATCH_SIZE", 4)
    monkeypatch.setattr(rerank_mod, "RERANKER_MAX_LENGTH", 512)
    monkeypatch.setattr(rerank_mod, "RERANKER_USE_FP16", True)

    reranker = rerank_mod.CrossEncoderReranker(model_name="test-reranker")

    assert reranker.batch_size == 4
    assert reranker.max_length == 512
    assert reranker.use_fp16 is True
    assert _FakeCrossEncoder.last_instance is not None
    assert _FakeCrossEncoder.last_instance.model_name == "test-reranker"
    assert _FakeCrossEncoder.last_instance.device == "cuda"
    assert _FakeCrossEncoder.last_instance.max_length == 512
    assert "torch_dtype" in _FakeCrossEncoder.last_instance.model_kwargs


def test_cross_encoder_reranker_predict_uses_configured_batch_size(monkeypatch) -> None:
    fake_module = ModuleType("sentence_transformers")
    fake_module.CrossEncoder = _FakeCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    monkeypatch.setattr(rerank_mod, "DEVICE", "cpu")
    monkeypatch.setattr(rerank_mod, "RERANKER_BATCH_SIZE", 4)
    monkeypatch.setattr(rerank_mod, "RERANKER_MAX_LENGTH", 512)
    monkeypatch.setattr(rerank_mod, "RERANKER_USE_FP16", False)

    reranker = rerank_mod.CrossEncoderReranker(model_name="test-reranker")
    chunks = [
        {"chunk_id": "c1", "text": "first"},
        {"chunk_id": "c2", "text": "second"},
        {"chunk_id": "c3", "text": "third"},
    ]

    ranked = reranker.rerank("query", chunks, top_k=2)

    assert [chunk["chunk_id"] for chunk, _score in ranked] == ["c2", "c3"]
    assert _FakeCrossEncoder.last_instance is not None
    assert _FakeCrossEncoder.last_instance.predict_calls == [
        {
            "pairs": [("query", "first"), ("query", "second"), ("query", "third")],
            "batch_size": 4,
            "show_progress_bar": False,
        }
    ]
