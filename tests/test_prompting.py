from __future__ import annotations

from scripts import run as run_mod
from src.generate.prompts import build_prompt


def test_build_prompt_limits_number_of_sources() -> None:
    chunks = [
        ({"doc_id": f"doc-{i}", "page_numbers": [i + 1], "section_path": f"Section {i}", "doc_title": f"Title {i}", "text": f"Text {i}"}, 1.0)
        for i in range(4)
    ]

    messages = build_prompt("What happened?", "free_text", chunks, max_chunks=2)
    user_content = messages[1]["content"]

    assert "[Source 1:" in user_content
    assert "[Source 2:" in user_content
    assert "[Source 3:" not in user_content


def test_build_architecture_summary_reflects_reranker_setting(monkeypatch) -> None:
    monkeypatch.setattr(run_mod, "GENERATION_TOP_K", 4)

    monkeypatch.setattr(run_mod, "ENABLE_RERANKER", False)
    summary_without_reranker = run_mod.build_architecture_summary()
    assert "reranker disabled" in summary_without_reranker
    assert "top-4 chunks" in summary_without_reranker

    monkeypatch.setattr(run_mod, "ENABLE_RERANKER", True)
    summary_with_reranker = run_mod.build_architecture_summary()
    assert "cross-encoder rerank" in summary_with_reranker
