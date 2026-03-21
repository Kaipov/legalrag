from __future__ import annotations

from src.preprocess.index_enrichment import (
    build_doc_level_metadata,
    build_embedding_input,
    enrich_record_for_indexing,
)
from src.retrieve.lexical import build_bm25_document_tokens


def test_enrich_record_propagates_doc_level_case_ids() -> None:
    doc_metadata = build_doc_level_metadata(
        [
            {
                "doc_id": "doc-1",
                "case_ids": ["CFI 057/2025"],
                "claim_numbers": ["CFI-057-2025"],
            }
        ]
    )

    enriched = enrich_record_for_indexing(
        {
            "doc_id": "doc-1",
            "doc_title": "ORDER WITH REASONS",
            "section_path": "Reasons",
            "text": "The court held that the application should be dismissed.",
        },
        doc_metadata["doc-1"],
    )

    assert enriched["case_ids"] == ["CFI 057/2025"]
    assert enriched["claim_numbers"] == ["CFI-057-2025"]
    assert enriched["doc_case_ids"] == ["CFI 057/2025"]


def test_build_embedding_input_includes_structured_metadata() -> None:
    embedding_input = build_embedding_input(
        {
            "page_num": 1,
            "is_first_page": True,
            "case_ids": ["CFI 057/2025"],
            "issue_date": "2025-01-15",
            "judges": ["Justice Jane Doe"],
            "parties": ["Claimant: Alpha LLC"],
            "text": "Application for permission to appeal.",
        }
    )

    assert "Page role: first page" in embedding_input
    assert "Case IDs: CFI 057/2025" in embedding_input
    assert "Issue date: 2025-01-15" in embedding_input
    assert "Judges: Justice Jane Doe" in embedding_input
    assert "Passage:" in embedding_input


def test_build_bm25_document_tokens_uses_metadata_fields() -> None:
    tokens = build_bm25_document_tokens(
        {
            "doc_title": "ORDER WITH REASONS",
            "section_path": "Reasons",
            "text": "The application should be dismissed.",
            "case_ids": ["CFI 057/2025"],
            "article_refs": ["Article 7(1)"],
        }
    )

    assert "cfi057" in tokens
    assert "article7" in tokens
