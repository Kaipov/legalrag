from src.retrieve.grounding import collect_grounding_pages


def test_collect_grounding_pages_merges_pages_by_document():
    reranked_chunks = [
        ({"doc_id": "doc-a", "page_numbers": [1, 2]}, 0.8),
        ({"doc_id": "doc-a", "page_numbers": [2, 3]}, 0.7),
        ({"doc_id": "doc-b", "page_numbers": [5]}, -2.0),
    ]

    assert collect_grounding_pages(reranked_chunks, score_threshold=0.0) == [
        {"doc_id": "doc-a", "page_numbers": [1, 2, 3]},
    ]


def test_collect_grounding_pages_returns_empty_for_null_answers():
    reranked_chunks = [({"doc_id": "doc-a", "page_numbers": [1]}, 0.9)]

    assert collect_grounding_pages(reranked_chunks, is_null=True) == []
