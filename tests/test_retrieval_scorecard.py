from __future__ import annotations

from scripts import retrieval_scorecard as retrieval_scorecard_mod


def test_build_retrieval_row_tracks_page_and_doc_hits() -> None:
    question = {
        "id": "q-1",
        "question": "What was the outcome of the order?",
        "answer_type": "free_text",
    }
    retrieved_chunks = [
        ({"doc_id": "doc-x", "page_numbers": [1]}, 0.9),
        ({"doc_id": "doc-a", "page_numbers": [2]}, 0.8),
        ({"doc_id": "doc-a", "page_numbers": [3]}, 0.7),
        ({"doc_id": "doc-b", "page_numbers": [5]}, 0.6),
    ]

    row = retrieval_scorecard_mod.build_retrieval_row(
        question,
        retrieved_chunks,
        {("doc-a", 2), ("doc-b", 5)},
        (1, 2, 4),
    )

    assert row["first_page_hit_rank"] == 2
    assert row["first_doc_hit_rank"] == 2
    assert row["first_doc_full_hit_rank"] == 4
    assert row["page_mrr"] == 0.5
    assert row["doc_mrr"] == 0.5
    assert row["page_hit_at"] == {1: False, 2: True, 4: True}
    assert row["doc_hit_at"] == {1: False, 2: True, 4: True}
    assert row["doc_full_hit_at"] == {1: False, 2: False, 4: True}
    assert row["page_recall_at"] == {1: 0.0, 2: 0.5, 4: 1.0}
    assert row["doc_coverage_at"] == {1: 0.0, 2: 0.5, 4: 1.0}


def test_summarize_retrieval_rows_aggregates_core_metrics() -> None:
    rows = [
        {
            "first_page_hit_rank": 2,
            "first_doc_hit_rank": 2,
            "first_doc_full_hit_rank": 4,
            "page_mrr": 0.5,
            "doc_mrr": 0.5,
            "page_hit_at": {1: False, 5: True},
            "doc_hit_at": {1: False, 5: True},
            "doc_full_hit_at": {1: False, 5: True},
            "page_recall_at": {1: 0.0, 5: 1.0},
            "doc_coverage_at": {1: 0.0, 5: 1.0},
        },
        {
            "first_page_hit_rank": None,
            "first_doc_hit_rank": None,
            "first_doc_full_hit_rank": None,
            "page_mrr": 0.0,
            "doc_mrr": 0.0,
            "page_hit_at": {1: False, 5: False},
            "doc_hit_at": {1: False, 5: False},
            "doc_full_hit_at": {1: False, 5: False},
            "page_recall_at": {1: 0.0, 5: 0.0},
            "doc_coverage_at": {1: 0.0, 5: 0.0},
        },
    ]

    summary = retrieval_scorecard_mod.summarize_retrieval_rows(rows, (1, 5))

    assert summary["question_count"] == 2
    assert summary["page_mrr"] == 0.25
    assert summary["doc_mrr"] == 0.25
    assert summary["mean_first_page_hit_rank"] == 2
    assert summary["mean_first_doc_hit_rank"] == 2
    assert summary["mean_first_doc_full_hit_rank"] == 4
    assert summary["page_hit_at"] == {1: 0.0, 5: 0.5}
    assert summary["doc_hit_at"] == {1: 0.0, 5: 0.5}
    assert summary["doc_full_hit_at"] == {1: 0.0, 5: 0.5}
    assert summary["page_recall_at"] == {1: 0.0, 5: 0.5}
    assert summary["doc_coverage_at"] == {1: 0.0, 5: 0.5}
    assert summary["misses_at_max_k"] == 1


def test_parse_top_ks_sorts_and_deduplicates() -> None:
    assert retrieval_scorecard_mod.parse_top_ks("10,3,3,1") == (1, 3, 10)


def test_retrieval_scorecard_parser_defaults() -> None:
    parser = retrieval_scorecard_mod.build_parser()
    args = parser.parse_args([])

    assert args.gold == str(retrieval_scorecard_mod.DEFAULT_GOLD_PATH)
    assert args.questions == str(retrieval_scorecard_mod.DEFAULT_QUESTIONS_PATH)
    assert args.modes == ",".join(retrieval_scorecard_mod.DEFAULT_MODES)
    assert args.top_ks == ",".join(str(k) for k in retrieval_scorecard_mod.DEFAULT_TOP_KS)
