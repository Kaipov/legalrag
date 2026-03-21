from __future__ import annotations

from scripts import retrieval_bottleneck_report as bottleneck_mod


def test_build_stage_row_tracks_hits_for_each_stage() -> None:
    question = {
        "id": "q-1",
        "question": "What was the outcome of the order?",
        "answer_type": "free_text",
    }
    gold_pages = {("doc-a", 2), ("doc-b", 5)}
    stage_snapshots = {
        "rrf_candidates": {"size": 30, "page_refs": {("doc-a", 2), ("doc-b", 5)}, "doc_ids": {"doc-a", "doc-b"}},
        "intent_ranked": {"size": 30, "page_refs": {("doc-a", 2)}, "doc_ids": {"doc-a"}},
        "retriever_output": {"size": 10, "page_refs": {("doc-a", 2)}, "doc_ids": {"doc-a"}},
        "generation_chunks": {"size": 4, "page_refs": set(), "doc_ids": set()},
    }

    row = bottleneck_mod.build_stage_row(question, gold_pages, stage_snapshots)

    assert row["stages"]["rrf_candidates"]["page_hit"] is True
    assert row["stages"]["rrf_candidates"]["page_recall"] == 1.0
    assert row["stages"]["intent_ranked"]["page_recall"] == 0.5
    assert row["stages"]["intent_ranked"]["doc_full_hit"] is False
    assert row["stages"]["generation_chunks"]["page_hit"] is False
    assert row["stages"]["generation_chunks"]["doc_coverage"] == 0.0


def test_summarize_transition_reports_lost_hits() -> None:
    rows = [
        {
            "stages": {
                "a": {"size": 30, "page_hit": True, "page_recall": 1.0},
                "b": {"size": 10, "page_hit": False, "page_recall": 0.0},
            }
        },
        {
            "stages": {
                "a": {"size": 30, "page_hit": True, "page_recall": 0.5},
                "b": {"size": 10, "page_hit": True, "page_recall": 0.5},
            }
        },
        {
            "stages": {
                "a": {"size": 30, "page_hit": False, "page_recall": 0.0},
                "b": {"size": 10, "page_hit": False, "page_recall": 0.0},
            }
        },
    ]

    summary = bottleneck_mod.summarize_transition(rows, "a", "b")

    assert summary["prev_page_hit_count"] == 2
    assert summary["next_page_hit_count"] == 1
    assert summary["lost_page_hits"] == 1
    assert summary["retained_page_hit_rate"] == 0.5
    assert summary["mean_page_recall_delta"] == -0.3333
    assert summary["mean_size_delta"] == -20


def test_bottleneck_report_parser_defaults() -> None:
    parser = bottleneck_mod.build_parser()
    args = parser.parse_args([])

    assert args.gold == str(bottleneck_mod.DEFAULT_GOLD_PATH)
    assert args.questions == str(bottleneck_mod.DEFAULT_QUESTIONS_PATH)
    assert args.details_limit == bottleneck_mod.DEFAULT_DETAILS_LIMIT
