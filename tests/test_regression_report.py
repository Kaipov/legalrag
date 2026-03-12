from __future__ import annotations

import json
from pathlib import Path

from scripts import regression_report as regression_report_mod


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_build_regression_report_summarizes_key_guardrail_metrics(tmp_path) -> None:
    questions = [
        {
            "id": "q-num",
            "question": "From the title page of the document, what is its official DIFC Law number?",
            "answer_type": "number",
        },
        {
            "id": "q-name",
            "question": "According to Article 16(1), what document must be filed at licence renewal?",
            "answer_type": "name",
        },
        {
            "id": "q-ft",
            "question": "What was the outcome of the order?",
            "answer_type": "free_text",
        },
        {
            "id": "q-null-ft",
            "question": "On what date was the Personal Property Law enacted?",
            "answer_type": "free_text",
        },
    ]
    baseline = {
        "answers": [
            {
                "question_id": "q-num",
                "answer": 7,
                "telemetry": {
                    "retrieval": {"retrieved_chunk_pages": [{"doc_id": "doc-a", "page_numbers": [1]}]}
                },
            },
            {
                "question_id": "q-name",
                "answer": "Confirmation Statement",
                "telemetry": {
                    "retrieval": {"retrieved_chunk_pages": [{"doc_id": "doc-b", "page_numbers": [2]}]}
                },
            },
            {
                "question_id": "q-ft",
                "answer": "The order was dismissed with costs.",
                "telemetry": {
                    "retrieval": {"retrieved_chunk_pages": [{"doc_id": "doc-c", "page_numbers": [3]}]}
                },
            },
            {
                "question_id": "q-null-ft",
                "answer": None,
                "telemetry": {"retrieval": {"retrieved_chunk_pages": []}},
            },
        ]
    }
    candidate = {
        "answers": [
            {
                "question_id": "q-num",
                "answer": 7,
                "telemetry": {
                    "retrieval": {
                        "retrieved_chunk_pages": [{"doc_id": "doc-a", "page_numbers": [1, 2]}]
                    }
                },
            },
            {
                "question_id": "q-name",
                "answer": None,
                "telemetry": {"retrieval": {"retrieved_chunk_pages": []}},
            },
            {
                "question_id": "q-ft",
                "answer": "The order was dismissed with costs.",
                "telemetry": {
                    "retrieval": {"retrieved_chunk_pages": [{"doc_id": "doc-c", "page_numbers": [3]}]}
                },
            },
            {
                "question_id": "q-null-ft",
                "answer": "The exact date is not provided in the context.",
                "telemetry": {
                    "retrieval": {"retrieved_chunk_pages": [{"doc_id": "doc-d", "page_numbers": [4]}]}
                },
            },
        ]
    }

    summary = regression_report_mod.build_regression_report(
        _write_json(tmp_path / "baseline.json", baseline),
        _write_json(tmp_path / "candidate.json", candidate),
        _write_json(tmp_path / "questions.json", questions),
    )

    assert summary["structured"]["question_count"] == 2
    assert summary["structured"]["score"] == 0.5
    assert summary["structured"]["mismatches"] == 1
    assert summary["structured"]["by_type"]["number"] == {"count": 1, "score": 1.0, "mismatches": 0}
    assert summary["structured"]["by_type"]["name"] == {"count": 1, "score": 0.0, "mismatches": 1}

    assert summary["grounding"]["question_count"] == 4
    assert summary["grounding"]["macro_fbeta"] == 0.4697
    assert summary["grounding"]["macro_jaccard"] == 0.375
    assert summary["grounding"]["exact_page_set_matches"] == 1
    assert summary["grounding"]["answer_exact_grounding_bad"] == 0

    assert summary["free_text"]["question_count"] == 2
    assert summary["free_text"]["gold_null_count"] == 1
    assert summary["free_text"]["gold_null_safe_null_like_count"] == 0
    assert summary["free_text"]["null_risk_count"] == 1
    assert summary["free_text"]["substantive_count"] == 1
    assert summary["free_text"]["substantive_strong"] == 1
    assert summary["free_text"]["substantive_mid"] == 0
    assert summary["free_text"]["substantive_weak"] == 0


def test_evaluate_guardrails_reports_threshold_violations() -> None:
    summary = {
        "structured": {"mismatches": 12},
        "grounding": {"macro_fbeta": 0.55, "answer_exact_grounding_bad": 20},
        "free_text": {"substantive_weak": 6, "null_risk_count": 2},
    }
    thresholds = {
        "min_grounding_fbeta": 0.57,
        "max_structured_mismatches": 10,
        "max_free_text_weak": 5,
        "max_free_text_null_risk": 1,
        "max_answer_exact_grounding_bad": 16,
    }

    violations = regression_report_mod.evaluate_guardrails(summary, thresholds)

    assert len(violations) == 5
    assert any("grounding macro F-beta" in violation for violation in violations)
    assert any("structured mismatches" in violation for violation in violations)
    assert any("free-text weak cases" in violation for violation in violations)
    assert any("free-text null-risk cases" in violation for violation in violations)
    assert any("answer-exact but grounding-bad cases" in violation for violation in violations)


def test_regression_report_parser_defaults_to_golden_and_submission() -> None:
    parser = regression_report_mod.build_parser()
    args = parser.parse_args([])

    assert args.baseline == str(regression_report_mod.DEFAULT_BASELINE_PATH)
    assert args.candidate == str(regression_report_mod.DEFAULT_CANDIDATE_PATH)
    assert args.questions == str(regression_report_mod.DEFAULT_QUESTIONS_PATH)
    assert args.min_grounding_fbeta == regression_report_mod.DEFAULT_GUARDRAILS["min_grounding_fbeta"]
