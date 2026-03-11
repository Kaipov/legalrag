from __future__ import annotations

import json

from scripts.compare_submissions import compare_submissions


def test_compare_submissions_reports_totals_and_intent_buckets(tmp_path, capsys) -> None:
    questions = [
        {
            "id": "q-date",
            "question": "Which case has an earlier Date of Issue: CA 004/2025 or SCT 295/2025?",
            "answer_type": "name",
        },
        {
            "id": "q-title",
            "question": "From the title page of the document, what is its official DIFC Law number?",
            "answer_type": "number",
        },
    ]
    baseline = {
        "answers": [
            {
                "question_id": "q-date",
                "telemetry": {
                    "retrieval": {
                        "retrieved_chunk_pages": [
                            {"doc_id": "doc-a", "page_numbers": [1, 2, 3]},
                            {"doc_id": "doc-b", "page_numbers": [1, 2, 3]},
                        ]
                    }
                },
            },
            {
                "question_id": "q-title",
                "telemetry": {
                    "retrieval": {
                        "retrieved_chunk_pages": [
                            {"doc_id": "doc-c", "page_numbers": [1, 2]},
                        ]
                    }
                },
            },
        ]
    }
    candidate = {
        "answers": [
            {
                "question_id": "q-date",
                "telemetry": {
                    "retrieval": {
                        "retrieved_chunk_pages": [
                            {"doc_id": "doc-a", "page_numbers": [2]},
                            {"doc_id": "doc-b", "page_numbers": [2]},
                        ]
                    }
                },
            },
            {
                "question_id": "q-title",
                "telemetry": {
                    "retrieval": {
                        "retrieved_chunk_pages": [
                            {"doc_id": "doc-c", "page_numbers": [1]},
                        ]
                    }
                },
            },
        ]
    }

    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    questions_path = tmp_path / "questions.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    questions_path.write_text(json.dumps(questions), encoding="utf-8")

    exit_code = compare_submissions(baseline_path, candidate_path, questions_path)
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Questions compared: 2" in output
    assert "Baseline total pages:  8" in output
    assert "Candidate total pages: 3" in output
    assert "date_of_issue" in output
    assert "title_page" in output
