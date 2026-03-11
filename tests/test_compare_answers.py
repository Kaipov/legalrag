from scripts import compare_answers as compare_answers_mod


def test_summarize_answer_changes_detects_changed_answers_and_null_regressions() -> None:
    questions_by_id = {
        "q1": {"id": "q1", "question": "Question one", "answer_type": "name"},
        "q2": {"id": "q2", "question": "Question two", "answer_type": "free_text"},
        "q3": {"id": "q3", "question": "Question three", "answer_type": "boolean"},
    }
    baseline_answers = {
        "q1": {"question_id": "q1", "answer": "Alpha"},
        "q2": {"question_id": "q2", "answer": "Some answer"},
        "q3": {"question_id": "q3", "answer": False},
    }
    candidate_answers = {
        "q1": {"question_id": "q1", "answer": "Alpha"},
        "q2": {"question_id": "q2", "answer": None},
        "q3": {"question_id": "q3", "answer": True},
    }

    summary = compare_answers_mod.summarize_answer_changes(
        baseline_answers,
        candidate_answers,
        questions_by_id,
    )

    assert summary["questions_compared"] == 3
    assert summary["unchanged"] == 1
    assert summary["changed"] == 2
    assert summary["baseline_null_like"] == 0
    assert summary["candidate_null_like"] == 1
    assert summary["null_regressions"] == 1
    assert summary["by_type"] == {"boolean": 1, "free_text": 1}


def test_answer_signature_is_stable_for_lists() -> None:
    assert compare_answers_mod._answer_signature(["Alpha", "Beta"]) == '["Alpha", "Beta"]'
