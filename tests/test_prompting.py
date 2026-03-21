from __future__ import annotations

from scripts import run as run_mod
from src.generate.prompts import build_prompt
from src.retrieve.grounding_policy import GroundingIntent


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


def test_build_prompt_includes_answer_aware_output_contract() -> None:
    chunks = [
        ({"doc_id": "doc-1", "page_numbers": [1], "section_path": "Section 1", "doc_title": "Title 1", "text": "Text 1"}, 1.0)
    ]

    messages = build_prompt("What happened?", "free_text", chunks)
    user_content = messages[1]["content"]

    assert "SOURCES:" in user_content
    assert "ANSWER:" in user_content
    assert "Use the minimum number of sources needed" in user_content


def test_build_prompt_name_instruction_demands_single_winning_option() -> None:
    chunks = [
        ({"doc_id": "doc-1", "page_numbers": [1], "section_path": "Section 1", "doc_title": "Title 1", "text": "Text 1"}, 1.0)
    ]

    messages = build_prompt(
        "Between ARB 034/2025 and SCT 295/2025, which was issued first?",
        "name",
        chunks,
    )
    user_content = messages[1]["content"]

    assert "return ONLY the winning option exactly as written in the question" in user_content
    assert "Do not return dates, amounts, both options, or a sentence" in user_content


def test_build_prompt_free_text_discourages_context_prefaces() -> None:
    chunks = [
        ({"doc_id": "doc-1", "page_numbers": [1], "section_path": "Section 1", "doc_title": "Title 1", "text": "Text 1"}, 1.0)
    ]

    messages = build_prompt("Who administers the Foundations Law?", "free_text", chunks)
    user_content = messages[1]["content"]

    assert "Do not begin with 'According to the context'" in user_content
    assert "standalone sentence" in user_content
    assert "not as a fragment like 'USD 1,500.' or 'The DIFCA.'" in user_content


def test_build_prompt_free_text_policy_discourages_extra_outcome_details() -> None:
    chunks = [
        ({"doc_id": "doc-1", "page_numbers": [1], "section_path": "Section 1", "doc_title": "Title 1", "text": "Text 1"}, 1.0)
    ]

    messages = build_prompt("What was the outcome of the application?", "free_text", chunks)
    user_content = messages[1]["content"]

    assert "For outcome or order questions, state the ruling first" in user_content
    assert "mention costs only if the question asks for costs" in user_content


def test_build_prompt_adds_page_local_instruction_for_date_of_issue() -> None:
    chunks = [
        ({"doc_id": "doc-1", "page_numbers": [1], "section_path": "Section 1", "doc_title": "Title 1", "text": "Text 1"}, 1.0)
    ]
    intent = GroundingIntent(kind="date_of_issue", page_focus="first")

    messages = build_prompt("Which case has an earlier Date of Issue?", "name", chunks, intent=intent)
    user_content = messages[1]["content"]

    assert "page-local question" in user_content
    assert "first-page title/header/date-of-issue evidence" in user_content


def test_build_architecture_summary_reflects_reranker_setting(monkeypatch) -> None:
    monkeypatch.setattr(run_mod, "GENERATION_TOP_K", 4)

    monkeypatch.setattr(run_mod, "ENABLE_RERANKER", False)
    summary_without_reranker = run_mod.build_architecture_summary()
    assert "reranker disabled" in summary_without_reranker
    assert "adaptive top-k chunks" in summary_without_reranker

    monkeypatch.setattr(run_mod, "ENABLE_RERANKER", True)
    monkeypatch.setattr(run_mod, "RERANKER_PROVIDER", "local")
    monkeypatch.setattr(run_mod, "RERANKER_ENABLED_INTENTS", ("all",))
    summary_with_local_reranker = run_mod.build_architecture_summary()
    assert "cross-encoder rerank" in summary_with_local_reranker

    monkeypatch.setattr(run_mod, "RERANKER_PROVIDER", "voyage")
    monkeypatch.setattr(run_mod, "RERANKER_ENABLED_INTENTS", ("article_ref",))
    summary_with_voyage_reranker = run_mod.build_architecture_summary()
    assert "Voyage rerank-2.5 API rerank" in summary_with_voyage_reranker
    assert "(article_ref)" in summary_with_voyage_reranker

def test_build_prompt_adds_outcome_question_rule() -> None:
    chunks = [
        ({"doc_id": "doc-1", "page_numbers": [1], "section_path": "Section 1", "doc_title": "Title 1", "text": "Text 1"}, 1.0)
    ]

    messages = build_prompt("What was the result of the application heard in case CFI 057/2025?", "free_text", chunks)
    user_content = messages[1]["content"]

    assert "OUTCOME QUESTION RULE" in user_content
    assert "Do not answer that the outcome or result is unspecified" in user_content


def test_build_prompt_adds_law_scope_rule_for_boolean_law_subject_questions() -> None:
    chunks = [
        ({"doc_id": "doc-1", "page_numbers": [1], "section_path": "Section 1", "doc_title": "Title 1", "text": "Text 1"}, 1.0)
    ]

    messages = build_prompt(
        "Does the DIFC law numbered DIFC Law No. 7 of 2018 deal with insolvency and preferential debts?",
        "boolean",
        chunks,
    )
    user_content = messages[1]["content"]

    assert "LAW SCOPE RULE" in user_content
    assert "Do not answer true based only on passing definitions" in user_content
