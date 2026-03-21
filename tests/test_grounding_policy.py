from src.retrieve.grounding_policy import detect_grounding_intent, score_chunk_for_intent


def test_detect_grounding_intent_for_date_of_issue() -> None:
    intent = detect_grounding_intent(
        "Which case has an earlier Date of Issue: CA 004/2025 or SCT 295/2025?",
        "name",
    )

    assert intent.kind == "date_of_issue"
    assert intent.page_focus == "front"
    assert intent.prefer_unique_docs is True


def test_detect_grounding_intent_for_last_page() -> None:
    intent = detect_grounding_intent(
        "According to the 'IT IS HEREBY ORDERED THAT' section on the last page, what was the outcome?",
        "free_text",
    )

    assert intent.kind == "last_page"
    assert intent.page_focus == "last"


def test_detect_grounding_intent_for_article_ref_law_question() -> None:
    intent = detect_grounding_intent(
        "According to Article 9(9)(a) of the Operating Law 2018, how many months does a Licence typically have effect from its issue date by the Registrar?",
        "number",
    )

    assert intent.kind == "article_ref"
    assert intent.article_refs == ("Article 9(9)(a)",)


def test_detect_grounding_intent_for_article_ref_ignores_generic_law_phrases() -> None:
    intent = detect_grounding_intent(
        "Under Article 8(1) of the Operating Law 2018, what happens to a licence under this Law if it was granted or continued under a Prescribed Law?",
        "boolean",
    )

    assert intent.kind == "article_ref"
    assert intent.article_refs == ("Article 8(1)",)


def test_score_chunk_for_intent_prefers_first_page_title_chunks() -> None:
    title_chunk = {
        "doc_id": "doc-a",
        "page_numbers": [1],
        "section_path": "Pages 1-1",
        "doc_title": "Title",
        "text": "Claimant Defendant caption",
    }
    body_chunk = {
        "doc_id": "doc-a",
        "page_numbers": [4],
        "section_path": "Pages 4-4",
        "doc_title": "Title",
        "text": "Body text only",
    }
    intent = detect_grounding_intent("From the title page, identify the Claimant.", "names")

    assert score_chunk_for_intent(title_chunk, intent, doc_max_page=4) > score_chunk_for_intent(body_chunk, intent, doc_max_page=4)


def test_detect_grounding_intent_routes_title_page_multi_case_party_compare() -> None:
    intent = detect_grounding_intent(
        "From the title pages of all documents in case CA 005/2025 and case CFI 067/2025, identify whether any individual or company is named as a main party in both cases.",
        "boolean",
    )

    assert intent.kind == "party_compare"
    assert intent.page_focus == "first"
    assert intent.max_docs == 6
    assert intent.max_total_pages == 6


def test_detect_grounding_intent_allows_multi_doc_title_page_coverage() -> None:
    intent = detect_grounding_intent(
        "From the header/caption section of each document in case TCD 001/2024, identify all parties listed as Claimant.",
        "names",
    )

    assert intent.kind == "title_page"
    assert intent.max_docs == 3
    assert intent.max_total_pages == 3


def test_detect_grounding_intent_detects_judge_compare_when_question_says_in_common() -> None:
    intent = detect_grounding_intent(
        "Did cases CA 004/2025 and ARB 034/2025 have any judges in common?",
        "boolean",
    )

    assert intent.kind == "judge_compare"
    assert intent.max_docs == 4
    assert intent.max_total_pages == 4


def test_detect_grounding_intent_detects_party_compare_when_question_says_to_both() -> None:
    intent = detect_grounding_intent(
        "Identify whether any person or company is a main party to both ENF 269/2023 and SCT 514/2025.",
        "boolean",
    )

    assert intent.kind == "party_compare"
    assert intent.max_docs == 4
    assert intent.max_total_pages == 4
