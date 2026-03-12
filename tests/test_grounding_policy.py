from src.retrieve.grounding_policy import detect_grounding_intent, score_chunk_for_intent


def test_detect_grounding_intent_for_date_of_issue() -> None:
    intent = detect_grounding_intent(
        "Which case has an earlier Date of Issue: CA 004/2025 or SCT 295/2025?",
        "name",
    )

    assert intent.kind == "date_of_issue"
    assert intent.page_focus == "first"
    assert intent.prefer_unique_docs is True


def test_detect_grounding_intent_for_last_page() -> None:
    intent = detect_grounding_intent(
        "According to the 'IT IS HEREBY ORDERED THAT' section on the last page, what was the outcome?",
        "free_text",
    )

    assert intent.kind == "last_page"
    assert intent.page_focus == "last"


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