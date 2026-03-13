from src.retrieve.grounding_policy import GroundingIntent
from src.retrieve.grounding_utils import (
    classify_article_page_match,
    extract_question_anchors,
    match_target_law_doc,
    score_grounding_page,
)


def test_extract_question_anchors_captures_article_law_and_quotes() -> None:
    anchors = extract_question_anchors(
        "According to 'IT IS HEREBY ORDERED THAT' and Article 9(9)(a) of the Operating Law 2018, what is the outcome in case CA 005/2025?"
    )

    assert anchors.article_refs == ("Article 9(9)(a)",)
    assert anchors.quoted_sections == ("IT IS HEREBY ORDERED THAT",)
    assert anchors.law_title_mentions == ("Operating Law",)
    assert anchors.case_ids == ("CA 005/2025",)


def test_extract_question_anchors_ignores_generic_this_and_prescribed_law_phrases() -> None:
    anchors = extract_question_anchors(
        "Under Article 8(1) of the Operating Law 2018, what happens to a licence under this Law if it was granted or continued under a Prescribed Law?"
    )

    assert anchors.article_refs == ("Article 8(1)",)
    assert anchors.law_title_mentions == ("Operating Law",)


def test_extract_question_anchors_keeps_single_employment_law_title() -> None:
    anchors = extract_question_anchors(
        "According to Article 10 of the Employment Law 2019, when must a claim be presented to the Court?"
    )

    assert anchors.article_refs == ("Article 10",)
    assert anchors.law_title_mentions == ("Employment Law",)


def test_match_target_law_doc_prefers_explicit_law_number() -> None:
    first_page_records = [
        {"doc_id": "doc-a", "page_num": 1, "doc_title": "OPERATING LAW", "text": "DIFC Law No. 7 of 2018"},
        {"doc_id": "doc-b", "page_num": 1, "doc_title": "EMPLOYMENT LAW", "text": "DIFC Law No. 4 of 2019"},
    ]

    assert (
        match_target_law_doc(
            "According to Article 9 of DIFC Law No. 7 of 2018, what is the effect period?",
            first_page_records,
        )
        == "doc-a"
    )


def test_match_target_law_doc_uses_unique_title_match() -> None:
    first_page_records = [
        {"doc_id": "doc-a", "page_num": 1, "doc_title": "OPERATING LAW", "text": "DIFC Law No. 7 of 2018"},
        {"doc_id": "doc-b", "page_num": 1, "doc_title": "EMPLOYMENT LAW", "text": "DIFC Law No. 4 of 2019"},
    ]

    assert (
        match_target_law_doc(
            "According to Article 10 of the Employment Law 2019, when must a claim be presented to the Court?",
            first_page_records,
        )
        == "doc-b"
    )


def test_match_target_law_doc_returns_none_for_ambiguous_title_match() -> None:
    first_page_records = [
        {"doc_id": "doc-a", "page_num": 1, "doc_title": "OPERATING LAW", "text": "DIFC Law No. 7 of 2018"},
        {"doc_id": "doc-b", "page_num": 1, "doc_title": "OPERATING LAW", "text": "DIFC Law No. 9 of 2018"},
    ]

    assert (
        match_target_law_doc(
            "According to Article 9 of the Operating Law, what is the effect period?",
            first_page_records,
        )
        is None
    )


def test_score_grounding_page_prefers_cited_page() -> None:
    anchors = extract_question_anchors("What does the alpha clause provide?")
    intent = GroundingIntent(kind="generic")
    cited_page = {"doc_id": "doc-a", "page_num": 2, "doc_title": "Title", "text": "alpha clause support", "article_refs": []}
    uncited_page = {"doc_id": "doc-a", "page_num": 3, "doc_title": "Title", "text": "alpha clause support", "article_refs": []}

    assert (
        score_grounding_page(cited_page, anchors, "Alpha clause support.", intent, {("doc-a", 2)})
        > score_grounding_page(uncited_page, anchors, "Alpha clause support.", intent, set())
    )


def test_score_grounding_page_prefers_exact_article_page_over_generic_topic_page() -> None:
    anchors = extract_question_anchors(
        "According to Article 9(9)(a) of the Operating Law 2018, how many months does a Licence have effect from its issue date?"
    )
    intent = GroundingIntent(kind="article_ref", article_refs=anchors.article_refs)
    exact_article_page = {
        "doc_id": "doc-a",
        "page_num": 9,
        "doc_title": "OPERATING LAW",
        "text": "Article 9(9)(a) A Licence shall have effect for 12 months from its issue date.",
        "article_refs": ["Article 9(9)(a)"],
    }
    topical_page = {
        "doc_id": "doc-a",
        "page_num": 2,
        "doc_title": "OPERATING LAW",
        "text": "A Licence shall have effect from its issue date by the Registrar.",
        "article_refs": [],
    }

    assert (
        score_grounding_page(exact_article_page, anchors, "12", intent, set(), target_law_doc_id="doc-a")
        > score_grounding_page(topical_page, anchors, "12", intent, set(), target_law_doc_id="doc-a")
    )


def test_classify_article_page_match_prefers_definition_page_over_cross_reference() -> None:
    anchors = extract_question_anchors(
        "Under Article 11(2)(b) of the Employment Law 2019, can an Employee waive any right under this Law?"
    )
    definition_page = {
        "doc_id": "doc-a",
        "page_num": 7,
        "doc_title": "EMPLOYMENT LAW",
        "text": (
            "EMPLOYMENT LAW\n"
            "11. No waiver\n"
            "(1) The requirements of this Law are minimum requirements.\n"
            "(2) Nothing in this Law precludes:\n"
            "(a) an Employer from providing more favourable terms; or\n"
            "(b) an Employee from waiving any right under this Law by entering into a written agreement."
        ),
        "article_refs": ["Article 66(13)"],
    }
    citation_page = {
        "doc_id": "doc-a",
        "page_num": 32,
        "doc_title": "EMPLOYMENT LAW",
        "text": "The provisions of Article 11(2)(b) and Article 66(11) shall apply retrospectively.",
        "article_refs": ["Article 11(2)(b)", "Article 66(11)"],
    }

    assert classify_article_page_match(definition_page, anchors, answer_text="true").kind == "definition"
    assert classify_article_page_match(citation_page, anchors, answer_text="true").kind == "citation"
    assert (
        score_grounding_page(definition_page, anchors, "true", GroundingIntent(kind="article_ref", article_refs=anchors.article_refs), set(), target_law_doc_id="doc-a")
        > score_grounding_page(citation_page, anchors, "true", GroundingIntent(kind="article_ref", article_refs=anchors.article_refs), set(), target_law_doc_id="doc-a")
    )


def test_classify_article_page_match_prefers_clause_definition_page_over_reference_page() -> None:
    anchors = extract_question_anchors(
        "According to Article 9(9)(a) of the Operating Law 2018, how many months does a Licence typically have effect from its issue date by the Registrar?"
    )
    continuation_page = {
        "doc_id": "doc-a",
        "page_num": 9,
        "doc_title": "OPERATING LAW",
        "text": (
            "OPERATING LAW\n"
            "(9) A Licence shall have effect for:\n"
            "(a) a period of twelve (12) months from the date of its issue by the Registrar; or\n"
            "(b) such other period as may be issued."
        ),
        "article_refs": ["Article 9(2)", "Article 9(10)"],
    }
    citation_page = {
        "doc_id": "doc-a",
        "page_num": 6,
        "doc_title": "OPERATING LAW",
        "text": "The Registrar may exercise powers under Article 9(9)(a) and Article 9(2) where appropriate.",
        "article_refs": ["Article 9(2)", "Article 9(9)(a)"],
    }

    assert classify_article_page_match(continuation_page, anchors, answer_text="12").kind == "definition"
    assert classify_article_page_match(citation_page, anchors, answer_text="12").kind == "citation"
    assert (
        score_grounding_page(continuation_page, anchors, "12", GroundingIntent(kind="article_ref", article_refs=anchors.article_refs), set(), target_law_doc_id="doc-a")
        > score_grounding_page(citation_page, anchors, "12", GroundingIntent(kind="article_ref", article_refs=anchors.article_refs), set(), target_law_doc_id="doc-a")
    )
