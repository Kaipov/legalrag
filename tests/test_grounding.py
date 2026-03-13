from src.retrieve import grounding as grounding_mod
from src.retrieve.grounding_policy import GroundingIntent



def test_collect_grounding_pages_merges_pages_by_document():
    reranked_chunks = [
        ({"doc_id": "doc-a", "page_numbers": [1, 2], "text": "alpha beta"}, 0.8),
        ({"doc_id": "doc-a", "page_numbers": [2, 3], "text": "beta gamma"}, 0.7),
        ({"doc_id": "doc-b", "page_numbers": [5], "text": "other"}, -2.0),
    ]

    assert grounding_mod.collect_grounding_pages(reranked_chunks, score_threshold=0.0) == [
        {"doc_id": "doc-a", "page_numbers": [1, 2]},
    ]



def test_collect_grounding_pages_returns_empty_for_null_answers():
    reranked_chunks = [({"doc_id": "doc-a", "page_numbers": [1], "text": "alpha"}, 0.9)]

    assert grounding_mod.collect_grounding_pages(reranked_chunks, is_null=True) == []



def test_collect_grounding_pages_prunes_wide_chunks_to_best_pages():
    reranked_chunks = [
        (
            {
                "doc_id": "doc-a",
                "page_numbers": [1, 2, 3, 4],
                "text": "alpha beta gamma delta clause",
            },
            0.8,
        )
    ]
    page_texts_by_doc = {
        "doc-a": {
            1: "unrelated text",
            2: "alpha beta gamma delta clause and supporting wording",
            3: "alpha beta clause with nearby support",
            4: "other appendix language",
        }
    }

    assert grounding_mod.collect_grounding_pages(
        reranked_chunks,
        question_text="What does the alpha clause provide?",
        answer_text="The alpha clause applies here.",
        page_texts_by_doc=page_texts_by_doc,
    ) == [{"doc_id": "doc-a", "page_numbers": [2, 3]}]



def test_collect_grounding_pages_prefers_first_pages_for_title_page_intent():
    reranked_chunks = [
        ({"doc_id": "doc-a", "page_numbers": [1, 2, 3], "text": "header and caption text"}, 0.8)
    ]
    page_texts_by_doc = {
        "doc-a": {
            1: "claimant defendant case no date of issue",
            2: "body provisions and recital",
            3: "later body text",
        }
    }
    intent = GroundingIntent(kind="title_page", page_focus="first", keyphrases=("claimant", "defendant"), max_pages_per_chunk=1)

    assert grounding_mod.collect_grounding_pages(
        reranked_chunks,
        question_text="From the title page of the document, what is its official DIFC Law number?",
        page_texts_by_doc=page_texts_by_doc,
        intent=intent,
    ) == [{"doc_id": "doc-a", "page_numbers": [1]}]



def test_collect_grounding_pages_prefers_last_pages_for_last_page_intent():
    reranked_chunks = [
        ({"doc_id": "doc-a", "page_numbers": [4, 5, 6], "text": "conclusion and order"}, 0.8)
    ]
    page_texts_by_doc = {
        "doc-a": {
            1: "intro",
            2: "background",
            3: "analysis",
            4: "analysis continued",
            5: "conclusion text",
            6: "it is hereby ordered that the application is refused",
        }
    }
    intent = GroundingIntent(kind="last_page", page_focus="last", keyphrases=("it is hereby ordered that",), max_pages_per_chunk=1)

    assert grounding_mod.collect_grounding_pages(
        reranked_chunks,
        question_text="According to the 'IT IS HEREBY ORDERED THAT' section, what was the outcome?",
        page_texts_by_doc=page_texts_by_doc,
        intent=intent,
    ) == [{"doc_id": "doc-a", "page_numbers": [6]}]



def test_collect_grounding_pages_keeps_two_pages_for_last_page_outcome_when_needed():
    reranked_chunks = [
        ({"doc_id": "doc-a", "page_numbers": [5, 6, 7], "text": "ordered that application refused no order as to costs"}, 0.8)
    ]
    page_texts_by_doc = {
        "doc-a": {
            4: "analysis",
            5: "intro to conclusion",
            6: "it is hereby ordered that the application is refused",
            7: "the applicant shall bear its own costs of the application",
        }
    }
    intent = GroundingIntent(
        kind="last_page",
        page_focus="last",
        keyphrases=("it is hereby ordered that", "costs"),
        max_pages_per_chunk=2,
        max_pages_per_doc=2,
    )

    assert grounding_mod.collect_grounding_pages(
        reranked_chunks,
        question_text="According to the 'IT IS HEREBY ORDERED THAT' section, what was the outcome of the application and costs order?",
        answer_text="The application was refused and the Applicant had to bear its own costs.",
        page_texts_by_doc=page_texts_by_doc,
        intent=intent,
        answer_type="free_text",
    ) == [{"doc_id": "doc-a", "page_numbers": [6, 7]}]



def test_collect_grounding_pages_adds_page_retrieval_hits_with_doc_restriction(monkeypatch):
    reranked_chunks = [
        ({"doc_id": "doc-a", "page_numbers": [2], "text": "alpha clause"}, 0.8)
    ]
    page_texts_by_doc = {
        "doc-a": {
            1: "title page with alpha clause",
            2: "body page with alpha clause",
        },
        "doc-b": {
            1: "other document with alpha clause",
        },
    }

    class FakePageRetriever:
        def search(self, query: str, *, top_k: int = 6, allowed_doc_ids: set[str] | None = None):
            assert allowed_doc_ids == {"doc-a"}
            return [
                ({"doc_id": "doc-a", "page_num": 1}, 0.9),
                ({"doc_id": "doc-b", "page_num": 1}, 0.8),
            ]

    monkeypatch.setattr(grounding_mod, "_get_page_retriever", lambda: FakePageRetriever())

    assert grounding_mod.collect_grounding_pages(
        reranked_chunks,
        question_text="What does the alpha clause provide?",
        answer_text="Alpha clause answer.",
        page_texts_by_doc=page_texts_by_doc,
        allowed_doc_ids={"doc-a"},
    ) == [{"doc_id": "doc-a", "page_numbers": [1, 2]}]



def test_collect_grounding_pages_caps_generic_results_to_two_pages_per_doc(monkeypatch):
    reranked_chunks = [
        ({"doc_id": "doc-a", "page_numbers": [2], "text": "alpha answer"}, 0.8)
    ]
    page_texts_by_doc = {
        "doc-a": {
            1: "alpha answer exact support",
            2: "alpha answer exact support",
            3: "alpha answer support",
        }
    }

    class FakePageRetriever:
        def search(self, query: str, *, top_k: int = 6, allowed_doc_ids: set[str] | None = None):
            return [
                ({"doc_id": "doc-a", "page_num": 1}, 0.95),
                ({"doc_id": "doc-a", "page_num": 3}, 0.9),
            ]

    monkeypatch.setattr(grounding_mod, "_get_page_retriever", lambda: FakePageRetriever())

    assert grounding_mod.collect_grounding_pages(
        reranked_chunks,
        question_text="What does alpha answer provide?",
        answer_text="Alpha answer.",
        page_texts_by_doc=page_texts_by_doc,
        allowed_doc_ids={"doc-a"},
    ) == [{"doc_id": "doc-a", "page_numbers": [1, 2]}]



def test_collect_grounding_pages_caps_page_local_results_to_one_page_per_doc(monkeypatch):
    reranked_chunks = [
        ({"doc_id": "doc-a", "page_numbers": [1, 2], "text": "claim no case no caption"}, 0.8)
    ]
    page_texts_by_doc = {
        "doc-a": {
            1: "claim no case no date of issue on title page",
            2: "body text with parties and analysis",
            3: "another later page",
        }
    }
    intent = GroundingIntent(kind="title_page", page_focus="first", keyphrases=("claim no", "case no"), max_pages_per_chunk=2, max_pages_per_doc=1)

    class FakePageRetriever:
        def search(self, query: str, *, top_k: int = 6, allowed_doc_ids: set[str] | None = None):
            return [
                ({"doc_id": "doc-a", "page_num": 3}, 0.9),
            ]

    monkeypatch.setattr(grounding_mod, "_get_page_retriever", lambda: FakePageRetriever())

    assert grounding_mod.collect_grounding_pages(
        reranked_chunks,
        question_text="From the title page, what is the claim number?",
        answer_text="Claim number answer.",
        page_texts_by_doc=page_texts_by_doc,
        intent=intent,
        allowed_doc_ids={"doc-a"},
        answer_type="name",
    ) == [{"doc_id": "doc-a", "page_numbers": [1]}]



def test_collect_grounding_pages_prunes_compare_results_to_case_coverage(monkeypatch):
    reranked_chunks = [
        ({"doc_id": "doc-a", "page_numbers": [1], "text": "Justice Alice Example case CA 005/2025"}, 0.9),
        ({"doc_id": "doc-b", "page_numbers": [1], "text": "Justice Alice Example case TCD 001/2024"}, 0.88),
        ({"doc_id": "doc-c", "page_numbers": [1], "text": "Justice Alice Example unrelated case"}, 0.87),
    ]
    page_texts_by_doc = {
        "doc-a": {1: "Before: Justice Alice Example CA 005/2025"},
        "doc-b": {1: "Before: Justice Alice Example TCD 001/2024"},
        "doc-c": {1: "Before: Justice Alice Example unrelated case"},
    }
    intent = GroundingIntent(
        kind="judge_compare",
        page_focus="first",
        keyphrases=("before", "justice", "judge"),
        case_ids=("CA 005/2025", "TCD 001/2024"),
        max_pages_per_chunk=2,
        max_pages_per_doc=1,
    )

    class FakePageRetriever:
        def search(self, query: str, *, top_k: int = 6, allowed_doc_ids: set[str] | None = None):
            return [
                ({"doc_id": "doc-c", "page_num": 1}, 0.95),
            ]

    monkeypatch.setattr(grounding_mod, "_get_page_retriever", lambda: FakePageRetriever())

    assert grounding_mod.collect_grounding_pages(
        reranked_chunks,
        question_text="Based on all documents in case CA 005/2025 and case TCD 001/2024, did any judge appear in both cases?",
        answer_text="True",
        page_texts_by_doc=page_texts_by_doc,
        intent=intent,
        allowed_doc_ids={"doc-a", "doc-b", "doc-c"},
        answer_type="boolean",
    ) == [
        {"doc_id": "doc-a", "page_numbers": [1]},
        {"doc_id": "doc-b", "page_numbers": [1]},
    ]
