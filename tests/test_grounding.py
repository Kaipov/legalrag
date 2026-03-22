from src.retrieve import grounding as grounding_mod
from src.retrieve.grounding_policy import GroundingIntent



def test_collect_grounding_pages_compacts_generic_structured_pages_by_document():
    reranked_chunks = [
        ({"doc_id": "doc-a", "page_numbers": [1, 2], "text": "alpha beta"}, 0.8),
        ({"doc_id": "doc-a", "page_numbers": [2, 3], "text": "beta gamma"}, 0.7),
        ({"doc_id": "doc-b", "page_numbers": [5], "text": "other"}, -2.0),
    ]

    assert grounding_mod.collect_grounding_pages(reranked_chunks, score_threshold=0.0, answer_type="number") == [
        {"doc_id": "doc-a", "page_numbers": [1]},
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


def test_collect_grounding_pages_keeps_multi_doc_title_page_coverage():
    reranked_chunks = [
        ({"doc_id": "doc-a", "page_numbers": [1], "text": "Claimant Alpha LLC title page"}, 0.9),
        ({"doc_id": "doc-b", "page_numbers": [1], "text": "Claimant Alpha LLC title page"}, 0.88),
        ({"doc_id": "doc-c", "page_numbers": [1], "text": "Claimant Alpha LLC title page"}, 0.86),
    ]
    page_texts_by_doc = {
        "doc-a": {1: "Claimant Alpha LLC"},
        "doc-b": {1: "Claimant Alpha LLC"},
        "doc-c": {1: "Claimant Alpha LLC"},
    }
    intent = GroundingIntent(
        kind="title_page",
        page_focus="first",
        keyphrases=("claimant",),
        max_pages_per_doc=1,
        max_docs=3,
        max_total_pages=3,
    )

    assert grounding_mod.collect_grounding_pages(
        reranked_chunks,
        question_text="From the header/caption section of each document in case TCD 001/2024, identify all parties listed as Claimant.",
        answer_text="ARCHITERIORS INTERIOR DESIGN (L.L.C)",
        page_texts_by_doc=page_texts_by_doc,
        intent=intent,
        answer_type="names",
    ) == [
        {"doc_id": "doc-a", "page_numbers": [1]},
        {"doc_id": "doc-b", "page_numbers": [1]},
        {"doc_id": "doc-c", "page_numbers": [1]},
    ]


def test_collect_grounding_pages_generic_structured_keeps_single_doc_without_multi_doc_signal():
    reranked_chunks = [
        ({"doc_id": "doc-a", "page_numbers": [2], "text": "alpha answer", "__is_cited__": True}, 0.9),
        ({"doc_id": "doc-b", "page_numbers": [4], "text": "alpha answer appendix"}, 0.88),
    ]
    page_texts_by_doc = {
        "doc-a": {
            1: "title page",
            2: "alpha answer exact support",
        },
        "doc-b": {
            4: "alpha answer appendix support",
        },
    }
    page_records_by_doc = {
        "doc-a": {
            2: {"doc_id": "doc-a", "page_num": 2, "doc_title": "TITLE A", "text": "alpha answer exact support", "article_refs": [], "case_ids": []},
        },
        "doc-b": {
            4: {"doc_id": "doc-b", "page_num": 4, "doc_title": "TITLE B", "text": "alpha answer appendix support", "article_refs": [], "case_ids": []},
        },
    }

    assert grounding_mod.collect_grounding_pages(
        reranked_chunks,
        question_text="What is the alpha answer?",
        answer_text="Alpha answer.",
        page_texts_by_doc=page_texts_by_doc,
        page_records_by_doc=page_records_by_doc,
        answer_type="number",
    ) == [{"doc_id": "doc-a", "page_numbers": [2]}]


def test_collect_grounding_pages_generic_free_text_keeps_top_cited_page_and_caps_total_pages():
    reranked_chunks = [
        ({"doc_id": "doc-a", "page_numbers": [5], "text": "application dismissed", "__is_cited__": True}, 0.92),
        ({"doc_id": "doc-a", "page_numbers": [6], "text": "costs order"}, 0.9),
        ({"doc_id": "doc-b", "page_numbers": [2], "text": "background material"}, 0.85),
        ({"doc_id": "doc-c", "page_numbers": [3], "text": "appendix material"}, 0.8),
    ]
    page_texts_by_doc = {
        "doc-a": {
            5: "the application is dismissed",
            6: "there shall be no order as to costs",
        },
        "doc-b": {2: "background material"},
        "doc-c": {3: "appendix material"},
    }
    page_records_by_doc = {
        "doc-a": {
            5: {"doc_id": "doc-a", "page_num": 5, "doc_title": "TITLE A", "text": "the application is dismissed", "article_refs": [], "case_ids": []},
            6: {"doc_id": "doc-a", "page_num": 6, "doc_title": "TITLE A", "text": "there shall be no order as to costs", "article_refs": [], "case_ids": []},
        },
        "doc-b": {
            2: {"doc_id": "doc-b", "page_num": 2, "doc_title": "TITLE B", "text": "background material", "article_refs": [], "case_ids": []},
        },
        "doc-c": {
            3: {"doc_id": "doc-c", "page_num": 3, "doc_title": "TITLE C", "text": "appendix material", "article_refs": [], "case_ids": []},
        },
    }

    result = grounding_mod.collect_grounding_pages(
        reranked_chunks,
        question_text="What was the outcome of the application?",
        answer_text="The application was dismissed and there was no order as to costs.",
        page_texts_by_doc=page_texts_by_doc,
        page_records_by_doc=page_records_by_doc,
        answer_type="free_text",
    )

    assert sum(len(entry["page_numbers"]) for entry in result) <= 4
    assert any(entry["doc_id"] == "doc-a" and 5 in entry["page_numbers"] for entry in result)


def test_collect_grounding_pages_generic_structured_allows_three_pages_total_with_multi_doc_signal(monkeypatch):
    reranked_chunks = [
        ({"doc_id": "doc-a", "page_numbers": [2], "text": "alpha answer", "__is_cited__": True}, 0.92),
        ({"doc_id": "doc-b", "page_numbers": [4], "text": "alpha answer appendix", "__is_cited__": True}, 0.9),
    ]
    page_texts_by_doc = {
        "doc-a": {
            1: "alpha answer exact support",
            2: "alpha answer exact support",
        },
        "doc-b": {
            3: "alpha answer appendix support",
            4: "alpha answer appendix support",
        },
    }
    page_records_by_doc = {
        "doc-a": {
            1: {"doc_id": "doc-a", "page_num": 1, "doc_title": "TITLE A", "text": "alpha answer exact support", "article_refs": [], "case_ids": []},
            2: {"doc_id": "doc-a", "page_num": 2, "doc_title": "TITLE A", "text": "alpha answer exact support", "article_refs": [], "case_ids": []},
        },
        "doc-b": {
            3: {"doc_id": "doc-b", "page_num": 3, "doc_title": "TITLE B", "text": "alpha answer appendix support", "article_refs": [], "case_ids": []},
            4: {"doc_id": "doc-b", "page_num": 4, "doc_title": "TITLE B", "text": "alpha answer appendix support", "article_refs": [], "case_ids": []},
        },
    }

    class FakePageRetriever:
        def search(self, query: str, *, top_k: int = 6, allowed_doc_ids: set[str] | None = None):
            return [
                ({"doc_id": "doc-a", "page_num": 1}, 0.95),
            ]

    monkeypatch.setattr(grounding_mod, "_get_page_retriever", lambda: FakePageRetriever())

    result = grounding_mod.collect_grounding_pages(
        reranked_chunks,
        question_text="What is the alpha answer?",
        answer_text="Alpha answer.",
        page_texts_by_doc=page_texts_by_doc,
        page_records_by_doc=page_records_by_doc,
        answer_type="number",
        cited_page_keys={("doc-a", 2), ("doc-b", 4)},
    )

    assert result == [
        {"doc_id": "doc-b", "page_numbers": [4]},
        {"doc_id": "doc-a", "page_numbers": [1, 2]},
    ]


def test_collect_grounding_pages_article_ref_prefers_exact_page_for_explicit_law_number(monkeypatch):
    reranked_chunks = [
        ({"doc_id": "doc-a", "page_numbers": [2], "text": "licence effect period", "__is_cited__": True}, 0.9),
    ]
    page_texts_by_doc = {
        "doc-a": {
            1: "OPERATING LAW\nDIFC Law No. 7 of 2018",
            2: "A Licence has effect from issue date.",
            9: "Article 9(9)(a) A Licence shall have effect for 12 months from its issue date by the Registrar.",
        },
        "doc-b": {
            1: "EMPLOYMENT LAW\nDIFC Law No. 4 of 2019",
            9: "Article 9(9)(a) unrelated provision.",
        },
    }
    page_records_by_doc = {
        "doc-a": {
            1: {"doc_id": "doc-a", "page_num": 1, "doc_title": "OPERATING LAW", "text": page_texts_by_doc["doc-a"][1], "article_refs": [], "case_ids": []},
            2: {"doc_id": "doc-a", "page_num": 2, "doc_title": "OPERATING LAW", "text": page_texts_by_doc["doc-a"][2], "article_refs": [], "case_ids": []},
            9: {"doc_id": "doc-a", "page_num": 9, "doc_title": "OPERATING LAW", "text": page_texts_by_doc["doc-a"][9], "article_refs": ["Article 9(9)(a)"], "case_ids": []},
        },
        "doc-b": {
            1: {"doc_id": "doc-b", "page_num": 1, "doc_title": "EMPLOYMENT LAW", "text": page_texts_by_doc["doc-b"][1], "article_refs": [], "case_ids": []},
            9: {"doc_id": "doc-b", "page_num": 9, "doc_title": "EMPLOYMENT LAW", "text": page_texts_by_doc["doc-b"][9], "article_refs": ["Article 9(9)(a)"], "case_ids": []},
        },
    }
    first_page_records = [page_records_by_doc["doc-a"][1], page_records_by_doc["doc-b"][1]]

    class FakePageRetriever:
        def search(self, query: str, *, top_k: int = 6, allowed_doc_ids: set[str] | None = None):
            assert allowed_doc_ids == {"doc-a"}
            return [
                ({"doc_id": "doc-a", "page_num": 9}, 0.98),
                ({"doc_id": "doc-b", "page_num": 9}, 0.97),
            ]

    monkeypatch.setattr(grounding_mod, "_get_page_retriever", lambda: FakePageRetriever())

    intent = GroundingIntent(kind="article_ref", article_refs=("Article 9(9)(a)",), law_number="DIFC Law No. 7 of 2018")
    result = grounding_mod.collect_grounding_pages(
        reranked_chunks,
        question_text="According to Article 9(9)(a) of DIFC Law No. 7 of 2018, how many months does a Licence typically have effect from its issue date by the Registrar?",
        answer_text="12",
        page_texts_by_doc=page_texts_by_doc,
        page_records_by_doc=page_records_by_doc,
        first_page_records=first_page_records,
        intent=intent,
        allowed_doc_ids={"doc-a"},
        answer_type="number",
    )

    assert result == [{"doc_id": "doc-a", "page_numbers": [9]}]


def test_collect_grounding_pages_article_ref_uses_law_title_match(monkeypatch):
    reranked_chunks = [
        ({"doc_id": "doc-b", "page_numbers": [3], "text": "termination date claim", "__is_cited__": True}, 0.9),
    ]
    page_texts_by_doc = {
        "doc-a": {
            1: "OPERATING LAW\nDIFC Law No. 7 of 2018",
            10: "Article 10 unrelated provision.",
        },
        "doc-b": {
            1: "EMPLOYMENT LAW\nDIFC Law No. 4 of 2019",
            3: "termination date claim context",
            6: "Article 10 A claim under this Law must be presented within 6 months after the Termination Date.",
        },
    }
    page_records_by_doc = {
        "doc-a": {
            1: {"doc_id": "doc-a", "page_num": 1, "doc_title": "OPERATING LAW", "text": page_texts_by_doc["doc-a"][1], "article_refs": [], "case_ids": []},
            10: {"doc_id": "doc-a", "page_num": 10, "doc_title": "OPERATING LAW", "text": page_texts_by_doc["doc-a"][10], "article_refs": ["Article 10"], "case_ids": []},
        },
        "doc-b": {
            1: {"doc_id": "doc-b", "page_num": 1, "doc_title": "EMPLOYMENT LAW", "text": page_texts_by_doc["doc-b"][1], "article_refs": [], "case_ids": []},
            3: {"doc_id": "doc-b", "page_num": 3, "doc_title": "EMPLOYMENT LAW", "text": page_texts_by_doc["doc-b"][3], "article_refs": [], "case_ids": []},
            6: {"doc_id": "doc-b", "page_num": 6, "doc_title": "EMPLOYMENT LAW", "text": page_texts_by_doc["doc-b"][6], "article_refs": ["Article 10"], "case_ids": []},
        },
    }
    first_page_records = [page_records_by_doc["doc-a"][1], page_records_by_doc["doc-b"][1]]

    class FakePageRetriever:
        def search(self, query: str, *, top_k: int = 6, allowed_doc_ids: set[str] | None = None):
            return [
                ({"doc_id": "doc-b", "page_num": 6}, 0.99),
                ({"doc_id": "doc-a", "page_num": 10}, 0.97),
            ]

    monkeypatch.setattr(grounding_mod, "_get_page_retriever", lambda: FakePageRetriever())

    intent = GroundingIntent(kind="article_ref", article_refs=("Article 10",))
    result = grounding_mod.collect_grounding_pages(
        reranked_chunks,
        question_text="According to Article 10 of the Employment Law 2019, how many months after the Termination Date must a claim be presented to the Court?",
        answer_text="6",
        page_texts_by_doc=page_texts_by_doc,
        page_records_by_doc=page_records_by_doc,
        first_page_records=first_page_records,
        intent=intent,
        allowed_doc_ids={"doc-a", "doc-b"},
        answer_type="number",
    )

    assert result == [{"doc_id": "doc-b", "page_numbers": [6]}]


def test_collect_grounding_pages_article_ref_prefers_definition_page_over_cross_reference(monkeypatch):
    reranked_chunks = [
        ({"doc_id": "doc-a", "page_numbers": [7, 32], "text": "No waiver waiver written agreement", "__is_cited__": True}, 0.9),
    ]
    page_texts_by_doc = {
        "doc-a": {
            1: "EMPLOYMENT LAW\nDIFC Law No. 4 of 2019",
            7: (
                "EMPLOYMENT LAW\n"
                "11. No waiver\n"
                "(1) The requirements of this Law are minimum requirements.\n"
                "(2) Nothing in this Law precludes:\n"
                "(a) an Employer from providing more favourable terms; or\n"
                "(b) an Employee from waiving any right under this Law by entering into a written agreement."
            ),
            32: "EMPLOYMENT LAW\nThe provisions of Article 11(2)(b) and Article 66(11) shall apply retrospectively.",
        },
    }
    page_records_by_doc = {
        "doc-a": {
            1: {"doc_id": "doc-a", "page_num": 1, "doc_title": "EMPLOYMENT LAW", "text": page_texts_by_doc["doc-a"][1], "article_refs": [], "case_ids": []},
            7: {"doc_id": "doc-a", "page_num": 7, "doc_title": "EMPLOYMENT LAW", "text": page_texts_by_doc["doc-a"][7], "article_refs": ["Article 66(13)"], "case_ids": []},
            32: {"doc_id": "doc-a", "page_num": 32, "doc_title": "EMPLOYMENT LAW", "text": page_texts_by_doc["doc-a"][32], "article_refs": ["Article 11(2)(b)", "Article 66(11)"], "case_ids": []},
        },
    }
    first_page_records = [page_records_by_doc["doc-a"][1]]

    class FakePageRetriever:
        def search(self, query: str, *, top_k: int = 6, allowed_doc_ids: set[str] | None = None):
            return [
                ({"doc_id": "doc-a", "page_num": 32}, 0.99),
                ({"doc_id": "doc-a", "page_num": 7}, 0.98),
            ]

    monkeypatch.setattr(grounding_mod, "_get_page_retriever", lambda: FakePageRetriever())

    intent = GroundingIntent(kind="article_ref", article_refs=("Article 11(2)(b)",))
    result = grounding_mod.collect_grounding_pages(
        reranked_chunks,
        question_text="Under Article 11(2)(b) of the Employment Law 2019, can an Employee waive any right under this Law?",
        answer_text="True",
        page_texts_by_doc=page_texts_by_doc,
        page_records_by_doc=page_records_by_doc,
        first_page_records=first_page_records,
        intent=intent,
        allowed_doc_ids={"doc-a"},
        answer_type="boolean",
    )

    assert result == [{"doc_id": "doc-a", "page_numbers": [7]}]


def test_collect_grounding_pages_article_ref_prefers_definition_clause_page_over_reference(monkeypatch):
    reranked_chunks = [
        ({"doc_id": "doc-a", "page_numbers": [6, 9], "text": "Licence issue date Registrar months", "__is_cited__": True}, 0.9),
    ]
    page_texts_by_doc = {
        "doc-a": {
            1: "OPERATING LAW\nDIFC Law No. 7 of 2018",
            6: "OPERATING LAW\nThe Registrar may exercise powers under Article 9(9)(a) and Article 9(2) where appropriate.",
            9: (
                "OPERATING LAW\n"
                "(9) A Licence shall have effect for:\n"
                "(a) a period of twelve (12) months from the date of its issue by the Registrar; or\n"
                "(b) such other period as may be issued."
            ),
        },
    }
    page_records_by_doc = {
        "doc-a": {
            1: {"doc_id": "doc-a", "page_num": 1, "doc_title": "OPERATING LAW", "text": page_texts_by_doc["doc-a"][1], "article_refs": [], "case_ids": []},
            6: {"doc_id": "doc-a", "page_num": 6, "doc_title": "OPERATING LAW", "text": page_texts_by_doc["doc-a"][6], "article_refs": ["Article 9(9)(a)", "Article 9(2)"], "case_ids": []},
            9: {"doc_id": "doc-a", "page_num": 9, "doc_title": "OPERATING LAW", "text": page_texts_by_doc["doc-a"][9], "article_refs": ["Article 9(2)", "Article 9(10)"], "case_ids": []},
        },
    }
    first_page_records = [page_records_by_doc["doc-a"][1]]

    class FakePageRetriever:
        def search(self, query: str, *, top_k: int = 6, allowed_doc_ids: set[str] | None = None):
            return [
                ({"doc_id": "doc-a", "page_num": 6}, 0.99),
                ({"doc_id": "doc-a", "page_num": 9}, 0.98),
            ]

    monkeypatch.setattr(grounding_mod, "_get_page_retriever", lambda: FakePageRetriever())

    intent = GroundingIntent(kind="article_ref", article_refs=("Article 9(9)(a)",))
    result = grounding_mod.collect_grounding_pages(
        reranked_chunks,
        question_text="According to Article 9(9)(a) of the Operating Law 2018, how many months does a Licence typically have effect from its issue date by the Registrar?",
        answer_text="12",
        page_texts_by_doc=page_texts_by_doc,
        page_records_by_doc=page_records_by_doc,
        first_page_records=first_page_records,
        intent=intent,
        allowed_doc_ids={"doc-a"},
        answer_type="number",
    )

    assert result == [{"doc_id": "doc-a", "page_numbers": [9]}]


def test_collect_grounding_pages_article_ref_free_text_keeps_exact_and_context_pages(monkeypatch):
    reranked_chunks = [
        ({"doc_id": "doc-b", "page_numbers": [3], "text": "termination date claim", "__is_cited__": True}, 0.9),
    ]
    page_texts_by_doc = {
        "doc-b": {
            1: "EMPLOYMENT LAW\nDIFC Law No. 4 of 2019",
            3: "termination date claim context",
            6: "Article 10 A claim under this Law must be presented within 6 months after the Termination Date.",
        },
    }
    page_records_by_doc = {
        "doc-b": {
            1: {"doc_id": "doc-b", "page_num": 1, "doc_title": "EMPLOYMENT LAW", "text": page_texts_by_doc["doc-b"][1], "article_refs": [], "case_ids": []},
            3: {"doc_id": "doc-b", "page_num": 3, "doc_title": "EMPLOYMENT LAW", "text": page_texts_by_doc["doc-b"][3], "article_refs": [], "case_ids": []},
            6: {"doc_id": "doc-b", "page_num": 6, "doc_title": "EMPLOYMENT LAW", "text": page_texts_by_doc["doc-b"][6], "article_refs": ["Article 10"], "case_ids": []},
        },
    }
    first_page_records = [page_records_by_doc["doc-b"][1]]

    class FakePageRetriever:
        def search(self, query: str, *, top_k: int = 6, allowed_doc_ids: set[str] | None = None):
            return [
                ({"doc_id": "doc-b", "page_num": 6}, 0.99),
            ]

    monkeypatch.setattr(grounding_mod, "_get_page_retriever", lambda: FakePageRetriever())

    intent = GroundingIntent(kind="article_ref", article_refs=("Article 10",))
    result = grounding_mod.collect_grounding_pages(
        reranked_chunks,
        question_text="According to Article 10 of the Employment Law 2019, what does this Article require about when a claim must be presented?",
        answer_text="Article 10 requires that a claim be presented within 6 months after the Termination Date.",
        page_texts_by_doc=page_texts_by_doc,
        page_records_by_doc=page_records_by_doc,
        first_page_records=first_page_records,
        intent=intent,
        allowed_doc_ids={"doc-b"},
        answer_type="free_text",
    )

    assert result == [{"doc_id": "doc-b", "page_numbers": [3, 6]}]


def test_collect_grounding_pages_article_ref_falls_back_to_generic_when_target_is_ambiguous(monkeypatch):
    reranked_chunks = [
        ({"doc_id": "doc-a", "page_numbers": [2], "text": "article context", "__is_cited__": True}, 0.9),
    ]
    page_texts_by_doc = {
        "doc-a": {
            1: "OPERATING LAW\nDIFC Law No. 7 of 2018",
            2: "article context support",
            9: "Article 9 duty one.",
        },
        "doc-b": {
            1: "OPERATING LAW\nDIFC Law No. 9 of 2018",
            9: "Article 9 duty two.",
        },
    }
    page_records_by_doc = {
        "doc-a": {
            1: {"doc_id": "doc-a", "page_num": 1, "doc_title": "OPERATING LAW", "text": page_texts_by_doc["doc-a"][1], "article_refs": [], "case_ids": []},
            2: {"doc_id": "doc-a", "page_num": 2, "doc_title": "OPERATING LAW", "text": page_texts_by_doc["doc-a"][2], "article_refs": [], "case_ids": []},
            9: {"doc_id": "doc-a", "page_num": 9, "doc_title": "OPERATING LAW", "text": page_texts_by_doc["doc-a"][9], "article_refs": ["Article 9"], "case_ids": []},
        },
        "doc-b": {
            1: {"doc_id": "doc-b", "page_num": 1, "doc_title": "OPERATING LAW", "text": page_texts_by_doc["doc-b"][1], "article_refs": [], "case_ids": []},
            9: {"doc_id": "doc-b", "page_num": 9, "doc_title": "OPERATING LAW", "text": page_texts_by_doc["doc-b"][9], "article_refs": ["Article 9"], "case_ids": []},
        },
    }
    first_page_records = [page_records_by_doc["doc-a"][1], page_records_by_doc["doc-b"][1]]

    class FakePageRetriever:
        def search(self, query: str, *, top_k: int = 6, allowed_doc_ids: set[str] | None = None):
            return [
                ({"doc_id": "doc-b", "page_num": 9}, 0.99),
            ]

    monkeypatch.setattr(grounding_mod, "_get_page_retriever", lambda: FakePageRetriever())

    intent = GroundingIntent(kind="article_ref", article_refs=("Article 9",))
    result = grounding_mod.collect_grounding_pages(
        reranked_chunks,
        question_text="According to Article 9 of the Operating Law, what does the law provide?",
        answer_text="Article context support.",
        page_texts_by_doc=page_texts_by_doc,
        page_records_by_doc=page_records_by_doc,
        first_page_records=first_page_records,
        intent=intent,
        allowed_doc_ids={"doc-a"},
        answer_type="free_text",
    )

    assert result == [{"doc_id": "doc-a", "page_numbers": [2]}]
