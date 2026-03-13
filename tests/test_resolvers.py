from __future__ import annotations

import json

from src.resolve.compare import resolve_date_of_issue_compare, resolve_judge_compare, resolve_party_compare
from src.resolve.metadata_store import PageMetadataStore
from src.resolve.outcome import resolve_last_page_outcome
from src.resolve.page_local import resolve_page_local_lookup
from src.retrieve.question_plan import QuestionPlan



def _write_metadata(tmp_path, rows: list[dict]) -> PageMetadataStore:
    path = tmp_path / "page_metadata.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return PageMetadataStore(path)



def test_resolve_date_of_issue_compare_returns_earlier_case(tmp_path) -> None:
    store = _write_metadata(
        tmp_path,
        [
            {"doc_id": "doc-a", "page_num": 2, "case_ids": ["CA 004/2025"], "issue_date": "2025-01-11", "judges": [], "parties": [], "claim_numbers": [], "is_first_page": False, "is_last_page": False},
            {"doc_id": "doc-b", "page_num": 2, "case_ids": ["SCT 295/2025"], "issue_date": "2025-02-14", "judges": [], "parties": [], "claim_numbers": [], "is_first_page": False, "is_last_page": False},
        ],
    )
    plan = QuestionPlan(
        mode="date_of_issue_compare",
        answer_type="name",
        case_ids=("CA 004/2025", "SCT 295/2025"),
        page_hint="page_2",
        compare_op="min_date",
        target_field="issue_date",
    )

    resolution = resolve_date_of_issue_compare(plan, store)

    assert resolution is not None
    assert resolution.answer == "CA 004/2025"
    assert [(page.doc_id, page.page_num) for page in resolution.evidence_pages] == [("doc-a", 2), ("doc-b", 2)]



def test_resolve_page_local_lookup_returns_claim_number(tmp_path) -> None:
    store = _write_metadata(
        tmp_path,
        [
            {"doc_id": "doc-a", "page_num": 2, "case_ids": ["CA 009/2024"], "issue_date": None, "judges": [], "parties": [], "claim_numbers": ["ENF-316-2023/2"], "is_first_page": False, "is_last_page": False, "doc_title": "Example", "text": "Claim No. ENF-316-2023/2"},
        ],
    )
    plan = QuestionPlan(
        mode="page_local_lookup",
        answer_type="name",
        case_ids=("CA 009/2024",),
        page_hint="page_2",
        target_field="claim_number",
    )

    resolution = resolve_page_local_lookup(plan, store, question_text="What claim number did the appeal originate from?")

    assert resolution is not None
    assert resolution.answer == "ENF-316-2023/2"
    assert [(page.doc_id, page.page_num) for page in resolution.evidence_pages] == [("doc-a", 2)]



def test_resolve_title_page_parties_returns_entity_names_only(tmp_path) -> None:
    store = _write_metadata(
        tmp_path,
        [
            {
                "doc_id": "doc-a",
                "page_num": 1,
                "case_ids": ["TCD 001/2024"],
                "issue_date": None,
                "judges": ["Chief Justice Wayne Martin"],
                "parties": ["Claimant: ARCHITERIORS INTERIOR DESIGN (L.L.C)", "Defendant: EMIRATES NATIONAL INVESTMENT CO (L.L.C)"],
                "claim_numbers": ["TCD-001/2024"],
                "is_first_page": True,
                "is_last_page": False,
                "doc_title": "Example",
                "text": "BETWEEN ARCHITERIORS INTERIOR DESIGN (L.L.C) Claimant and EMIRATES NATIONAL INVESTMENT CO (L.L.C) Defendant",
            },
        ],
    )
    plan = QuestionPlan(
        mode="title_page_metadata",
        answer_type="names",
        case_ids=("TCD 001/2024",),
        page_hint="first",
        target_field="party",
    )

    resolution = resolve_page_local_lookup(
        plan,
        store,
        question_text="From the header/caption section of each document in case TCD 001/2024, identify all parties listed as Claimant.",
    )

    assert resolution is not None
    assert resolution.answer == ["ARCHITERIORS INTERIOR DESIGN (L.L.C)"]



def test_resolve_last_page_outcome_returns_order_clauses_across_pages(tmp_path) -> None:
    store = _write_metadata(
        tmp_path,
        [
            {
                "doc_id": "doc-a",
                "page_num": 1,
                "case_ids": ["SCT 454/2024"],
                "issue_date": None,
                "judges": [],
                "parties": [],
                "claim_numbers": [],
                "is_first_page": True,
                "is_last_page": False,
                "text": "IT IS HEREBY ORDERED THAT:\n1. The request for an oral hearing of the Application is refused.\n2. The Application is refused.",
            },
            {
                "doc_id": "doc-a",
                "page_num": 2,
                "case_ids": ["SCT 454/2024"],
                "issue_date": None,
                "judges": [],
                "parties": [],
                "claim_numbers": [],
                "is_first_page": False,
                "is_last_page": False,
                "text": "3. The Applicant may not request that the decision be reconsidered at a hearing.\n4. The Applicant shall bear its own costs of the Application.\nIssued by:\nRegistrar",
            },
            {
                "doc_id": "doc-a",
                "page_num": 3,
                "case_ids": ["SCT 454/2024"],
                "issue_date": None,
                "judges": [],
                "parties": [],
                "claim_numbers": [],
                "is_first_page": False,
                "is_last_page": True,
                "text": "Further reasons.",
            },
        ],
    )
    plan = QuestionPlan(
        mode="last_page_outcome",
        answer_type="free_text",
        case_ids=("SCT 454/2024",),
        page_hint="last",
        target_field="outcome",
    )

    resolution = resolve_last_page_outcome(
        plan,
        store,
        question_text="According to the 'IT IS HEREBY ORDERED THAT' section of case SCT 454/2024, what was the outcome of the application for permission to appeal?",
    )

    assert resolution is not None
    assert resolution.answer == (
        "The request for an oral hearing was refused, the application was refused, "
        "reconsideration at a hearing was barred, and the Applicant had to bear its own costs."
    )
    assert [(page.doc_id, page.page_num) for page in resolution.evidence_pages] == [("doc-a", 1), ("doc-a", 2)]



def test_resolve_judge_compare_returns_boolean_overlap(tmp_path) -> None:
    store = _write_metadata(
        tmp_path,
        [
            {"doc_id": "doc-a", "page_num": 1, "case_ids": ["CA 005/2025"], "issue_date": None, "judges": ["Chief Justice Wayne Martin", "Justice Rene Le Miere"], "parties": [], "claim_numbers": [], "is_first_page": True, "is_last_page": False},
            {"doc_id": "doc-b", "page_num": 1, "case_ids": ["TCD 001/2024"], "issue_date": None, "judges": ["Chief Justice Wayne Martin"], "parties": [], "claim_numbers": [], "is_first_page": True, "is_last_page": False},
        ],
    )
    plan = QuestionPlan(
        mode="judge_compare",
        answer_type="boolean",
        case_ids=("CA 005/2025", "TCD 001/2024"),
        page_hint="first",
        compare_op="set_overlap",
        target_field="judge",
    )

    resolution = resolve_judge_compare(plan, store)

    assert resolution is not None
    assert resolution.answer is True



def test_resolve_party_compare_ignores_role_prefix_during_overlap(tmp_path) -> None:
    store = _write_metadata(
        tmp_path,
        [
            {"doc_id": "doc-a", "page_num": 1, "case_ids": ["CA 004/2025"], "issue_date": None, "judges": [], "parties": ["Claimant: Alpha LLC"], "claim_numbers": [], "is_first_page": True, "is_last_page": False},
            {"doc_id": "doc-b", "page_num": 1, "case_ids": ["SCT 514/2025"], "issue_date": None, "judges": [], "parties": ["Defendant: Alpha LLC"], "claim_numbers": [], "is_first_page": True, "is_last_page": False},
        ],
    )
    plan = QuestionPlan(
        mode="party_compare",
        answer_type="boolean",
        case_ids=("CA 004/2025", "SCT 514/2025"),
        page_hint="first",
        compare_op="set_overlap",
        target_field="party",
    )

    resolution = resolve_party_compare(plan, store)

    assert resolution is not None
    assert resolution.answer is True
