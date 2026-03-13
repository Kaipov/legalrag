from __future__ import annotations

import json

from src.resolve.compare import (
    resolve_date_of_issue_compare,
    resolve_judge_compare,
    resolve_monetary_claim_compare,
    resolve_party_compare,
)
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



def test_resolve_date_of_issue_compare_prefers_date_of_issue_text_over_stale_metadata(tmp_path) -> None:
    store = _write_metadata(
        tmp_path,
        [
            {
                "doc_id": "doc-a",
                "page_num": 2,
                "case_ids": ["SCT 169/2025"],
                "issue_date": "2025-10-24",
                "judges": [],
                "parties": [],
                "claim_numbers": [],
                "is_first_page": False,
                "is_last_page": False,
                "text": "Issued by: Delvin Sumo. Date of Issue: 24 December 2025. The Judgment was issued on 24 October 2025.",
            },
            {
                "doc_id": "doc-b",
                "page_num": 2,
                "case_ids": ["SCT 295/2025"],
                "issue_date": "2025-12-10",
                "judges": [],
                "parties": [],
                "claim_numbers": [],
                "is_first_page": False,
                "is_last_page": False,
                "text": "Issued by: Delvin Sumo. Date of Issue: 10 December 2025.",
            },
        ],
    )
    plan = QuestionPlan(
        mode="date_of_issue_compare",
        answer_type="name",
        case_ids=("SCT 169/2025", "SCT 295/2025"),
        page_hint="page_2",
        compare_op="min_date",
        target_field="issue_date",
    )

    resolution = resolve_date_of_issue_compare(plan, store)

    assert resolution is not None
    assert resolution.answer == "SCT 295/2025"
    assert resolution.facts["dates"] == {"SCT 169/2025": "2025-12-24", "SCT 295/2025": "2025-12-10"}


def test_resolve_monetary_claim_compare_returns_higher_case(tmp_path) -> None:
    store = _write_metadata(
        tmp_path,
        [
            {
                "doc_id": "doc-a",
                "page_num": 2,
                "case_ids": ["SCT 169/2025"],
                "issue_date": "2025-12-24",
                "judges": [],
                "parties": [],
                "claim_numbers": [],
                "money_values": [391123.45],
                "is_first_page": False,
                "is_last_page": False,
                "text": "The Claimant filed a Claim with the DIFC Courts' Small Claims Tribunal seeking payment from the Defendant for brokerage services in the amount of AED 391,123.45.",
            },
            {
                "doc_id": "doc-b",
                "page_num": 2,
                "case_ids": ["SCT 295/2025"],
                "issue_date": "2025-12-10",
                "judges": [],
                "parties": [],
                "claim_numbers": [],
                "money_values": [165000, 300162.86],
                "is_first_page": False,
                "is_last_page": False,
                "text": "The Claimant filed his Claim against the Defendant. The Claimant's outstanding Claim at the time of the Judgment was for four months of his basic salary at AED 165,000. A separate penalty request of AED 300,162.86 was also mentioned later.",
            },
        ],
    )
    plan = QuestionPlan(
        mode="monetary_claim_compare",
        answer_type="name",
        case_ids=("SCT 169/2025", "SCT 295/2025"),
        page_hint="page_2",
        compare_op="max_number",
        target_field="money_value",
    )

    resolution = resolve_monetary_claim_compare(plan, store)

    assert resolution is not None
    assert resolution.answer == "SCT 169/2025"
    assert resolution.facts["amounts"] == {"SCT 169/2025": 391123.45, "SCT 295/2025": 165000.0}
    assert [(page.doc_id, page.page_num) for page in resolution.evidence_pages] == [("doc-a", 2), ("doc-b", 2)]



def test_resolve_page_local_lookup_returns_money_value(tmp_path) -> None:
    store = _write_metadata(
        tmp_path,
        [
            {
                "doc_id": "doc-a",
                "page_num": 3,
                "case_ids": ["CA 005/2025"],
                "issue_date": None,
                "judges": [],
                "parties": [],
                "claim_numbers": [],
                "money_values": [999999999, 405351504],
                "is_first_page": False,
                "is_last_page": False,
                "doc_title": "Example",
                "text": (
                    "A costs amount of AED 999,999,999 was mentioned elsewhere. "
                    "The Claimant claims debt or damages of AED 405,351,504, exclusive of interest and costs."
                ),
            },
        ],
    )
    plan = QuestionPlan(
        mode="page_local_lookup",
        answer_type="number",
        case_ids=("CA 005/2025",),
        page_hint="any",
        target_field="money_value",
    )

    resolution = resolve_page_local_lookup(
        plan,
        store,
        question_text="What was the claim value in AED referenced in the appeal judgment CA 005/2025?",
    )

    assert resolution is not None
    assert resolution.answer == 405351504
    assert [(page.doc_id, page.page_num) for page in resolution.evidence_pages] == [("doc-a", 3)]


def test_resolve_page_local_lookup_skips_non_claim_money_values_from_other_doc(tmp_path) -> None:
    store = _write_metadata(
        tmp_path,
        [
            {
                "doc_id": "doc-a",
                "page_num": 3,
                "case_ids": ["CA 005/2025"],
                "issue_date": None,
                "judges": [],
                "parties": [],
                "claim_numbers": [],
                "money_values": [550000],
                "is_first_page": False,
                "is_last_page": True,
                "doc_title": "Short order",
                "text": "assessed and fixed in the amount of AED 550,000, to be paid within 14 days.",
            },
            {
                "doc_id": "doc-b",
                "page_num": 3,
                "case_ids": ["CA 005/2025"],
                "issue_date": None,
                "judges": [],
                "parties": [],
                "claim_numbers": [],
                "money_values": [550000, 405351504],
                "is_first_page": False,
                "is_last_page": False,
                "doc_title": "Appeal reasons",
                "text": (
                    "The Claimant shall pay the Defendant's costs assessed at AED 550,000. "
                    "The Claimant claims debt or damages of AED 405,351,504, exclusive of interest and costs."
                ),
            },
        ],
    )
    plan = QuestionPlan(
        mode="page_local_lookup",
        answer_type="number",
        case_ids=("CA 005/2025",),
        page_hint="any",
        target_field="money_value",
    )

    resolution = resolve_page_local_lookup(
        plan,
        store,
        question_text="What was the claim value in AED referenced in the appeal judgment CA 005/2025?",
    )

    assert resolution is not None
    assert resolution.answer == 405351504
    assert [(page.doc_id, page.page_num) for page in resolution.evidence_pages] == [("doc-b", 3)]


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



def test_resolve_title_page_law_number_without_case_id_uses_document_title_query(tmp_path) -> None:
    store = _write_metadata(
        tmp_path,
        [
            {
                "doc_id": "doc-a",
                "page_num": 1,
                "case_ids": [],
                "issue_date": None,
                "judges": [],
                "parties": [],
                "claim_numbers": [],
                "is_first_page": True,
                "is_last_page": False,
                "doc_title": "COMMON REPORTING",
                "text": "COMMON REPORTING STANDARD LAW\nDIFC LAW NO. 2 OF 2018\nConsolidated Version",
            },
            {
                "doc_id": "doc-b",
                "page_num": 1,
                "case_ids": [],
                "issue_date": None,
                "judges": [],
                "parties": [],
                "claim_numbers": [],
                "is_first_page": True,
                "is_last_page": False,
                "doc_title": "OPERATING",
                "text": "OPERATING LAW\nDIFC LAW NO. 7 OF 2018\nConsolidated Version",
            },
        ],
    )
    plan = QuestionPlan(
        mode="title_page_metadata",
        answer_type="number",
        case_ids=(),
        page_hint="first",
        target_field="law_number",
    )

    resolution = resolve_page_local_lookup(
        plan,
        store,
        question_text="According to the title page of the Common Reporting Standard Law, what is its official DIFC Law number?",
    )

    assert resolution is not None
    assert resolution.answer == 2
    assert [(page.doc_id, page.page_num) for page in resolution.evidence_pages] == [("doc-a", 1)]



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
