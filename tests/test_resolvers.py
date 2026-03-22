from __future__ import annotations

import json

from src.resolve.compare import (
    resolve_date_of_issue_compare,
    resolve_judge_compare,
    resolve_monetary_claim_compare,
    resolve_party_compare,
)
from src.resolve.article import select_article_evidence_pages
from src.resolve.metadata_store import PageMetadataStore
from src.resolve.models import EvidencePage
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


def test_resolve_date_of_issue_compare_prefers_first_page_issue_date_when_later_front_pages_only_contain_event_dates(tmp_path) -> None:
    store = _write_metadata(
        tmp_path,
        [
            {
                "doc_id": "doc-enf",
                "page_num": 1,
                "case_ids": ["ENF 269/2023"],
                "issue_date": "2023-09-27",
                "judges": [],
                "parties": [],
                "claim_numbers": [],
                "is_first_page": True,
                "is_last_page": False,
                "text": "ENF 269/2023 ORDER WITH REASONS OF H.E. CHIEF JUSTICE WAYNE MARTIN",
            },
            {
                "doc_id": "doc-enf",
                "page_num": 2,
                "case_ids": ["ENF 269/2023"],
                "issue_date": "2024-10-14",
                "judges": [],
                "parties": [],
                "claim_numbers": [],
                "is_first_page": False,
                "is_last_page": False,
                "text": "UPON the application dated 14 October 2024 under Rule 50.2 of the RDC",
            },
            {
                "doc_id": "doc-sct",
                "page_num": 2,
                "case_ids": ["SCT 295/2025"],
                "issue_date": "2025-12-10",
                "judges": [],
                "parties": [],
                "claim_numbers": [],
                "is_first_page": False,
                "is_last_page": False,
                "text": "Issued by: Delvin Sumo Date of Issue: 10 December 2025",
            },
        ],
    )
    plan = QuestionPlan(
        mode="date_of_issue_compare",
        answer_type="name",
        case_ids=("ENF 269/2023", "SCT 295/2025"),
        page_hint="front",
        compare_op="min_date",
        target_field="issue_date",
    )

    resolution = resolve_date_of_issue_compare(plan, store)

    assert resolution is not None
    assert resolution.answer == "ENF 269/2023"
    assert resolution.facts["dates"] == {"ENF 269/2023": "2023-09-27", "SCT 295/2025": "2025-12-10"}
    assert [(page.doc_id, page.page_num) for page in resolution.evidence_pages] == [("doc-enf", 1), ("doc-sct", 2)]


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


def test_resolve_page_local_lookup_prefers_explicit_date_of_issue_page_over_first_page_metadata(tmp_path) -> None:
    store = _write_metadata(
        tmp_path,
        [
            {
                "doc_id": "doc-a",
                "page_num": 1,
                "case_ids": ["CFI 057/2025"],
                "issue_date": "2026-02-02",
                "judges": [],
                "parties": [],
                "claim_numbers": [],
                "is_first_page": True,
                "is_last_page": False,
                "doc_title": "Example",
                "text": "CFI 057/2025 ORDERS FEBRUARY 02, 2026",
            },
            {
                "doc_id": "doc-a",
                "page_num": 2,
                "case_ids": ["CFI 057/2025"],
                "issue_date": "2026-02-02",
                "judges": [],
                "parties": [],
                "claim_numbers": [],
                "is_first_page": False,
                "is_last_page": False,
                "doc_title": "Example",
                "text": "Issued by: Michael Black KC Date of Issue: 2 February 2026 At: 4pm",
            },
        ],
    )
    plan = QuestionPlan(
        mode="page_local_lookup",
        answer_type="date",
        case_ids=("CFI 057/2025",),
        page_hint="front",
        target_field="issue_date",
    )

    resolution = resolve_page_local_lookup(
        plan,
        store,
        question_text="What is the Date of Issue of the document in case CFI 057/2025?",
    )

    assert resolution is not None
    assert resolution.answer == "2026-02-02"
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


def test_resolve_title_page_parties_keeps_case_file_coverage_pages(tmp_path) -> None:
    store = _write_metadata(
        tmp_path,
        [
            {
                "doc_id": "doc-a",
                "page_num": 1,
                "case_ids": ["TCD 001/2024"],
                "issue_date": None,
                "judges": [],
                "parties": ["Claimant: ARCHITERIORS INTERIOR DESIGN (L.L.C)", "Defendant: ALPHA LLC"],
                "claim_numbers": ["TCD-001/2024"],
                "is_first_page": True,
                "is_last_page": False,
                "doc_title": "Main claim",
                "text": "Claimant ARCHITERIORS INTERIOR DESIGN (L.L.C) Defendant ALPHA LLC",
            },
            {
                "doc_id": "doc-b",
                "page_num": 1,
                "case_ids": ["TCD 001/2024"],
                "issue_date": None,
                "judges": [],
                "parties": ["Claimant: ARCHITERIORS INTERIOR DESIGN (L.L.C)", "Defendant: BETA LLC"],
                "claim_numbers": ["TCD-001/2024"],
                "is_first_page": True,
                "is_last_page": False,
                "doc_title": "Application",
                "text": "Claimant ARCHITERIORS INTERIOR DESIGN (L.L.C) Defendant BETA LLC",
            },
            {
                "doc_id": "doc-c",
                "page_num": 1,
                "case_ids": ["TCD 001/2024"],
                "issue_date": None,
                "judges": [],
                "parties": ["Claimant: ARCHITERIORS INTERIOR DESIGN (L.L.C)", "Defendant: GAMMA LLC"],
                "claim_numbers": ["TCD-001/2024"],
                "is_first_page": True,
                "is_last_page": False,
                "doc_title": "Order",
                "text": "Claimant ARCHITERIORS INTERIOR DESIGN (L.L.C) Defendant GAMMA LLC",
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
    assert [(page.doc_id, page.page_num) for page in resolution.evidence_pages] == [
        ("doc-a", 1),
        ("doc-b", 1),
        ("doc-c", 1),
    ]



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


def test_resolve_last_page_outcome_adds_terminal_page_when_final_outcome_is_restated(tmp_path) -> None:
    store = _write_metadata(
        tmp_path,
        [
            {
                "doc_id": "doc-a",
                "page_num": 1,
                "case_ids": ["ARB 034/2025"],
                "issue_date": None,
                "judges": [],
                "parties": [],
                "claim_numbers": [],
                "is_first_page": True,
                "is_last_page": False,
                "order_signals": [],
                "text": "Opening page.",
            },
            {
                "doc_id": "doc-a",
                "page_num": 2,
                "case_ids": ["ARB 034/2025"],
                "issue_date": None,
                "judges": [],
                "parties": [],
                "claim_numbers": [],
                "is_first_page": False,
                "is_last_page": False,
                "order_signals": ["dismissed", "granted", "costs"],
                "text": (
                    "IT IS HEREBY ORDERED THAT:\n"
                    "1. The ASI Order is discharged with immediate effect.\n"
                    "2. The Defendant's Set Aside Application is granted.\n"
                    "3. The Claimant shall pay the Defendant its costs.\n"
                    "Issued by:\nRegistrar"
                ),
            },
            {
                "doc_id": "doc-a",
                "page_num": 3,
                "case_ids": ["ARB 034/2025"],
                "issue_date": None,
                "judges": [],
                "parties": [],
                "claim_numbers": [],
                "is_first_page": False,
                "is_last_page": False,
                "order_signals": [],
                "text": "Discussion of the background only.",
            },
            {
                "doc_id": "doc-a",
                "page_num": 4,
                "case_ids": ["ARB 034/2025"],
                "issue_date": None,
                "judges": [],
                "parties": [],
                "claim_numbers": [],
                "is_first_page": False,
                "is_last_page": True,
                "order_signals": ["dismissed", "granted"],
                "text": (
                    "20. The appropriate course is therefore to refuse final anti-suit relief and to discharge the ASI Order.\n"
                    "21. For the reasons set out above, the ASI Order is dismissed. The Defendant's Set Aside Application is granted."
                ),
            },
        ],
    )
    plan = QuestionPlan(
        mode="last_page_outcome",
        answer_type="free_text",
        case_ids=("ARB 034/2025",),
        page_hint="last",
        target_field="outcome",
    )

    resolution = resolve_last_page_outcome(
        plan,
        store,
        question_text="According to the 'IT IS HEREBY ORDERED THAT' section of arbitration case ARB 034/2025, what did the court decide?",
    )

    assert resolution is not None
    assert [(page.doc_id, page.page_num) for page in resolution.evidence_pages] == [("doc-a", 2), ("doc-a", 4)]


def test_resolve_last_page_outcome_does_not_add_terminal_page_without_outcome_support(tmp_path) -> None:
    store = _write_metadata(
        tmp_path,
        [
            {
                "doc_id": "doc-a",
                "page_num": 1,
                "case_ids": ["CFI 001/2026"],
                "issue_date": None,
                "judges": [],
                "parties": [],
                "claim_numbers": [],
                "is_first_page": True,
                "is_last_page": False,
                "order_signals": [],
                "text": "Opening page.",
            },
            {
                "doc_id": "doc-a",
                "page_num": 2,
                "case_ids": ["CFI 001/2026"],
                "issue_date": None,
                "judges": [],
                "parties": [],
                "claim_numbers": [],
                "is_first_page": False,
                "is_last_page": False,
                "order_signals": ["dismissed", "costs"],
                "text": (
                    "IT IS HEREBY ORDERED THAT:\n"
                    "1. The Application is dismissed.\n"
                    "2. Costs are awarded to the Respondent.\n"
                    "Issued by:\nRegistrar"
                ),
            },
            {
                "doc_id": "doc-a",
                "page_num": 3,
                "case_ids": ["CFI 001/2026"],
                "issue_date": None,
                "judges": [],
                "parties": [],
                "claim_numbers": [],
                "is_first_page": False,
                "is_last_page": True,
                "order_signals": [],
                "text": "Additional factual background with no operative order language.",
            },
        ],
    )
    plan = QuestionPlan(
        mode="last_page_outcome",
        answer_type="free_text",
        case_ids=("CFI 001/2026",),
        page_hint="last",
        target_field="outcome",
    )

    resolution = resolve_last_page_outcome(
        plan,
        store,
        question_text="According to the 'IT IS HEREBY ORDERED THAT' section, what was the outcome?",
    )

    assert resolution is not None
    assert [(page.doc_id, page.page_num) for page in resolution.evidence_pages] == [("doc-a", 2)]



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
    assert [(page.doc_id, page.page_num) for page in resolution.evidence_pages] == [
        ("doc-a", 1),
        ("doc-b", 1),
    ]


def test_resolve_judge_compare_false_keeps_case_file_coverage_pages(tmp_path) -> None:
    store = _write_metadata(
        tmp_path,
        [
            {"doc_id": "doc-a", "page_num": 1, "case_ids": ["DEC 001/2025"], "issue_date": None, "judges": ["Justice Michael Black KC"], "parties": [], "claim_numbers": [], "is_first_page": True, "is_last_page": False},
            {"doc_id": "doc-b", "page_num": 1, "case_ids": ["DEC 001/2025"], "issue_date": None, "judges": ["Justice Michael Black KC"], "parties": [], "claim_numbers": [], "is_first_page": True, "is_last_page": False},
            {"doc_id": "doc-c", "page_num": 1, "case_ids": ["TCD 001/2024"], "issue_date": None, "judges": ["Chief Justice Wayne Martin"], "parties": [], "claim_numbers": [], "is_first_page": True, "is_last_page": False},
            {"doc_id": "doc-d", "page_num": 1, "case_ids": ["TCD 001/2024"], "issue_date": None, "judges": ["Justice Roger Stewart"], "parties": [], "claim_numbers": [], "is_first_page": True, "is_last_page": False},
        ],
    )
    plan = QuestionPlan(
        mode="judge_compare",
        answer_type="boolean",
        case_ids=("DEC 001/2025", "TCD 001/2024"),
        page_hint="first",
        compare_op="set_overlap",
        target_field="judge",
    )

    resolution = resolve_judge_compare(plan, store)

    assert resolution is not None
    assert resolution.answer is False
    assert [(page.doc_id, page.page_num) for page in resolution.evidence_pages] == [
        ("doc-a", 1),
        ("doc-b", 1),
        ("doc-c", 1),
        ("doc-d", 1),
    ]



def test_resolve_judge_compare_false_prefers_front_matter_page_with_judge_name(tmp_path) -> None:
    store = _write_metadata(
        tmp_path,
        [
            {"doc_id": "doc-a", "page_num": 1, "case_ids": ["DEC 001/2025"], "issue_date": None, "judges": [], "parties": [], "claim_numbers": [], "is_first_page": True, "is_last_page": False, "text": "Opening cover page only."},
            {"doc_id": "doc-a", "page_num": 2, "case_ids": ["DEC 001/2025"], "issue_date": None, "judges": ["Justice Michael Black KC"], "parties": [], "claim_numbers": [], "is_first_page": False, "is_last_page": False, "text": "BEFORE Justice Michael Black KC"},
            {"doc_id": "doc-b", "page_num": 1, "case_ids": ["DEC 001/2025"], "issue_date": None, "judges": ["Justice Michael Black KC"], "parties": [], "claim_numbers": [], "is_first_page": True, "is_last_page": False, "text": "Judgment of Justice Michael Black KC"},
            {"doc_id": "doc-c", "page_num": 1, "case_ids": ["TCD 001/2024"], "issue_date": None, "judges": ["Chief Justice Wayne Martin"], "parties": [], "claim_numbers": [], "is_first_page": True, "is_last_page": False, "text": "BEFORE Chief Justice Wayne Martin"},
            {"doc_id": "doc-d", "page_num": 1, "case_ids": ["TCD 001/2024"], "issue_date": None, "judges": ["Justice Roger Stewart"], "parties": [], "claim_numbers": [], "is_first_page": True, "is_last_page": False, "text": "BEFORE Justice Roger Stewart"},
        ],
    )
    plan = QuestionPlan(
        mode="judge_compare",
        answer_type="boolean",
        case_ids=("DEC 001/2025", "TCD 001/2024"),
        page_hint="front",
        compare_op="set_overlap",
        target_field="judge",
    )

    resolution = resolve_judge_compare(plan, store)

    assert resolution is not None
    assert resolution.answer is False
    assert [(page.doc_id, page.page_num) for page in resolution.evidence_pages] == [
        ("doc-a", 2),
        ("doc-b", 1),
        ("doc-c", 1),
        ("doc-d", 1),
    ]


def test_resolve_judge_compare_true_uses_best_overlap_pair_without_duplicate_case_pages(tmp_path) -> None:
    store = _write_metadata(
        tmp_path,
        [
            {"doc_id": "doc-ca-1", "page_num": 1, "case_ids": ["CA 005/2025"], "issue_date": None, "judges": [". CHIEF JUSTICE WAYNE MARTIN"], "parties": [], "claim_numbers": [], "is_first_page": True, "is_last_page": False, "text": "Interlocutory page."},
            {"doc_id": "doc-ca-2", "page_num": 1, "case_ids": ["CA 005/2025"], "issue_date": None, "judges": ["Chief Justice Wayne Martin", "Justice Roger Stewart"], "parties": [], "claim_numbers": [], "is_first_page": True, "is_last_page": False, "text": "BEFORE H.E. CHIEF JUSTICE WAYNE MARTIN and Justice Roger Stewart"},
            {"doc_id": "doc-tcd", "page_num": 1, "case_ids": ["TCD 001/2024"], "issue_date": None, "judges": ["Chief Justice Wayne Martin"], "parties": [], "claim_numbers": [], "is_first_page": True, "is_last_page": False, "text": "BEFORE Chief Justice Wayne Martin"},
        ],
    )
    plan = QuestionPlan(
        mode="judge_compare",
        answer_type="boolean",
        case_ids=("CA 005/2025", "TCD 001/2024"),
        page_hint="front",
        compare_op="set_overlap",
        target_field="judge",
    )

    resolution = resolve_judge_compare(plan, store)

    assert resolution is not None
    assert resolution.answer is True
    assert [(page.doc_id, page.page_num) for page in resolution.evidence_pages] == [
        ("doc-ca-2", 1),
        ("doc-tcd", 1),
    ]


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
    assert [(page.doc_id, page.page_num) for page in resolution.evidence_pages] == [
        ("doc-a", 1),
        ("doc-b", 1),
    ]


def test_resolve_party_compare_false_keeps_one_page_per_doc_across_both_cases(tmp_path) -> None:
    store = _write_metadata(
        tmp_path,
        [
            {"doc_id": "doc-a", "page_num": 1, "case_ids": ["CA 004/2025"], "issue_date": None, "judges": [], "parties": ["Claimant: Alpha LLC"], "claim_numbers": [], "is_first_page": True, "is_last_page": False, "text": "Claimant Alpha LLC"},
            {"doc_id": "doc-b", "page_num": 1, "case_ids": ["CA 004/2025"], "issue_date": None, "judges": [], "parties": ["Defendant: Beta LLC"], "claim_numbers": [], "is_first_page": True, "is_last_page": False, "text": "Defendant Beta LLC"},
            {"doc_id": "doc-c", "page_num": 1, "case_ids": ["SCT 514/2025"], "issue_date": None, "judges": [], "parties": ["Claimant: Gamma LLC"], "claim_numbers": [], "is_first_page": True, "is_last_page": False, "text": "Claimant Gamma LLC"},
            {"doc_id": "doc-d", "page_num": 1, "case_ids": ["SCT 514/2025"], "issue_date": None, "judges": [], "parties": ["Defendant: Delta LLC"], "claim_numbers": [], "is_first_page": True, "is_last_page": False, "text": "Defendant Delta LLC"},
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
    assert resolution.answer is False
    assert [(page.doc_id, page.page_num) for page in resolution.evidence_pages] == [
        ("doc-a", 1),
        ("doc-b", 1),
        ("doc-c", 1),
        ("doc-d", 1),
    ]


def test_select_article_evidence_pages_prefers_exact_definition_page(tmp_path) -> None:
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
                "doc_title": "OPERATING LAW",
                "text": "OPERATING LAW\nDIFC Law No. 7 of 2018",
                "article_refs": [],
            },
            {
                "doc_id": "doc-a",
                "page_num": 6,
                "case_ids": [],
                "issue_date": None,
                "judges": [],
                "parties": [],
                "claim_numbers": [],
                "is_first_page": False,
                "is_last_page": False,
                "doc_title": "OPERATING LAW",
                "text": "Article 7(3)(j)\nThe Registrar may delegate its functions...",
                "article_refs": ["Article 7(3)(j)"],
            },
            {
                "doc_id": "doc-a",
                "page_num": 31,
                "case_ids": [],
                "issue_date": None,
                "judges": [],
                "parties": [],
                "claim_numbers": [],
                "is_first_page": False,
                "is_last_page": False,
                "doc_title": "OPERATING LAW",
                "text": "Under Article 7(3)(j), administrative matters are discussed elsewhere.",
                "article_refs": ["Article 7(3)(j)"],
            },
        ],
    )

    evidence_pages = select_article_evidence_pages(
        "According to Article 7(3)(j) of the Operating Law 2018, can the Registrar delegate its functions?",
        "boolean",
        answer_text="true",
        store=store,
    )

    assert evidence_pages == [EvidencePage(doc_id="doc-a", page_num=6)]


def test_select_article_evidence_pages_requires_unique_law_match(tmp_path) -> None:
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
                "doc_title": "OPERATING LAW",
                "text": "OPERATING LAW\nDIFC Law No. 7 of 2018",
                "article_refs": [],
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
                "doc_title": "EMPLOYMENT LAW",
                "text": "EMPLOYMENT LAW\nDIFC Law No. 4 of 2019",
                "article_refs": [],
            },
            {
                "doc_id": "doc-a",
                "page_num": 10,
                "case_ids": [],
                "issue_date": None,
                "judges": [],
                "parties": [],
                "claim_numbers": [],
                "is_first_page": False,
                "is_last_page": False,
                "doc_title": "OPERATING LAW",
                "text": "Article 10\nOperating Law clause.",
                "article_refs": ["Article 10"],
            },
            {
                "doc_id": "doc-b",
                "page_num": 10,
                "case_ids": [],
                "issue_date": None,
                "judges": [],
                "parties": [],
                "claim_numbers": [],
                "is_first_page": False,
                "is_last_page": False,
                "doc_title": "EMPLOYMENT LAW",
                "text": "Article 10\nEmployment Law clause.",
                "article_refs": ["Article 10"],
            },
        ],
    )

    evidence_pages = select_article_evidence_pages(
        "According to Article 10, what happens next?",
        "name",
        answer_text="Example",
        store=store,
    )

    assert evidence_pages == []


def test_select_article_evidence_pages_avoids_contents_page_for_structured_lookup(tmp_path) -> None:
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
                "doc_title": "EMPLOYMENT LAW",
                "text": "EMPLOYMENT LAW\nDIFC Law No. 4 of 2019",
                "article_refs": [],
            },
            {
                "doc_id": "doc-a",
                "page_num": 2,
                "case_ids": [],
                "issue_date": None,
                "judges": [],
                "parties": [],
                "claim_numbers": [],
                "is_first_page": False,
                "is_last_page": False,
                "doc_title": "EMPLOYMENT LAW",
                "text": "CONTENTS\n23. Ramadan ........................................ 14",
                "article_refs": [],
            },
            {
                "doc_id": "doc-a",
                "page_num": 14,
                "case_ids": [],
                "issue_date": None,
                "judges": [],
                "parties": [],
                "claim_numbers": [],
                "is_first_page": False,
                "is_last_page": False,
                "doc_title": "EMPLOYMENT LAW",
                "text": (
                    "PART 4: WORKING TIME AND LEAVE\n"
                    "23. Ramadan\n"
                    "During the holy month of Ramadan, a Muslim Employee shall not be required to work in excess of six (6) hours each day."
                ),
                "article_refs": [],
            },
        ],
    )

    evidence_pages = select_article_evidence_pages(
        "If a Muslim Employee works during the holy month of Ramadan, what is the maximum number of hours they can be required to work each day according to Article 23 of the Employment Law 2019?",
        "number",
        answer_text="6",
        store=store,
    )

    assert evidence_pages == [EvidencePage(doc_id="doc-a", page_num=14)]


def test_select_article_evidence_pages_prefers_later_clause_page_when_query_overlap_is_stronger(tmp_path) -> None:
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
                "doc_title": "GENERAL PARTNERSHIP LAW",
                "text": "GENERAL PARTNERSHIP LAW\nDIFC Law No. 11 of 2004",
                "article_refs": [],
            },
            {
                "doc_id": "doc-a",
                "page_num": 5,
                "case_ids": [],
                "issue_date": None,
                "judges": [],
                "parties": [],
                "claim_numbers": [],
                "is_first_page": False,
                "is_last_page": False,
                "doc_title": "GENERAL PARTNERSHIP LAW",
                "text": (
                    "13. Recognised Partnership\n"
                    "(4) An application for registration shall set out the address for service of the Recognised Partnership."
                ),
                "article_refs": [],
            },
            {
                "doc_id": "doc-a",
                "page_num": 8,
                "case_ids": [],
                "issue_date": None,
                "judges": [],
                "parties": [],
                "claim_numbers": [],
                "is_first_page": False,
                "is_last_page": False,
                "doc_title": "GENERAL PARTNERSHIP LAW",
                "text": (
                    "(4) Within six (6) months after the end of the financial year, the accounts for that year shall be prepared and approved by the Partners."
                ),
                "article_refs": [],
            },
        ],
    )

    evidence_pages = select_article_evidence_pages(
        "According to Article 19(4) of the General Partnership Law 2004, how many months after the end of the financial year must the accounts for that year be prepared and approved by the Partners?",
        "number",
        answer_text="6",
        store=store,
    )

    assert evidence_pages == [EvidencePage(doc_id="doc-a", page_num=8)]


def test_select_article_evidence_pages_prefers_target_clause_window_over_unrelated_same_clause_number(tmp_path) -> None:
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
                "doc_title": "OPERATING LAW",
                "text": "OPERATING LAW\nDIFC Law No. 7 of 2018",
                "article_refs": [],
            },
            {
                "doc_id": "doc-a",
                "page_num": 7,
                "case_ids": [],
                "issue_date": None,
                "judges": [],
                "parties": [],
                "claim_numbers": [],
                "is_first_page": False,
                "is_last_page": False,
                "doc_title": "OPERATING LAW",
                "text": (
                    "(7) Subject to Article 7(8), neither the Registrar nor any delegate can be held liable.\n"
                    "(8) Article 7(7) does not apply if the act or omission is shown to have been in bad faith."
                ),
                "article_refs": ["Article 7(8)", "Article 7(7)"],
            },
            {
                "doc_id": "doc-a",
                "page_num": 33,
                "case_ids": [],
                "issue_date": None,
                "judges": [],
                "parties": [],
                "claim_numbers": [],
                "is_first_page": False,
                "is_last_page": False,
                "doc_title": "OPERATING LAW",
                "text": (
                    "(8) The Board shall submit approved financial statements to the President.\n"
                    "Liability\n"
                    "(2) Neither the Registrar nor the DIFCA can be held liable.\n"
                    "(3) Article 58(2) does not apply if the act or omission is shown to have been in bad faith."
                ),
                "article_refs": ["Article 58(2)"],
            },
        ],
    )

    evidence_pages = select_article_evidence_pages(
        "Under the Operating Law 2018, can the Registrar be held liable for acts or omissions in performing their functions if the act or omission is shown to have been in bad faith, according to Article 7(8)?",
        "boolean",
        answer_text="true",
        store=store,
    )

    assert evidence_pages == [EvidencePage(doc_id="doc-a", page_num=7)]


def test_select_article_evidence_pages_prefers_clause_window_over_cross_referenced_next_article(tmp_path) -> None:
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
                "doc_title": "OPERATING LAW",
                "text": "OPERATING LAW\nDIFC Law No. 7 of 2018",
                "article_refs": [],
            },
            {
                "doc_id": "doc-a",
                "page_num": 9,
                "case_ids": [],
                "issue_date": None,
                "judges": [],
                "parties": [],
                "claim_numbers": [],
                "is_first_page": False,
                "is_last_page": False,
                "doc_title": "OPERATING LAW",
                "text": (
                    "Names\n"
                    "(2) A Registered Person shall not use a misleading or deceptive name.\n"
                    "(3) A Registered Person shall, within thirty (30) days, change its name if it becomes misleading, deceptive or conflicting.\n"
                    "(4) A Registered Person is deemed to be aware of those circumstances."
                ),
                "article_refs": [],
            },
            {
                "doc_id": "doc-a",
                "page_num": 10,
                "case_ids": [],
                "issue_date": None,
                "judges": [],
                "parties": [],
                "claim_numbers": [],
                "is_first_page": False,
                "is_last_page": False,
                "doc_title": "OPERATING LAW",
                "text": (
                    "(2) A Registered Person shall file a notification of change of name within thirty (30) days.\n"
                    "(3) Where a Registered Person has complied with the requirement under Article 11(1), the Registrar shall issue a certificate.\n"
                    "(1) Without prejudice to the requirements in Article 10, the Registrar may direct a Registered Person to change its name."
                ),
                "article_refs": ["Article 10", "Article 11(1)", "Article 12(1)"],
            },
        ],
    )

    evidence_pages = select_article_evidence_pages(
        "Under Article 10(3) of the Operating Law 2018, how many days does a Registered Person have to change its name if it becomes misleading, deceptive, or conflicting?",
        "number",
        answer_text="30",
        store=store,
    )

    assert evidence_pages == [EvidencePage(doc_id="doc-a", page_num=9)]
