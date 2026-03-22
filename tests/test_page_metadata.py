from __future__ import annotations

import json

from src.preprocess import page_metadata as metadata_mod



def test_build_page_metadata_records_extracts_page_local_fields(tmp_path) -> None:
    pages_path = tmp_path / "pages.jsonl"
    records = [
        {
            "doc_id": "doc-a",
            "page_num": 1,
            "text": "COURT OF FIRST INSTANCE\nCase No. CA 009/2024\nBefore: Justice Alice Example\nBETWEEN\nAlpha LLC\nClaimant\nand\nBeta Ltd\nDefendant",
        },
        {
            "doc_id": "doc-a",
            "page_num": 2,
            "text": "Date of Issue: 2 February 2026\nClaim No. ENF-316-2023/2\nThe claim value was AED 405,351,504.",
        },
        {
            "doc_id": "doc-a",
            "page_num": 3,
            "text": "IT IS HEREBY ORDERED THAT the application is dismissed with no order as to costs.",
        },
    ]
    with open(pages_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    page_records = metadata_mod.build_page_metadata_records(pages_path)

    assert len(page_records) == 3
    first_page = page_records[0]
    second_page = page_records[1]
    last_page = page_records[2]

    assert first_page["case_ids"] == ["CA 009/2024"]
    assert "Justice Alice Example" in first_page["judges"]
    assert first_page["parties"] == ["Claimant: Alpha LLC", "Defendant: Beta Ltd"]
    assert second_page["case_ids"] == ["CA 009/2024"]
    assert second_page["issue_date"] == "2026-02-02"
    assert second_page["claim_numbers"] == ["ENF-316-2023/2"]
    assert second_page["money_values"] == [405351504]
    assert last_page["is_last_page"] is True
    assert "dismissed" in last_page["order_signals"]
    assert "no order as to costs" in last_page["order_signals"]



def test_extract_issue_date_prefers_explicit_date_of_issue_label() -> None:
    text = (
        "IT IS HEREBY ORDERED THAT: "
        "Issued by: Delvin Sumo SCT Judge and Assistant Registrar "
        "Date of Issue: 24 December 2025 At: 9am "
        "The Judgment of SCT Judge Hayley Norton was issued on 24 October 2025."
    )

    assert metadata_mod.extract_issue_date(text) == "2025-12-24"


def test_build_case_metadata_groups_pages_by_case_id() -> None:
    page_records = [
        {
            "doc_id": "doc-a",
            "page_num": 1,
            "doc_title": "Example",
            "is_first_page": True,
            "is_last_page": False,
            "case_ids": ["CA 009/2024"],
            "issue_date": None,
            "judges": ["Justice Alice Example"],
            "parties": [],
            "claim_numbers": [],
            "money_values": [],
            "order_signals": [],
        },
        {
            "doc_id": "doc-a",
            "page_num": 2,
            "doc_title": "Example",
            "is_first_page": False,
            "is_last_page": False,
            "case_ids": ["CA 009/2024"],
            "issue_date": "2026-02-02",
            "judges": [],
            "parties": [],
            "claim_numbers": ["ENF-316-2023/2"],
            "money_values": [],
            "order_signals": [],
        },
    ]

    case_metadata = metadata_mod.build_case_metadata(page_records)

    assert list(case_metadata.keys()) == ["CA 009/2024"]
    assert case_metadata["CA 009/2024"]["doc_ids"] == ["doc-a"]
    assert [page["page_num"] for page in case_metadata["CA 009/2024"]["pages"]] == [1, 2]


def test_extract_case_ids_normalizes_hyphenated_formats() -> None:
    assert metadata_mod.extract_case_ids("Order in ENF-022-2023 and TCD 003/2022") == [
        "ENF 022/2023",
        "TCD 003/2022",
    ]


def test_extract_case_ids_normalizes_slash_after_prefix_formats() -> None:
    assert metadata_mod.extract_case_ids("Order in ARB/031/2025 and SCT 169/2025") == [
        "ARB 031/2025",
        "SCT 169/2025",
    ]


def test_extract_judges_handles_single_line_case_management_and_judgment_headers() -> None:
    assert metadata_mod.extract_judges(
        "TCD 001/2023 CASE MANAGEMENT ORDER OF H.E. JUSTICE MAHA AL MHEIRI UPON reviewing the Court file"
    ) == ["Justice Maha Al Mheiri"]
    assert metadata_mod.extract_judges(
        "Bond v TR88 [2023] DIFC TCD 001 JUDGMENT OF H.E. JUSTICE WAYNE MARTIN Trial : 12 February 2024"
    ) == ["Justice Wayne Martin"]


def test_extract_judges_prefers_heading_panel_over_referenced_upon_judges() -> None:
    text = (
        "TCD 001/2024 ORDER WITH REASONS OF H.E. JUSTICE ROGER STEWART "
        "UPON the Case Management Order of H.E. Justice Maha Al Mheiri of 8 October 2024 "
        "AND UPON hearing Counsel at the Pre-Trial Review before H.E. Justice Roger Stewart on 2 June 2025"
    )

    assert metadata_mod.extract_judges(text) == ["Justice Roger Stewart"]


def test_extract_judges_cleans_held_before_panel_suffixes() -> None:
    text = (
        "ORDER OF THE COURT OF APPEAL "
        "AND UPON hearing Counsel at the Appeal hearing held before "
        "H.E. Chief Justice Wayne Martin, H.E. Justice Rene Le Miere and "
        "H.E. Justice Sir Peter Gross (the \"Hearing\") "
        "IT IS HEREBY ORDERED THAT..."
    )

    assert metadata_mod.extract_judges(text) == [
        "Chief Justice Wayne Martin",
        "Justice Rene Le Miere",
        "Justice Sir Peter Gross",
    ]
