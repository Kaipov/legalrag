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
