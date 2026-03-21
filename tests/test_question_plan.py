from src.retrieve.question_plan import build_question_plan



def test_build_question_plan_detects_date_of_issue_compare() -> None:
    plan = build_question_plan(
        "Which case has an earlier Date of Issue: CA 004/2025 or SCT 295/2025?",
        "name",
    )

    assert plan.mode == "date_of_issue_compare"
    assert plan.page_hint == "front"
    assert plan.target_field == "issue_date"
    assert plan.case_ids == ("CA 004/2025", "SCT 295/2025")



def test_build_question_plan_detects_single_case_date_lookup() -> None:
    plan = build_question_plan(
        "What is the Date of Issue of the document in case CFI 057/2025?",
        "date",
    )

    assert plan.mode == "page_local_lookup"
    assert plan.page_hint == "front"
    assert plan.target_field == "issue_date"



def test_build_question_plan_detects_page_local_claim_number_lookup() -> None:
    plan = build_question_plan(
        "According to page 2 of the judgment, from which specific claim number did the appeal in CA 009/2024 originate?",
        "name",
    )

    assert plan.mode == "page_local_lookup"
    assert plan.page_hint == "page_2"
    assert plan.target_field == "claim_number"
    assert plan.case_ids == ("CA 009/2024",)



def test_build_question_plan_detects_first_page_party_lookup() -> None:
    plan = build_question_plan(
        "From the first page of case CFI 010/2024, who were the claimants?",
        "names",
    )

    assert plan.mode == "page_local_lookup"
    assert plan.page_hint == "first"
    assert plan.target_field == "party"
    assert plan.case_ids == ("CFI 010/2024",)


def test_build_question_plan_detects_single_case_claim_value_lookup() -> None:
    plan = build_question_plan(
        "What was the claim value in AED referenced in the appeal judgment CA 005/2025?",
        "number",
    )

    assert plan.mode == "page_local_lookup"
    assert plan.page_hint == "any"
    assert plan.target_field == "money_value"
    assert plan.case_ids == ("CA 005/2025",)


def test_build_question_plan_detects_judge_compare() -> None:
    plan = build_question_plan(
        "Considering all documents across case CA 005/2025 and case TCD 001/2024, was there any judge who participated in both cases?",
        "boolean",
    )

    assert plan.mode == "judge_compare"
    assert plan.compare_op == "set_overlap"
    assert plan.target_field == "judge"



def test_build_question_plan_detects_monetary_claim_compare() -> None:
    plan = build_question_plan(
        "Identify the case with the higher monetary claim: SCT 169/2025 or SCT 295/2025?",
        "name",
    )

    assert plan.mode == "monetary_claim_compare"
    assert plan.page_hint == "page_2"
    assert plan.compare_op == "max_number"
    assert plan.target_field == "money_value"
    assert plan.case_ids == ("SCT 169/2025", "SCT 295/2025")



def test_build_question_plan_detects_last_page_outcome() -> None:
    plan = build_question_plan(
        "According to the 'IT IS HEREBY ORDERED THAT' section of case SCT 454/2024, what was the outcome of the application for permission to appeal?",
        "free_text",
    )

    assert plan.mode == "last_page_outcome"
    assert plan.page_hint == "last"
    assert plan.is_deterministic_candidate is True



def test_build_question_plan_flags_absence_risk() -> None:
    plan = build_question_plan(
        "Were the Miranda rights properly administered in case ENF 269/2023?",
        "free_text",
    )

    assert plan.mode == "absence_check"
    assert plan.abstention_risk is True



def test_build_question_plan_detects_title_page_law_number_without_case_id() -> None:
    plan = build_question_plan(
        "According to the title page of the Common Reporting Standard Law, what is its official DIFC Law number?",
        "number",
    )

    assert plan.mode == "title_page_metadata"
    assert plan.page_hint == "first"
    assert plan.target_field == "law_number"
    assert plan.case_ids == ()


def test_build_question_plan_routes_title_page_multi_case_party_compare() -> None:
    plan = build_question_plan(
        "From the title pages of all documents in case CA 005/2025 and case CFI 067/2025, identify whether any individual or company is named as a main party in both cases.",
        "boolean",
    )

    assert plan.mode == "party_compare"
    assert plan.page_hint == "first"
    assert plan.compare_op == "set_overlap"
    assert plan.target_field == "party"


def test_build_question_plan_detects_judge_compare_when_question_says_in_common() -> None:
    plan = build_question_plan(
        "Did cases CA 004/2025 and ARB 034/2025 have any judges in common?",
        "boolean",
    )

    assert plan.mode == "judge_compare"
    assert plan.compare_op == "set_overlap"
    assert plan.target_field == "judge"


def test_build_question_plan_detects_party_compare_when_question_says_to_both() -> None:
    plan = build_question_plan(
        "Identify whether any person or company is a main party to both ENF 269/2023 and SCT 514/2025.",
        "boolean",
    )

    assert plan.mode == "party_compare"
    assert plan.compare_op == "set_overlap"
    assert plan.target_field == "party"
