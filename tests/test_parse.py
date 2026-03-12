from src.generate.parse import NULL_MARKER, extract_source_ids, parse_answer, parse_model_output


def test_parse_number_returns_int_for_whole_numbers():
    assert parse_answer("42", "number") == 42


def test_parse_names_splits_semicolon_separated_values():
    assert parse_answer("Alpha; Beta; Gamma", "names") == ["Alpha", "Beta", "Gamma"]


def test_parse_names_deduplicates_values():
    assert parse_answer("Alpha; Beta; Alpha", "names") == ["Alpha", "Beta"]


def test_parse_null_marker_returns_none():
    assert parse_answer(f"prefix {NULL_MARKER} suffix", "free_text") is None


def test_parse_answer_ignores_sources_and_answer_prefix():
    assert parse_answer("SOURCES: 2\nANSWER: 42", "number") == 42


def test_extract_source_ids_returns_unique_1_based_ids():
    assert extract_source_ids("SOURCES: 3, 1, 3, 2") == [3, 1, 2]


def test_parse_name_prefers_first_case_mentioned_in_answer():
    assert parse_answer(
        "SCT 295/2025 was issued first.",
        "name",
        question_text="Between ARB 034/2025 and SCT 295/2025, which was issued first?",
    ) == "SCT 295/2025"


def test_parse_name_can_recover_case_from_date_only_segments():
    assert parse_answer(
        "1 July 2025, 10am; 24 December 2025, 9am",
        "name",
        question_text="Which case has an earlier Date of Issue: ENF 269/2023 or SCT 169/2025?",
    ) == "ENF 269/2023"


def test_parse_name_can_recover_case_from_amount_segments():
    assert parse_answer(
        "SCT 169/2025, AED 391,123.45; SCT 295/2025, AED 165,000",
        "name",
        question_text="Identify the case with the higher monetary claim: SCT 169/2025 or SCT 295/2025?",
    ) == "SCT 169/2025"


def test_parse_model_output_extracts_answer_text_and_sources():
    answer_value, source_ids, answer_text = parse_model_output(
        "SOURCES: 2,4\nANSWER: Alpha; Beta",
        "names",
    )

    assert answer_value == ["Alpha", "Beta"]
    assert source_ids == [2, 4]
    assert answer_text == "Alpha; Beta"


def test_parse_model_output_uses_question_text_for_name_cleanup():
    answer_value, source_ids, answer_text = parse_model_output(
        "SOURCES: 1\nANSWER: SCT 295/2025 was issued first.",
        "name",
        question_text="Between ARB 034/2025 and SCT 295/2025, which was issued first?",
    )

    assert answer_value == "SCT 295/2025"
    assert source_ids == [1]
    assert answer_text == "SCT 295/2025 was issued first."


def test_parse_free_text_strips_leading_context_boilerplate():
    assert (
        parse_answer("According to the context, the DIFCA administers the Employment Law.", "free_text")
        == "the DIFCA administers the Employment Law."
    )


def test_parse_free_text_strips_the_answer_is_prefix():
    assert parse_answer("The answer is USD 1,500.", "free_text") == "USD 1,500."