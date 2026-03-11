from src.generate.parse import NULL_MARKER, extract_source_ids, parse_answer, parse_model_output


def test_parse_number_returns_int_for_whole_numbers():
    assert parse_answer("42", "number") == 42


def test_parse_names_splits_semicolon_separated_values():
    assert parse_answer("Alpha; Beta; Gamma", "names") == ["Alpha", "Beta", "Gamma"]


def test_parse_null_marker_returns_none():
    assert parse_answer(f"prefix {NULL_MARKER} suffix", "free_text") is None


def test_parse_answer_ignores_sources_and_answer_prefix():
    assert parse_answer("SOURCES: 2\nANSWER: 42", "number") == 42


def test_extract_source_ids_returns_unique_1_based_ids():
    assert extract_source_ids("SOURCES: 3, 1, 3, 2") == [3, 1, 2]


def test_parse_model_output_extracts_answer_text_and_sources():
    answer_value, source_ids, answer_text = parse_model_output(
        "SOURCES: 2,4\nANSWER: Alpha; Beta",
        "names",
    )

    assert answer_value == ["Alpha", "Beta"]
    assert source_ids == [2, 4]
    assert answer_text == "Alpha; Beta"
