from src.generate.parse import NULL_MARKER, parse_answer


def test_parse_number_returns_int_for_whole_numbers():
    assert parse_answer("42", "number") == 42


def test_parse_names_splits_semicolon_separated_values():
    assert parse_answer("Alpha; Beta; Gamma", "names") == ["Alpha", "Beta", "Gamma"]


def test_parse_null_marker_returns_none():
    assert parse_answer(f"prefix {NULL_MARKER} suffix", "free_text") is None
