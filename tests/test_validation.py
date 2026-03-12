from src.constants import NULL_FREE_TEXT_ANSWER
from src.validation import validate_answer_value, validate_telemetry_payload


def test_validate_answer_value_accepts_null_for_any_type():
    assert validate_answer_value(None, "number") == []


def test_validate_answer_value_checks_boolean_shape():
    assert validate_answer_value("true", "boolean") == ["boolean answer must be true or false"]


def test_validate_telemetry_allows_null_free_text_with_empty_sources():
    payload = {
        "answer": NULL_FREE_TEXT_ANSWER,
        "telemetry": {
            "timing": {"ttft_ms": 10, "tpot_ms": 0, "total_time_ms": 10},
            "retrieval": {"retrieved_chunk_pages": []},
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "model_name": "gpt-4o",
        },
    }

    assert validate_telemetry_payload(payload) == []


def test_validate_telemetry_flags_non_null_answer_without_sources():
    payload = {
        "answer": "Some answer",
        "telemetry": {
            "timing": {"ttft_ms": 10, "tpot_ms": 0, "total_time_ms": 10},
            "retrieval": {"retrieved_chunk_pages": []},
            "usage": {"input_tokens": 5, "output_tokens": 3},
            "model_name": "gpt-4o",
        },
    }

    assert validate_telemetry_payload(payload) == ["non-null answer but empty retrieved_chunk_pages"]
