from src import pipeline as pipeline_mod


def test_generation_top_k_for_structured_types_is_narrower() -> None:
    assert pipeline_mod._generation_top_k_for("number") == 2
    assert pipeline_mod._generation_top_k_for("date") == 2
    assert pipeline_mod._generation_top_k_for("name") == 2
    assert pipeline_mod._generation_top_k_for("names") == 3


def test_generation_top_k_for_free_text_uses_default_limit() -> None:
    assert pipeline_mod._generation_top_k_for("free_text") == pipeline_mod.GENERATION_TOP_K


def test_select_grounding_chunks_prefers_cited_sources() -> None:
    chunks = [
        ({"chunk_id": "c1"}, 0.9),
        ({"chunk_id": "c2"}, 0.8),
        ({"chunk_id": "c3"}, 0.7),
    ]

    selected = pipeline_mod._select_grounding_chunks("free_text", chunks, [3, 1, 3])

    assert selected == [chunks[2], chunks[0]]


def test_select_grounding_chunks_falls_back_conservatively_for_number() -> None:
    chunks = [
        ({"chunk_id": "c1"}, 0.9),
        ({"chunk_id": "c2"}, 0.8),
    ]

    selected = pipeline_mod._select_grounding_chunks("number", chunks, [])

    assert selected == [chunks[0]]
