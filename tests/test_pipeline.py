from src import pipeline as pipeline_mod
from src.resolve.models import EvidencePage
from src.retrieve.grounding_policy import GroundingIntent
from src.retrieve.question_plan import QuestionPlan


def test_generation_top_k_for_structured_types_is_narrower() -> None:
    assert pipeline_mod._generation_top_k_for("number") == 2
    assert pipeline_mod._generation_top_k_for("date") == 2
    assert pipeline_mod._generation_top_k_for("name") == 2
    assert pipeline_mod._generation_top_k_for("names") == 3


def test_generation_top_k_for_free_text_uses_default_limit() -> None:
    assert pipeline_mod._generation_top_k_for("free_text") == pipeline_mod.GENERATION_TOP_K


def test_generation_top_k_expands_for_page_local_intent() -> None:
    intent = GroundingIntent(kind="date_of_issue", generation_top_k=3)

    assert pipeline_mod._generation_top_k_for("name", intent=intent) == 3


def test_select_generation_chunks_prefers_unique_docs_when_requested() -> None:
    chunks = [
        ({"chunk_id": "c1", "doc_id": "doc-a"}, 0.9),
        ({"chunk_id": "c2", "doc_id": "doc-a"}, 0.8),
        ({"chunk_id": "c3", "doc_id": "doc-b"}, 0.7),
    ]
    intent = GroundingIntent(kind="judge_compare", prefer_unique_docs=True)

    selected = pipeline_mod._select_generation_chunks(chunks, 2, intent=intent)

    assert selected == [chunks[0], chunks[2]]


def test_select_generation_chunks_compare_prefers_case_coverage() -> None:
    chunks = [
        ({"chunk_id": "body-a", "doc_id": "doc-a", "page_numbers": [3], "text": "ARB 034/2025 procedural history"}, 0.98),
        ({"chunk_id": "first-b", "doc_id": "doc-b", "page_numbers": [1], "text": "CFI 067/2025 title page"}, 0.97),
        ({"chunk_id": "first-a", "doc_id": "doc-c", "page_numbers": [1], "text": "ARB 034/2025 title page"}, 0.96),
    ]
    intent = GroundingIntent(
        kind="judge_compare",
        case_ids=("ARB 034/2025", "CFI 067/2025"),
        prefer_unique_docs=True,
    )

    selected = pipeline_mod._select_generation_chunks(chunks, 2, intent=intent)

    assert selected == [chunks[2], chunks[1]]


def test_should_override_compare_null_when_generic_first_pages_cover_both_cases() -> None:
    chunks = [
        ({"chunk_id": "a", "doc_id": "doc-a", "page_numbers": [1], "text": "ARB 034/2025 title page"}, 0.9),
        ({"chunk_id": "b", "doc_id": "doc-b", "page_numbers": [2], "text": "CFI 067/2025 title page"}, 0.8),
    ]
    intent = GroundingIntent(
        kind="judge_compare",
        case_ids=("ARB 034/2025", "CFI 067/2025"),
        prefer_unique_docs=True,
    )

    assert pipeline_mod._should_override_compare_null("boolean", chunks, intent) is True
    assert pipeline_mod._should_override_compare_null("number", chunks, intent) is False


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

    selected = pipeline_mod._select_grounding_chunks("number", chunks, [], intent=None)

    assert selected == [chunks[0]]


def test_select_grounding_chunks_supplements_page_local_citations() -> None:
    chunks = [
        ({"chunk_id": "c1", "doc_id": "doc-a"}, 0.9),
        ({"chunk_id": "c2", "doc_id": "doc-b"}, 0.8),
        ({"chunk_id": "c3", "doc_id": "doc-c"}, 0.7),
    ]
    intent = GroundingIntent(kind="date_of_issue", prefer_unique_docs=True, grounding_chunk_top_k=2)

    selected = pipeline_mod._select_grounding_chunks("name", chunks, [2], intent=intent)

    assert selected == [chunks[1], chunks[0]]


def test_answer_question_passes_answer_type_into_collect_grounding_pages(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeRetriever:
        def retrieve(self, query: str, rerank_top_k: int | None = None, intent: GroundingIntent | None = None):
            return [({"chunk_id": "c1", "doc_id": "doc-a", "page_numbers": [11], "text": "Article 16(1)"}, 0.9)]

    class FakeTokenizer:
        def encode(self, text: str) -> list[str]:
            return text.split()

    def fake_collect_grounding_pages(*args, **kwargs):
        captured["answer_type"] = kwargs.get("answer_type")
        captured["intent_kind"] = getattr(kwargs.get("intent"), "kind", None)
        return [{"doc_id": "doc-a", "page_numbers": [11]}]

    monkeypatch.setattr(pipeline_mod, "HybridRetriever", FakeRetriever)
    monkeypatch.setattr(pipeline_mod, "try_resolve_question", lambda question_item, plan: None)
    monkeypatch.setattr(
        pipeline_mod,
        "detect_grounding_intent",
        lambda question, answer_type: GroundingIntent(kind="article_ref", article_refs=("Article 16(1)",)),
    )
    monkeypatch.setattr(pipeline_mod, "_get_tokenizer", lambda: FakeTokenizer())
    monkeypatch.setattr(pipeline_mod, "detect_null", lambda question, answer_type, chunks, reranker_threshold: (False, None))
    monkeypatch.setattr(
        pipeline_mod,
        "build_prompt",
        lambda question, answer_type, chunks, max_chunks=None, intent=None: [{"role": "user", "content": "prompt"}],
    )
    monkeypatch.setattr(pipeline_mod, "stream_generate", lambda messages: iter(["SOURCES: 1\nANSWER: Confirmation Statement"]))
    monkeypatch.setattr(
        pipeline_mod,
        "parse_model_output",
        lambda response_text, answer_type, question_text=None: ("Confirmation Statement", [1], "Confirmation Statement"),
    )
    monkeypatch.setattr(pipeline_mod, "collect_grounding_pages", fake_collect_grounding_pages)
    monkeypatch.setattr(pipeline_mod, "select_article_evidence_pages", lambda *args, **kwargs: [])

    pipeline = pipeline_mod.RAGPipeline()
    result = pipeline.answer_question(
        {
            "id": "q-article-answer-type",
            "question": "According to Article 16(1) of the Operating Law 2018, what document must be delivered each year?",
            "answer_type": "name",
        }
    )

    assert captured["answer_type"] == "name"
    assert captured["intent_kind"] == "article_ref"
    assert result.telemetry.retrieval[0].page_numbers == [11]


def test_answer_question_overrides_article_grounding_with_structural_evidence(monkeypatch) -> None:
    class FakeRetriever:
        def retrieve(self, query: str, rerank_top_k: int | None = None, intent: GroundingIntent | None = None):
            return [({"chunk_id": "c1", "doc_id": "doc-a", "page_numbers": [6, 31], "text": "Article 7(3)(j)"}, 0.9)]

    class FakeTokenizer:
        def encode(self, text: str) -> list[str]:
            return text.split()

    monkeypatch.setattr(pipeline_mod, "HybridRetriever", FakeRetriever)
    monkeypatch.setattr(
        pipeline_mod,
        "build_question_plan",
        lambda question, answer_type: QuestionPlan(
            mode="article_lookup",
            answer_type=answer_type,
            article_refs=("Article 7(3)(j)",),
        ),
    )
    monkeypatch.setattr(pipeline_mod, "try_resolve_question", lambda question_item, plan: None)
    monkeypatch.setattr(
        pipeline_mod,
        "detect_grounding_intent",
        lambda question, answer_type: GroundingIntent(kind="article_ref", article_refs=("Article 7(3)(j)",)),
    )
    monkeypatch.setattr(pipeline_mod, "_get_tokenizer", lambda: FakeTokenizer())
    monkeypatch.setattr(pipeline_mod, "detect_null", lambda question, answer_type, chunks, reranker_threshold: (False, None))
    monkeypatch.setattr(
        pipeline_mod,
        "build_prompt",
        lambda question, answer_type, chunks, max_chunks=None, intent=None: [{"role": "user", "content": "prompt"}],
    )
    monkeypatch.setattr(pipeline_mod, "stream_generate", lambda messages: iter(["SOURCES: 1\nANSWER: true"]))
    monkeypatch.setattr(
        pipeline_mod,
        "parse_model_output",
        lambda response_text, answer_type, question_text=None: (True, [1], "true"),
    )
    monkeypatch.setattr(
        pipeline_mod,
        "collect_grounding_pages",
        lambda *args, **kwargs: [{"doc_id": "doc-a", "page_numbers": [6, 31]}],
    )
    monkeypatch.setattr(
        pipeline_mod,
        "select_article_evidence_pages",
        lambda question_text, answer_type, answer_text="": [EvidencePage(doc_id="doc-a", page_num=6)],
    )

    pipeline = pipeline_mod.RAGPipeline()
    result = pipeline.answer_question(
        {
            "id": "q-article-override",
            "question": "According to Article 7(3)(j) of the Operating Law 2018, can the Registrar delegate its functions?",
            "answer_type": "boolean",
        }
    )

    assert [(ref.doc_id, ref.page_numbers) for ref in result.telemetry.retrieval] == [("doc-a", [6])]


def test_answer_question_retries_compare_null_without_intent(monkeypatch) -> None:
    compare_intent = GroundingIntent(kind="judge_compare", prefer_unique_docs=True)
    retrieval_calls: list[str] = []

    class FakeRetriever:
        def retrieve(self, query: str, rerank_top_k: int | None = None, intent: GroundingIntent | None = None):
            retrieval_calls.append(intent.kind if intent is not None else "generic")
            chunk_id = "generic" if intent is None else "compare"
            doc_id = "doc-generic" if intent is None else "doc-compare"
            return [({"chunk_id": chunk_id, "doc_id": doc_id, "page_numbers": [1], "text": chunk_id}, 0.9)]

    class FakeTokenizer:
        def encode(self, text: str) -> list[str]:
            return text.split()

    monkeypatch.setattr(pipeline_mod, "HybridRetriever", FakeRetriever)
    monkeypatch.setattr(pipeline_mod, "try_resolve_question", lambda question_item, plan: None)
    monkeypatch.setattr(pipeline_mod, "detect_grounding_intent", lambda question, answer_type: compare_intent)
    monkeypatch.setattr(pipeline_mod, "_get_tokenizer", lambda: FakeTokenizer())
    monkeypatch.setattr(
        pipeline_mod,
        "detect_null",
        lambda question, answer_type, chunks, reranker_threshold: (chunks[0][0]["chunk_id"] == "compare", "compare-null"),
    )
    monkeypatch.setattr(pipeline_mod, "stream_generate", lambda messages: iter(["SOURCES: 1\nANSWER: false"]))
    monkeypatch.setattr(
        pipeline_mod,
        "collect_grounding_pages",
        lambda *args, **kwargs: [{"doc_id": "doc-generic", "page_numbers": [1]}],
    )

    pipeline = pipeline_mod.RAGPipeline()
    result = pipeline.answer_question(
        {
            "id": "q-null-fallback",
            "question": "Did both cases involve any of the same judge?",
            "answer_type": "boolean",
        }
    )

    assert retrieval_calls == ["judge_compare", "generic"]
    assert result.answer is False
    assert result.telemetry.retrieval[0].doc_id == "doc-generic"


def test_answer_question_overrides_compare_null_when_generic_first_pages_cover_both_cases(monkeypatch) -> None:
    compare_intent = GroundingIntent(
        kind="judge_compare",
        case_ids=("ARB 034/2025", "CFI 067/2025"),
        prefer_unique_docs=True,
        generation_top_k=2,
    )
    retrieval_calls: list[str] = []
    prompt_call: dict[str, object] = {}

    class FakeRetriever:
        def retrieve(self, query: str, rerank_top_k: int | None = None, intent: GroundingIntent | None = None):
            retrieval_calls.append(intent.kind if intent is not None else "generic")
            if intent is None:
                return [
                    ({"chunk_id": "body-a", "doc_id": "doc-a", "page_numbers": [3], "text": "ARB 034/2025 procedural history"}, 0.99),
                    ({"chunk_id": "first-b", "doc_id": "doc-b", "page_numbers": [1], "text": "CFI 067/2025 title page"}, 0.98),
                    ({"chunk_id": "first-a", "doc_id": "doc-c", "page_numbers": [1], "text": "ARB 034/2025 title page"}, 0.97),
                ]
            return [({"chunk_id": "compare", "doc_id": "doc-compare", "page_numbers": [4], "text": "judge compare body"}, 0.95)]

    class FakeTokenizer:
        def encode(self, text: str) -> list[str]:
            return text.split()

    def fake_build_prompt(question: str, answer_type: str, chunks, max_chunks=None, intent=None):
        prompt_call["intent_kind"] = intent.kind if intent is not None else "generic"
        prompt_call["chunk_ids"] = [chunk[0]["chunk_id"] for chunk in chunks]
        return [{"role": "user", "content": "compare prompt"}]

    monkeypatch.setattr(pipeline_mod, "HybridRetriever", FakeRetriever)
    monkeypatch.setattr(pipeline_mod, "try_resolve_question", lambda question_item, plan: None)
    monkeypatch.setattr(pipeline_mod, "detect_grounding_intent", lambda question, answer_type: compare_intent)
    monkeypatch.setattr(pipeline_mod, "_get_tokenizer", lambda: FakeTokenizer())
    monkeypatch.setattr(
        pipeline_mod,
        "detect_null",
        lambda question, answer_type, chunks, reranker_threshold: (True, "forced-null"),
    )
    monkeypatch.setattr(pipeline_mod, "build_prompt", fake_build_prompt)
    monkeypatch.setattr(pipeline_mod, "stream_generate", lambda messages: iter(["SOURCES: 1,2\nANSWER: false"]))
    monkeypatch.setattr(
        pipeline_mod,
        "collect_grounding_pages",
        lambda *args, **kwargs: [{"doc_id": "doc-generic", "page_numbers": [1]}],
    )

    pipeline = pipeline_mod.RAGPipeline()
    result = pipeline.answer_question(
        {
            "id": "q-null-override",
            "question": "Is there a judge who presided over both case ARB 034/2025 and case CFI 067/2025?",
            "answer_type": "boolean",
        }
    )

    assert retrieval_calls == ["judge_compare", "generic"]
    assert prompt_call["intent_kind"] == "judge_compare"
    assert prompt_call["chunk_ids"][:2] == ["first-a", "first-b"]
    assert result.answer is False


def test_answer_question_retries_compare_when_grounding_is_empty(monkeypatch) -> None:
    compare_intent = GroundingIntent(kind="party_compare", prefer_unique_docs=True)
    retrieval_calls: list[str] = []
    responses = iter(["SOURCES: 1\nANSWER: false", "SOURCES: 1\nANSWER: true"])

    class FakeRetriever:
        def retrieve(self, query: str, rerank_top_k: int | None = None, intent: GroundingIntent | None = None):
            retrieval_calls.append(intent.kind if intent is not None else "generic")
            chunk_id = "generic" if intent is None else "compare"
            doc_id = "doc-generic" if intent is None else "doc-compare"
            return [({"chunk_id": chunk_id, "doc_id": doc_id, "page_numbers": [1], "text": chunk_id}, 0.9)]

    class FakeTokenizer:
        def encode(self, text: str) -> list[str]:
            return text.split()

    def fake_collect_grounding_pages(reranked_chunks, **kwargs):
        chunk_id = reranked_chunks[0][0]["chunk_id"]
        if chunk_id == "compare":
            return []
        return [{"doc_id": "doc-generic", "page_numbers": [1]}]

    monkeypatch.setattr(pipeline_mod, "HybridRetriever", FakeRetriever)
    monkeypatch.setattr(pipeline_mod, "try_resolve_question", lambda question_item, plan: None)
    monkeypatch.setattr(pipeline_mod, "detect_grounding_intent", lambda question, answer_type: compare_intent)
    monkeypatch.setattr(pipeline_mod, "_get_tokenizer", lambda: FakeTokenizer())
    monkeypatch.setattr(
        pipeline_mod,
        "detect_null",
        lambda question, answer_type, chunks, reranker_threshold: (False, None),
    )
    monkeypatch.setattr(pipeline_mod, "stream_generate", lambda messages: iter([next(responses)]))
    monkeypatch.setattr(pipeline_mod, "collect_grounding_pages", fake_collect_grounding_pages)

    pipeline = pipeline_mod.RAGPipeline()
    result = pipeline.answer_question(
        {
            "id": "q-grounding-fallback",
            "question": "Did both cases involve any of the same parties?",
            "answer_type": "boolean",
        }
    )

    assert retrieval_calls == ["party_compare", "generic"]
    assert result.answer is True
    assert result.telemetry.retrieval[0].doc_id == "doc-generic"


def test_run_generation_pass_limits_page_retrieval_to_generation_docs(monkeypatch) -> None:
    call_kwargs: dict[str, object] = {}

    class FakeTokenizer:
        def encode(self, text: str) -> list[str]:
            return text.split()

    def fake_build_prompt(question: str, answer_type: str, chunks, max_chunks=None, intent=None):
        return [{"role": "user", "content": "prompt"}]

    def fake_collect_grounding_pages(reranked_chunks, **kwargs):
        call_kwargs["allowed_doc_ids"] = kwargs.get("allowed_doc_ids")
        return [{"doc_id": "doc-a", "page_numbers": [1]}]

    monkeypatch.setattr(pipeline_mod, "build_prompt", fake_build_prompt)
    monkeypatch.setattr(pipeline_mod, "stream_generate", lambda messages: iter(["SOURCES: 1\nANSWER: claimed answer"]))
    monkeypatch.setattr(
        pipeline_mod,
        "parse_model_output",
        lambda response_text, answer_type, question_text=None: ("claimed answer", [1], "claimed answer"),
    )
    monkeypatch.setattr(pipeline_mod, "collect_grounding_pages", fake_collect_grounding_pages)

    timer = pipeline_mod.TelemetryTimer()
    pipeline_mod._run_generation_pass(
        "Which claim number is referenced?",
        "name",
        [
            ({"chunk_id": "c1", "doc_id": "doc-a", "page_numbers": [1], "text": "claim text a"}, 0.9),
            ({"chunk_id": "c2", "doc_id": "doc-b", "page_numbers": [2], "text": "claim text b"}, 0.8),
        ],
        FakeTokenizer(),
        timer,
        grounding_threshold=-1.0,
        intent=None,
    )

    assert call_kwargs["allowed_doc_ids"] == {"doc-a", "doc-b"}


def test_answer_question_ttft_is_marked_before_grounding_collection(monkeypatch) -> None:
    class FakeRetriever:
        def retrieve(self, query: str, rerank_top_k: int | None = None, intent: GroundingIntent | None = None):
            return [({"chunk_id": "c1", "doc_id": "doc-a", "page_numbers": [1], "text": "claim text"}, 0.9)]

    class FakeTokenizer:
        def encode(self, text: str) -> list[str]:
            return text.split()

    class FakeTimer:
        def __init__(self) -> None:
            self.marked = False

        def mark_token(self) -> None:
            self.marked = True

        def finish(self):
            return pipeline_mod.TimingMetrics(ttft_ms=123, tpot_ms=0, total_time_ms=999)

    fake_timer = FakeTimer()
    grounding_seen = {"marked": False}

    def fake_collect_grounding_pages(reranked_chunks, **kwargs):
        grounding_seen["marked"] = fake_timer.marked
        return [{"doc_id": "doc-a", "page_numbers": [1]}]

    monkeypatch.setattr(pipeline_mod, "HybridRetriever", FakeRetriever)
    monkeypatch.setattr(pipeline_mod, "try_resolve_question", lambda question_item, plan: None)
    monkeypatch.setattr(pipeline_mod, "TelemetryTimer", lambda: fake_timer)
    monkeypatch.setattr(pipeline_mod, "detect_grounding_intent", lambda question, answer_type: GroundingIntent(kind="generic"))
    monkeypatch.setattr(pipeline_mod, "_get_tokenizer", lambda: FakeTokenizer())
    monkeypatch.setattr(pipeline_mod, "detect_null", lambda question, answer_type, chunks, reranker_threshold: (False, None))
    monkeypatch.setattr(pipeline_mod, "build_prompt", lambda question, answer_type, chunks, max_chunks=None, intent=None: [{"role": "user", "content": "prompt"}])
    monkeypatch.setattr(pipeline_mod, "stream_generate", lambda messages: iter(["SOURCES: 1\nANSWER: example answer"]))
    monkeypatch.setattr(
        pipeline_mod,
        "parse_model_output",
        lambda response_text, answer_type, question_text=None: ("example answer", [1], "example answer"),
    )
    monkeypatch.setattr(pipeline_mod, "collect_grounding_pages", fake_collect_grounding_pages)

    pipeline = pipeline_mod.RAGPipeline()
    result = pipeline.answer_question(
        {
            "id": "q-ttft-grounding",
            "question": "What is the claim number?",
            "answer_type": "name",
        }
    )

    assert grounding_seen["marked"] is True
    assert result.telemetry.timing.ttft_ms == 123
    assert result.telemetry.timing.total_time_ms == 999


def test_free_text_null_answer_uses_question_specific_templates() -> None:
    assert (
        pipeline_mod._free_text_null_answer("On what date was the DIFC Personal Property Law 2005 enacted?")
        == "The provided DIFC documents do not state the enactment date of the DIFC Personal Property Law 2005."
    )
    assert (
        pipeline_mod._free_text_null_answer("Were the Miranda rights properly administered in case ENF 269/2023?")
        == "The provided DIFC documents do not contain information showing whether Miranda rights were properly administered in case ENF 269/2023."
    )


def test_answer_question_uses_question_specific_free_text_abstention(monkeypatch) -> None:
    class FakeRetriever:
        def retrieve(self, query: str, rerank_top_k: int | None = None, intent: GroundingIntent | None = None):
            return []

    class FakeTimer:
        def mark_token(self) -> None:
            return None

        def finish(self):
            return pipeline_mod.TimingMetrics(ttft_ms=12, tpot_ms=0, total_time_ms=12)

    monkeypatch.setattr(pipeline_mod, "HybridRetriever", FakeRetriever)
    monkeypatch.setattr(pipeline_mod, "try_resolve_question", lambda question_item, plan: None)
    monkeypatch.setattr(pipeline_mod, "TelemetryTimer", lambda: FakeTimer())
    monkeypatch.setattr(pipeline_mod, "detect_grounding_intent", lambda question, answer_type: GroundingIntent(kind="generic"))
    monkeypatch.setattr(pipeline_mod, "_get_tokenizer", lambda: None)
    monkeypatch.setattr(pipeline_mod, "detect_null", lambda question, answer_type, chunks, reranker_threshold: (True, "forced-null"))

    pipeline = pipeline_mod.RAGPipeline()
    result = pipeline.answer_question(
        {
            "id": "q-free-text-null",
            "question": "What was the plea bargain in case ARB 034/2025?",
            "answer_type": "free_text",
        }
    )

    assert result.answer == "The provided DIFC documents do not contain information about any plea bargain in case ARB 034/2025."
    assert result.telemetry.retrieval == []


