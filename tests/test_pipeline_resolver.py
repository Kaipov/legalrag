from src import pipeline as pipeline_mod
from src.resolve.models import EvidencePage, Resolution



def test_answer_question_uses_deterministic_resolution_before_retrieval(monkeypatch) -> None:
    class FailingRetriever:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def retrieve(self, *args, **kwargs):
            raise AssertionError("retrieval should not be called when deterministic resolution succeeds")

    monkeypatch.setattr(pipeline_mod, "HybridRetriever", FailingRetriever)
    monkeypatch.setattr(
        pipeline_mod,
        "try_resolve_question",
        lambda question_item, plan: Resolution(
            answer="ENF-316-2023/2",
            evidence_pages=[EvidencePage(doc_id="doc-a", page_num=2)],
            confidence=0.99,
            method="page_local_lookup",
        ),
    )

    pipeline = pipeline_mod.RAGPipeline()
    result = pipeline.answer_question(
        {
            "id": "q-deterministic",
            "question": "According to page 2 of the judgment, from which specific claim number did the appeal in CA 009/2024 originate?",
            "answer_type": "name",
        }
    )

    assert result.answer == "ENF-316-2023/2"
    assert result.telemetry.model_name == "deterministic-resolver"
    assert [(ref.doc_id, ref.page_numbers) for ref in result.telemetry.retrieval] == [("doc-a", [2])]


def test_answer_question_marks_ttft_for_deterministic_resolution(monkeypatch) -> None:
    class FailingRetriever:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def retrieve(self, *args, **kwargs):
            raise AssertionError("retrieval should not be called when deterministic resolution succeeds")

    class FakeTimer:
        def __init__(self) -> None:
            self.marked = False

        def mark_token(self) -> None:
            self.marked = True

        def finish(self):
            ttft_ms = 17 if self.marked else 0
            return pipeline_mod.TimingMetrics(ttft_ms=ttft_ms, tpot_ms=0, total_time_ms=17)

    fake_timer = FakeTimer()

    monkeypatch.setattr(pipeline_mod, "HybridRetriever", FailingRetriever)
    monkeypatch.setattr(pipeline_mod, "TelemetryTimer", lambda: fake_timer)
    monkeypatch.setattr(
        pipeline_mod,
        "try_resolve_question",
        lambda question_item, plan: Resolution(
            answer="SCT 169/2025",
            evidence_pages=[EvidencePage(doc_id="doc-a", page_num=2), EvidencePage(doc_id="doc-b", page_num=2)],
            confidence=0.99,
            method="monetary_claim_compare",
        ),
    )

    pipeline = pipeline_mod.RAGPipeline()
    result = pipeline.answer_question(
        {
            "id": "q-deterministic-ttft",
            "question": "Identify the case with the higher monetary claim: SCT 169/2025 or SCT 295/2025?",
            "answer_type": "name",
        }
    )

    assert fake_timer.marked is True
    assert result.telemetry.timing.ttft_ms == 17
    assert result.telemetry.timing.total_time_ms == 17


def test_answer_question_truncates_deterministic_free_text_to_280_chars(monkeypatch) -> None:
    class FailingRetriever:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def retrieve(self, *args, **kwargs):
            raise AssertionError("retrieval should not be called when deterministic resolution succeeds")

    long_answer = " ".join(["The application was dismissed and costs were awarded."] * 12)

    monkeypatch.setattr(pipeline_mod, "HybridRetriever", FailingRetriever)
    monkeypatch.setattr(
        pipeline_mod,
        "try_resolve_question",
        lambda question_item, plan: Resolution(
            answer=long_answer,
            evidence_pages=[EvidencePage(doc_id="doc-a", page_num=2)],
            confidence=0.99,
            method="last_page_outcome",
        ),
    )

    pipeline = pipeline_mod.RAGPipeline()
    result = pipeline.answer_question(
        {
            "id": "q-deterministic-free-text",
            "question": "According to the conclusion section, what was the outcome of the application?",
            "answer_type": "free_text",
        }
    )

    assert isinstance(result.answer, str)
    assert len(result.answer) <= 280
    assert result.answer.endswith("...")
