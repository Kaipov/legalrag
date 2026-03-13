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
