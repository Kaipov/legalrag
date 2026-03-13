from __future__ import annotations

from src.resolve.compare import resolve_date_of_issue_compare, resolve_judge_compare, resolve_party_compare
from src.resolve.metadata_store import load_default_metadata_store
from src.resolve.models import Resolution
from src.resolve.outcome import resolve_last_page_outcome
from src.resolve.page_local import resolve_page_local_lookup
from src.retrieve.question_plan import QuestionPlan



def try_resolve_question(question_item: dict, plan: QuestionPlan) -> Resolution | None:
    if not plan.is_deterministic_candidate:
        return None

    store = load_default_metadata_store()
    if store is None:
        return None

    question_text = str(question_item.get("question") or "")

    if plan.mode == "date_of_issue_compare":
        return resolve_date_of_issue_compare(plan, store)
    if plan.mode in {"page_local_lookup", "title_page_metadata"}:
        return resolve_page_local_lookup(plan, store, question_text=question_text)
    if plan.mode == "judge_compare":
        return resolve_judge_compare(plan, store)
    if plan.mode == "party_compare":
        return resolve_party_compare(plan, store)
    if plan.mode == "last_page_outcome":
        return resolve_last_page_outcome(plan, store, question_text=question_text)
    return None
