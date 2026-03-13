from __future__ import annotations

import re


def verbalize_field_answer(field: str | None, value, *, question_text: str = "") -> str:
    if field == "claim_number":
        return f"The claim number is {value}."
    if field == "issue_date":
        return f"The Date of Issue is {value}."
    if field == "judge":
        return f"The judge is {value}."
    if field == "party":
        return f"The party is {value}."
    if field == "law_number":
        return f"The official law number is {value}."
    return str(value)


_OUTCOME_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^the request for an oral hearing(?: of the application)? is refused$", re.IGNORECASE), "the request for an oral hearing was refused"),
    (re.compile(r"^the application is refused$", re.IGNORECASE), "the application was refused"),
    (re.compile(r"^the application is dismissed$", re.IGNORECASE), "the application was dismissed"),
    (re.compile(r"^the application is granted$", re.IGNORECASE), "the application was granted"),
    (re.compile(r"^the appeal is dismissed$", re.IGNORECASE), "the appeal was dismissed"),
    (re.compile(r"^the appeal is allowed$", re.IGNORECASE), "the appeal was allowed"),
    (re.compile(r"^the applicant may not request that the decision be reconsidered at a hearing$", re.IGNORECASE), "reconsideration at a hearing was barred"),
    (re.compile(r"^the applicant shall bear its own costs(?: of the application)?$", re.IGNORECASE), "the Applicant had to bear its own costs"),
    (re.compile(r"^there shall be no order as to costs$", re.IGNORECASE), "there was no order as to costs"),
)


def _normalize_outcome_clause(clause: str, *, is_first: bool) -> str:
    normalized = re.sub(r"\s+", " ", str(clause or "")).strip().rstrip(".;:")
    if not normalized:
        return ""

    rendered = normalized
    for pattern, replacement in _OUTCOME_REPLACEMENTS:
        if pattern.match(normalized):
            rendered = replacement
            break

    if is_first:
        return rendered[0].upper() + rendered[1:] if rendered else rendered
    if rendered.startswith("The "):
        return "the " + rendered[4:]
    return rendered



def verbalize_outcome_clauses(clauses: list[str], *, question_text: str = "") -> str:
    cleaned: list[str] = []
    seen: set[str] = set()
    for clause in clauses:
        normalized_key = re.sub(r"\s+", " ", str(clause or "")).strip().casefold().rstrip(".;:")
        if not normalized_key or normalized_key in seen:
            continue
        seen.add(normalized_key)
        cleaned.append(str(clause))

    if not cleaned:
        return verbalize_absence(question_text)

    rendered = [
        _normalize_outcome_clause(clause, is_first=index == 0)
        for index, clause in enumerate(cleaned)
    ]
    rendered = [clause for clause in rendered if clause]
    if not rendered:
        return verbalize_absence(question_text)
    if len(rendered) == 1:
        return rendered[0].rstrip(".") + "."
    if len(rendered) == 2:
        return f"{rendered[0]}, and {rendered[1].rstrip('.')} .".replace(" .", ".")
    return f"{', '.join(part.rstrip('.') for part in rendered[:-1])}, and {rendered[-1].rstrip('.')} .".replace(" .", ".")



def verbalize_absence(question_text: str) -> str:
    normalized = " ".join(str(question_text or "").split()).strip().rstrip(" ?")
    if not normalized:
        return "The provided DIFC documents do not contain enough information to answer this question."
    return f"The provided DIFC documents do not contain information answering: {normalized}."
