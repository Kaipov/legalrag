from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvidencePage:
    doc_id: str
    page_num: int


@dataclass
class Resolution:
    answer: Any
    evidence_pages: list[EvidencePage]
    confidence: float
    method: str
    facts: dict[str, Any] = field(default_factory=dict)
