from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.config import PAGE_METADATA_JSONL


class PageMetadataStore:
    def __init__(self, page_metadata_path: Path | str | None = None):
        self.page_metadata_path = Path(page_metadata_path) if page_metadata_path else PAGE_METADATA_JSONL
        self.records: list[dict[str, Any]] = []
        self.records_by_case_id: dict[str, list[dict[str, Any]]] = {}

        with open(self.page_metadata_path, "r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                self.records.append(record)
                for case_id in record.get("case_ids", []):
                    self.records_by_case_id.setdefault(str(case_id), []).append(record)

        for case_id, records in self.records_by_case_id.items():
            self.records_by_case_id[case_id] = sorted(records, key=lambda item: (int(item.get("page_num") or 0), str(item.get("doc_id") or "")))

    def get_case_records(self, case_id: str, *, page_hint: str = "any") -> list[dict[str, Any]]:
        records = list(self.records_by_case_id.get(str(case_id), []))
        if not records:
            return []
        if page_hint == "first":
            first_records = [record for record in records if bool(record.get("is_first_page")) or int(record.get("page_num") or 0) == 1]
            return first_records or [record for record in records if int(record.get("page_num") or 0) <= 2]
        if page_hint == "page_2":
            second_page_records = [record for record in records if int(record.get("page_num") or 0) == 2]
            return second_page_records or [record for record in records if int(record.get("page_num") or 0) <= 2]
        if page_hint == "last":
            last_records = [record for record in records if bool(record.get("is_last_page"))]
            return last_records or records[-1:]
        return records


@lru_cache(maxsize=1)
def load_default_metadata_store() -> PageMetadataStore | None:
    path = PAGE_METADATA_JSONL
    if not path.exists():
        return None
    return PageMetadataStore(path)
