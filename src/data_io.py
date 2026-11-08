"""Validation at the query/document JSONL boundary."""
import json
from pathlib import Path

from pydantic import ValidationError

from .schemas import QueryDoc


def load_query_docs(path: str | Path) -> list[QueryDoc]:
    path = Path(path)
    records = []
    seen = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = QueryDoc.model_validate(json.loads(line))
                if record.qid in seen:
                    raise ValueError(f"duplicate qid: {record.qid}")
            except (ValueError, ValidationError) as exc:
                raise ValueError(f"Invalid query/document at {path}:{line_number}: {exc}") from exc
            seen.add(record.qid)
            records.append(record)
    return records
