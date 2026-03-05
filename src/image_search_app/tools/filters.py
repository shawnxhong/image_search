from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from image_search_app.db import ImageRecord
from image_search_app.tools.intent_parser import QueryIntent


@dataclass
class FilterOutcome:
    accepted: bool
    matched_constraints: list[str] = field(default_factory=list)
    missing_metadata: list[str] = field(default_factory=list)
    failed_constraints: list[str] = field(default_factory=list)


def apply_hard_filters(record: ImageRecord, intent: QueryIntent) -> FilterOutcome:
    matched: list[str] = []
    missing: list[str] = []
    failed: list[str] = []

    if intent.people:
        names = {p.name.lower() for p in record.people if p.name}
        for person in intent.people:
            if person.lower() in names:
                matched.append(f"person:{person}")
            else:
                failed.append(f"person:{person}")

    if intent.time_range:
        ts: datetime | None = record.capture_timestamp
        if ts is None:
            missing.append("capture_timestamp")
            failed.append("time")
        elif intent.time_range.start <= ts <= intent.time_range.end:
            matched.append("time")
        else:
            failed.append("time")

    if intent.needs_gps:
        if record.lat is None or record.lon is None:
            missing.append("gps")
            failed.append("gps")
        else:
            matched.append("gps")

    return FilterOutcome(accepted=not failed, matched_constraints=matched, missing_metadata=missing, failed_constraints=failed)
