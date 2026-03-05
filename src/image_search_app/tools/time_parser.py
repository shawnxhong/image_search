from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class TimeRange:
    start: datetime
    end: datetime
    source_text: str


class TimeParser:
    """Minimal parser for common phrases; expand in later iteration."""

    def parse(self, text: str, now: datetime | None = None) -> TimeRange | None:
        now = now or datetime.now(timezone.utc)
        lowered = text.lower()

        if "last year" in lowered:
            year = now.year - 1
            return TimeRange(
                start=datetime(year, 1, 1, tzinfo=timezone.utc),
                end=datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
                source_text="last year",
            )
        return None
