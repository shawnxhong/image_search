from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass
class TimeRange:
    start: datetime
    end: datetime
    source_text: str


# Month name lookup (lowercased)
_MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
_MONTHS_ABBR = {name.lower(): i for i, name in enumerate(calendar.month_abbr) if name}
_MONTHS.update(_MONTHS_ABBR)


class TimeParser:
    """Parse natural-language time expressions into TimeRange objects."""

    def parse(self, text: str, now: datetime | None = None) -> TimeRange | None:
        now = now or datetime.now(timezone.utc)
        lowered = text.lower()

        # "yesterday"
        if "yesterday" in lowered:
            day = now - timedelta(days=1)
            return TimeRange(
                start=datetime(day.year, day.month, day.day, tzinfo=timezone.utc),
                end=datetime(day.year, day.month, day.day, 23, 59, 59, tzinfo=timezone.utc),
                source_text="yesterday",
            )

        # "today"
        if "today" in lowered:
            return TimeRange(
                start=datetime(now.year, now.month, now.day, tzinfo=timezone.utc),
                end=datetime(now.year, now.month, now.day, 23, 59, 59, tzinfo=timezone.utc),
                source_text="today",
            )

        # "last week"
        if "last week" in lowered:
            # Monday to Sunday of the previous week
            days_since_monday = now.weekday()
            this_monday = now - timedelta(days=days_since_monday)
            last_monday = this_monday - timedelta(days=7)
            last_sunday = last_monday + timedelta(days=6)
            return TimeRange(
                start=datetime(last_monday.year, last_monday.month, last_monday.day, tzinfo=timezone.utc),
                end=datetime(last_sunday.year, last_sunday.month, last_sunday.day, 23, 59, 59, tzinfo=timezone.utc),
                source_text="last week",
            )

        # "this week"
        if "this week" in lowered:
            days_since_monday = now.weekday()
            monday = now - timedelta(days=days_since_monday)
            sunday = monday + timedelta(days=6)
            return TimeRange(
                start=datetime(monday.year, monday.month, monday.day, tzinfo=timezone.utc),
                end=datetime(sunday.year, sunday.month, sunday.day, 23, 59, 59, tzinfo=timezone.utc),
                source_text="this week",
            )

        # "last month"
        if "last month" in lowered:
            if now.month == 1:
                y, m = now.year - 1, 12
            else:
                y, m = now.year, now.month - 1
            last_day = calendar.monthrange(y, m)[1]
            return TimeRange(
                start=datetime(y, m, 1, tzinfo=timezone.utc),
                end=datetime(y, m, last_day, 23, 59, 59, tzinfo=timezone.utc),
                source_text="last month",
            )

        # "this month"
        if "this month" in lowered:
            last_day = calendar.monthrange(now.year, now.month)[1]
            return TimeRange(
                start=datetime(now.year, now.month, 1, tzinfo=timezone.utc),
                end=datetime(now.year, now.month, last_day, 23, 59, 59, tzinfo=timezone.utc),
                source_text="this month",
            )

        # "last year"
        if "last year" in lowered:
            year = now.year - 1
            return TimeRange(
                start=datetime(year, 1, 1, tzinfo=timezone.utc),
                end=datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
                source_text="last year",
            )

        # "this year"
        if "this year" in lowered:
            return TimeRange(
                start=datetime(now.year, 1, 1, tzinfo=timezone.utc),
                end=datetime(now.year, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
                source_text="this year",
            )

        # "last N days/weeks/months"
        m = re.search(r"last\s+(\d+)\s+(day|week|month)s?", lowered)
        if m:
            n = int(m.group(1))
            unit = m.group(2)
            if unit == "day":
                start = now - timedelta(days=n)
            elif unit == "week":
                start = now - timedelta(weeks=n)
            else:  # month — approximate
                start = now - timedelta(days=n * 30)
            return TimeRange(
                start=datetime(start.year, start.month, start.day, tzinfo=timezone.utc),
                end=datetime(now.year, now.month, now.day, 23, 59, 59, tzinfo=timezone.utc),
                source_text=m.group(0),
            )

        # "in YYYY" or just "YYYY" (4-digit year)
        m = re.search(r"\b(?:in\s+)?(\d{4})\b", lowered)
        if m:
            year = int(m.group(1))
            if 1900 <= year <= 2100:
                return TimeRange(
                    start=datetime(year, 1, 1, tzinfo=timezone.utc),
                    end=datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
                    source_text=m.group(0).strip(),
                )

        # "in January", "from march", "december photos", etc.
        for month_name, month_num in _MONTHS.items():
            if len(month_name) < 3:
                continue
            if month_name in lowered:
                # Default to current year; if month is in the future, use last year
                year = now.year
                if month_num > now.month:
                    year -= 1
                last_day = calendar.monthrange(year, month_num)[1]
                return TimeRange(
                    start=datetime(year, month_num, 1, tzinfo=timezone.utc),
                    end=datetime(year, month_num, last_day, 23, 59, 59, tzinfo=timezone.utc),
                    source_text=month_name,
                )

        return None
