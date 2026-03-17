from datetime import datetime, timezone

from image_search_app.tools.time_parser import TimeParser


def _now():
    """Fixed reference: Thursday 2026-03-12 10:00 UTC."""
    return datetime(2026, 3, 12, 10, 0, 0, tzinfo=timezone.utc)


parser = TimeParser()


def test_parse_last_year():
    tr = parser.parse("find photos from last year", now=_now())
    assert tr is not None
    assert tr.start == datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert tr.end == datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    assert tr.source_text == "last year"


def test_parse_this_year():
    tr = parser.parse("photos from this year", now=_now())
    assert tr is not None
    assert tr.start.year == 2026
    assert tr.end.year == 2026


def test_parse_last_month():
    tr = parser.parse("taken last month", now=_now())
    assert tr is not None
    assert tr.start == datetime(2026, 2, 1, tzinfo=timezone.utc)
    assert tr.end.day == 28  # Feb 2026 has 28 days


def test_parse_last_month_january_wraps():
    """When now is January, last month should be December of previous year."""
    jan = datetime(2026, 1, 15, tzinfo=timezone.utc)
    tr = parser.parse("last month photos", now=jan)
    assert tr is not None
    assert tr.start == datetime(2025, 12, 1, tzinfo=timezone.utc)
    assert tr.end.month == 12


def test_parse_this_month():
    tr = parser.parse("this month", now=_now())
    assert tr is not None
    assert tr.start == datetime(2026, 3, 1, tzinfo=timezone.utc)
    assert tr.end.day == 31  # March has 31 days


def test_parse_last_week():
    tr = parser.parse("photos from last week", now=_now())
    assert tr is not None
    # 2026-03-12 is Thursday; last week's Monday is 2026-03-02
    assert tr.start == datetime(2026, 3, 2, tzinfo=timezone.utc)
    assert tr.end == datetime(2026, 3, 8, 23, 59, 59, tzinfo=timezone.utc)


def test_parse_this_week():
    tr = parser.parse("this week", now=_now())
    assert tr is not None
    assert tr.start == datetime(2026, 3, 9, tzinfo=timezone.utc)
    assert tr.end == datetime(2026, 3, 15, 23, 59, 59, tzinfo=timezone.utc)


def test_parse_yesterday():
    tr = parser.parse("yesterday's photos", now=_now())
    assert tr is not None
    assert tr.start == datetime(2026, 3, 11, tzinfo=timezone.utc)
    assert tr.end == datetime(2026, 3, 11, 23, 59, 59, tzinfo=timezone.utc)


def test_parse_today():
    tr = parser.parse("photos from today", now=_now())
    assert tr is not None
    assert tr.start == datetime(2026, 3, 12, tzinfo=timezone.utc)
    assert tr.end == datetime(2026, 3, 12, 23, 59, 59, tzinfo=timezone.utc)


def test_parse_in_year():
    tr = parser.parse("photos in 2024", now=_now())
    assert tr is not None
    assert tr.start == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert tr.end == datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc)


def test_parse_bare_year():
    tr = parser.parse("photos from 2023", now=_now())
    assert tr is not None
    assert tr.start.year == 2023


def test_parse_last_n_days():
    tr = parser.parse("last 7 days", now=_now())
    assert tr is not None
    assert tr.start == datetime(2026, 3, 5, tzinfo=timezone.utc)
    assert tr.end.day == 12


def test_parse_last_n_weeks():
    tr = parser.parse("last 2 weeks", now=_now())
    assert tr is not None
    # 14 days before 2026-03-12 = 2026-02-26
    assert tr.start == datetime(2026, 2, 26, tzinfo=timezone.utc)


def test_parse_month_name():
    tr = parser.parse("photos from january", now=_now())
    assert tr is not None
    # Jan is in the past relative to March 2026 → use 2026
    assert tr.start == datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert tr.end.day == 31


def test_parse_month_name_future_uses_last_year():
    """If the named month hasn't occurred yet this year, use previous year."""
    tr = parser.parse("photos from december", now=_now())
    assert tr is not None
    assert tr.start == datetime(2025, 12, 1, tzinfo=timezone.utc)


def test_parse_unrecognized_returns_none():
    tr = parser.parse("find cute puppies", now=_now())
    assert tr is None


def test_parse_empty_returns_none():
    tr = parser.parse("", now=_now())
    assert tr is None
