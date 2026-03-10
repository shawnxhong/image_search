from datetime import datetime, timezone

from image_search_app.tools.time_parser import TimeParser


def test_parse_last_year():
    parser = TimeParser()
    now = datetime(2026, 6, 15, tzinfo=timezone.utc)
    tr = parser.parse("find photos from last year", now=now)
    assert tr is not None
    assert tr.start.year == 2025
    assert tr.end.year == 2025
