"""Tests for apply_hard_filters — person, time, and GPS constraints."""

from datetime import datetime, timezone

import pytest

from image_search_app.tools.filters import FilterOutcome, apply_hard_filters
from image_search_app.tools.intent_parser import QueryIntent
from image_search_app.tools.time_parser import TimeRange


class FakePerson:
    def __init__(self, name):
        self.name = name


class FakeRecord:
    """Mimics ImageRecord for filter testing without touching the DB."""

    def __init__(
        self,
        capture_timestamp=None,
        lat=None,
        lon=None,
        people=None,
    ):
        self.capture_timestamp = capture_timestamp
        self.lat = lat
        self.lon = lon
        self.people = people or []


# --- No constraints ---

def test_no_constraints_accepted():
    intent = QueryIntent(mode="text")
    record = FakeRecord()
    outcome = apply_hard_filters(record, intent)
    assert outcome.accepted is True
    assert outcome.matched_constraints == []
    assert outcome.failed_constraints == []


# --- Person filter ---

def test_person_matched():
    intent = QueryIntent(mode="text", people=["michael powell"])
    record = FakeRecord(people=[FakePerson("Michael Powell")])
    outcome = apply_hard_filters(record, intent)
    assert outcome.accepted is True
    assert "person:michael powell" in outcome.matched_constraints


def test_person_not_matched():
    intent = QueryIntent(mode="text", people=["john smith"])
    record = FakeRecord(people=[FakePerson("Michael Powell")])
    outcome = apply_hard_filters(record, intent)
    assert outcome.accepted is False
    assert "person:john smith" in outcome.failed_constraints


def test_person_case_insensitive():
    intent = QueryIntent(mode="text", people=["MICHAEL POWELL"])
    record = FakeRecord(people=[FakePerson("michael powell")])
    outcome = apply_hard_filters(record, intent)
    assert outcome.accepted is True


def test_multiple_people_all_present():
    intent = QueryIntent(mode="text", people=["alice", "bob"])
    record = FakeRecord(people=[FakePerson("Alice"), FakePerson("Bob")])
    outcome = apply_hard_filters(record, intent)
    assert outcome.accepted is True
    assert len(outcome.matched_constraints) == 2


def test_multiple_people_one_missing():
    intent = QueryIntent(mode="text", people=["alice", "bob"])
    record = FakeRecord(people=[FakePerson("Alice")])
    outcome = apply_hard_filters(record, intent)
    assert outcome.accepted is False
    assert "person:alice" in outcome.matched_constraints
    assert "person:bob" in outcome.failed_constraints


def test_person_no_people_in_record():
    intent = QueryIntent(mode="text", people=["alice"])
    record = FakeRecord(people=[])
    outcome = apply_hard_filters(record, intent)
    assert outcome.accepted is False


# --- Time filter ---

def test_time_in_range():
    tr = TimeRange(
        start=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end=datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
        source_text="2025",
    )
    intent = QueryIntent(mode="text", time_range=tr)
    record = FakeRecord(capture_timestamp=datetime(2025, 6, 15, tzinfo=timezone.utc))
    outcome = apply_hard_filters(record, intent)
    assert outcome.accepted is True
    assert "time" in outcome.matched_constraints


def test_time_out_of_range():
    tr = TimeRange(
        start=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end=datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
        source_text="2025",
    )
    intent = QueryIntent(mode="text", time_range=tr)
    record = FakeRecord(capture_timestamp=datetime(2024, 6, 15, tzinfo=timezone.utc))
    outcome = apply_hard_filters(record, intent)
    assert outcome.accepted is False
    assert "time" in outcome.failed_constraints


def test_time_missing_timestamp():
    tr = TimeRange(
        start=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end=datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
        source_text="2025",
    )
    intent = QueryIntent(mode="text", time_range=tr)
    record = FakeRecord(capture_timestamp=None)
    outcome = apply_hard_filters(record, intent)
    assert outcome.accepted is False
    assert "capture_timestamp" in outcome.missing_metadata
    assert "time" in outcome.failed_constraints


# --- GPS filter ---

def test_gps_present():
    intent = QueryIntent(mode="text", needs_gps=True)
    record = FakeRecord(lat=40.7128, lon=-74.0060)
    outcome = apply_hard_filters(record, intent)
    assert outcome.accepted is True
    assert "gps" in outcome.matched_constraints


def test_gps_missing():
    intent = QueryIntent(mode="text", needs_gps=True)
    record = FakeRecord(lat=None, lon=None)
    outcome = apply_hard_filters(record, intent)
    assert outcome.accepted is False
    assert "gps" in outcome.failed_constraints
    assert "gps" in outcome.missing_metadata


def test_gps_partial_missing():
    intent = QueryIntent(mode="text", needs_gps=True)
    record = FakeRecord(lat=40.7128, lon=None)
    outcome = apply_hard_filters(record, intent)
    assert outcome.accepted is False


# --- Combined filters ---

def test_all_filters_pass():
    tr = TimeRange(
        start=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end=datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
        source_text="2025",
    )
    intent = QueryIntent(mode="text", people=["alice"], needs_gps=True, time_range=tr)
    record = FakeRecord(
        capture_timestamp=datetime(2025, 6, 15, tzinfo=timezone.utc),
        lat=40.7128,
        lon=-74.0060,
        people=[FakePerson("Alice")],
    )
    outcome = apply_hard_filters(record, intent)
    assert outcome.accepted is True
    assert len(outcome.matched_constraints) == 3


def test_mixed_pass_fail():
    tr = TimeRange(
        start=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end=datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
        source_text="2025",
    )
    intent = QueryIntent(mode="text", people=["alice"], needs_gps=True, time_range=tr)
    record = FakeRecord(
        capture_timestamp=datetime(2025, 6, 15, tzinfo=timezone.utc),
        lat=None,
        lon=None,
        people=[FakePerson("Alice")],
    )
    outcome = apply_hard_filters(record, intent)
    assert outcome.accepted is False
    assert "person:alice" in outcome.matched_constraints
    assert "time" in outcome.matched_constraints
    assert "gps" in outcome.failed_constraints
