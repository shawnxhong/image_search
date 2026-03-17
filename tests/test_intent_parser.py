"""Tests for IntentParser — person extraction, GPS intent, and time range detection."""

from unittest.mock import patch

import pytest

from image_search_app.tools.intent_parser import IntentParser


@pytest.fixture()
def parser():
    """IntentParser with a mocked set of known names from the database."""
    p = IntentParser()
    p._known_names = {"michael powell", "norio ohga", "tom hanks"}
    return p


# --- Person extraction ---

def test_extract_full_name(parser):
    intent = parser.parse_text_query("find photos of Michael Powell")
    assert "michael powell" in intent.people


def test_extract_first_name_of_known_person(parser):
    intent = parser.parse_text_query("find photos with norio")
    assert "norio ohga" in intent.people


def test_no_match_for_unknown_name(parser):
    intent = parser.parse_text_query("find photos of John Smith")
    assert intent.people == []


def test_extract_multiple_people(parser):
    intent = parser.parse_text_query("Michael Powell and Norio Ohga at the park")
    assert len(intent.people) == 2
    names = {n for n in intent.people}
    assert "michael powell" in names
    assert "norio ohga" in names


def test_extract_last_name_of_known_person(parser):
    intent = parser.parse_text_query("find photos with Powell")
    assert "michael powell" in intent.people


def test_case_insensitive_name_match(parser):
    intent = parser.parse_text_query("MICHAEL POWELL on a beach")
    assert "michael powell" in intent.people


def test_empty_db_returns_no_people():
    p = IntentParser()
    p._known_names = set()
    intent = p.parse_text_query("find Michael Powell")
    assert intent.people == []


# --- GPS intent ---

def test_gps_detected_for_beach(parser):
    intent = parser.parse_text_query("photos at the beach")
    assert intent.needs_gps is True


def test_gps_detected_for_location(parser):
    intent = parser.parse_text_query("photos near the park")
    assert intent.needs_gps is True


def test_gps_not_detected_for_plain_text(parser):
    intent = parser.parse_text_query("photos of a cat")
    assert intent.needs_gps is False


def test_gps_keywords_various(parser):
    for keyword in ["outdoor", "mountain", "lake", "city"]:
        intent = parser.parse_text_query(f"photos from the {keyword}")
        assert intent.needs_gps is True, f"Expected GPS intent for '{keyword}'"


# --- Time range ---

def test_time_range_last_year(parser):
    intent = parser.parse_text_query("photos from last year")
    assert intent.time_range is not None
    assert intent.time_range.source_text == "last year"


def test_no_time_range_for_plain_query(parser):
    intent = parser.parse_text_query("cute puppies")
    assert intent.time_range is None


# --- Mode ---

def test_text_mode(parser):
    intent = parser.parse_text_query("find a dog")
    assert intent.mode == "text"


def test_image_mode_no_query(parser):
    intent = parser.parse_image_query(None)
    assert intent.mode == "image"
    assert intent.people == []


def test_image_plus_text_mode(parser):
    intent = parser.parse_image_query("Michael Powell at the beach")
    assert intent.mode == "image+text"
    assert "michael powell" in intent.people
    assert intent.needs_gps is True


# --- Combined intent ---

def test_combined_people_time_gps(parser):
    intent = parser.parse_text_query("Tom Hanks at the beach last year")
    assert "tom hanks" in intent.people
    assert intent.needs_gps is True
    assert intent.time_range is not None


# --- Cache invalidation ---

def test_invalidate_cache(parser):
    parser.invalidate_cache()
    assert parser._known_names is None
